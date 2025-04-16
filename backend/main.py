from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
import os
import json
import logging
import logging.config
import traceback
import sys
import pandas as pd
from datetime import datetime
import uuid
import secrets
import base64
from pathlib import Path
import shutil
import magic  # For file type validation
import time # <-- Import time module

# Explicit Relative Imports for modules in the same directory
from .job_matcher import JobMatcher
try:
    from .llm_evaluator import LLMEvaluator
except ImportError:
    LLMEvaluator = None
from .resume_parser import ResumeParser
try:
    from .pdf_processor import PDFProcessor
except ImportError:
    PDFProcessor = None
try:
    from .gemini_service import tailor_resume, generate_cover_letter, generate_roadmap as generate_roadmap_suggestions
except ImportError:
    def tailor_resume(*args, **kwargs): return None
    def generate_cover_letter(*args, **kwargs): return None
    def generate_roadmap_suggestions(*args, **kwargs): return None

from dotenv import load_dotenv
from fastapi.responses import FileResponse, JSONResponse

# Load environment variables from .env file in this directory
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))

# --- Path and File Setup ---
backend_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(backend_dir, 'job_title_des.csv')

# Check if CSV exists in backend, if not, check root
if not os.path.exists(csv_file):
    root_csv_file = os.path.join(backend_dir, '..', 'job_title_des.csv')
    if os.path.exists(root_csv_file):
        csv_file = root_csv_file
    else:
        # Create default if not found anywhere
        csv_file = os.path.join(backend_dir, 'job_title_des.csv')
        print(f"Warning: job_title_des.csv not found. Creating default at {csv_file}")
        pd.DataFrame({
            'id': ['sample-id'], 'title': ['Sample Job'], 'company': ['Sample Co'],
            'location': ['Remote'], 'salary': [0], 'description': ['Sample Desc'],
            'requirements': ['None'], 'postedDate': ['2023-01-01']
        }).to_csv(csv_file, index=False)

# --- Initialize components ---
try:
    resume_parser = ResumeParser()
except NameError:
    print("Warning: ResumeParser not initialized (Import failed?)")
    resume_parser = None

try:
    job_matcher = JobMatcher(csv_path=csv_file)
except NameError:
    print("Warning: JobMatcher not initialized (Import failed or TypeError?)")
    job_matcher = None

try:
    llm_evaluator = LLMEvaluator(os.getenv("GEMINI_API_KEY")) if LLMEvaluator else None
except NameError:
     print("Warning: LLMEvaluator not initialized (Import failed?)")
     llm_evaluator = None

# Optional components with try-except for import
try:
    from .resume_improver import ResumeImprover
    resume_improver = ResumeImprover(os.getenv("GEMINI_API_KEY"))
except (ImportError, NameError):
    print("Warning: ResumeImprover not available.")
    resume_improver = None

try:
    from .csv_processor import JobDataProcessor
    csv_processor = JobDataProcessor(csv_file)
except (ImportError, NameError):
    print("Warning: JobDataProcessor not available.")
    csv_processor = None

# --- FastAPI App Setup ---
app = FastAPI(title="CareeroOS API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging Setup
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# Request Logging Middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.monotonic()
    logger.info(f"Request Start: {request.method} {request.url}")
    try:
        response = await call_next(request)
        process_time = time.monotonic() - start_time
        logger.info(
            f"Request End: {request.method} {request.url} - Status: {response.status_code} - Took: {process_time:.4f}s"
        )
        return response
    except Exception as e:
        process_time = time.monotonic() - start_time
        logger.exception(f"Request Failed: {request.method} {request.url} - Took: {process_time:.4f}s - Error: {e}")
        # Reraise the exception to ensure FastAPI handles it correctly
        # Or return a generic error response
        # For now, re-raising is simpler
        raise HTTPException(status_code=500, detail="Internal Server Error")

# --- Pydantic Models ---
class EvaluationRequest(BaseModel):
    gaps: List[str] = []

class AddJobRequest(BaseModel):
    title: str
    description: str
    company: str
    location: str
    salary: Optional[str] = None
    requirements: List[str] = []

# --- API Endpoints ---
@app.get("/")
async def root():
    return {"message": "CareeroOS API"}

@app.post("/process-resume")
async def process_uploaded_resume(file: UploadFile = File(...)):
    start_time = time.monotonic()
    logger.info(f"Starting /process-resume for {file.filename}")
    if not resume_parser:
         raise HTTPException(status_code=500, detail="ResumeParser not available")
    # Simplified processing - assumes ResumeParser handles path correctly
    try:
        # Example: Save temporarily if needed by parser
        temp_path = f"temp_{file.filename}"
        logger.debug("Saving temporary file...")
        save_start = time.monotonic()
        with open(temp_path, "wb") as buffer:
             shutil.copyfileobj(file.file, buffer)
        logger.debug(f"Temporary file saved in {time.monotonic() - save_start:.4f}s")
        result = resume_parser.parse_resume(temp_path)
        os.remove(temp_path)
        end_time = time.monotonic()
        logger.info(f"Finished /process-resume for {file.filename} in {end_time - start_time:.4f}s")
        return result
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /process-resume for {file.filename} after {end_time - start_time:.4f}s")
        # Ensure temp file is removed even on error, if it exists
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_e:
                logger.error(f"Failed to cleanup temp file {temp_path} on error: {cleanup_e}")
        raise HTTPException(status_code=500, detail=f"Error processing resume: {e}")

@app.post("/match-jobs")
async def match_jobs_endpoint(file: UploadFile = File(...)):
    start_time = time.monotonic()
    logger.info(f"Starting /match-jobs for {file.filename}")
    if not job_matcher or not llm_evaluator:
         raise HTTPException(status_code=500, detail="Required components not available")
    # Simplified logic
    try:
        # Example: Save temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
             shutil.copyfileobj(file.file, buffer)

        # Parse resume
        resume_data = resume_parser.parse_resume(temp_path) if resume_parser else {"text": ""}
        resume_text = resume_data.get("text", "")
        os.remove(temp_path)

        matching_jobs = job_matcher.find_matching_jobs(resume_text)
        if not matching_jobs:
            logger.warning(f"No matching jobs found for {file.filename}")
            end_time = time.monotonic()
            logger.info(f"Finished /match-jobs for {file.filename} (no matches) in {end_time - start_time:.4f}s")
            return {"error": "No matching jobs found"}

        # Placeholder evaluation
        evaluation = {"strengths": [], "gaps": [], "overall_fit": "N/A"}
        if llm_evaluator and matching_jobs:
             top_job_details = job_matcher.get_job_details(matching_jobs[0]['id'])
             if top_job_details:
                 logger.debug("Evaluating candidate with LLM...")
                 eval_start = time.monotonic()
                 evaluation = llm_evaluator.evaluate_candidate(resume_text, top_job_details['title'], top_job_details['description'])
                 logger.debug(f"LLM evaluation took {time.monotonic() - eval_start:.4f}s")
             else:
                 logger.warning("Could not get details for top job to perform evaluation.")

        # Ensure data types are serializable
        top_match = matching_jobs[0]
        top_match_details = job_matcher.get_job_details(top_match['id'])
        if top_match_details:
             # Convert potential numpy types if necessary
             serializable_details = {k: (int(v) if isinstance(v, pd.np.integer) else float(v) if isinstance(v, pd.np.floating) else v) for k, v in top_match_details.items()}
        else:
             serializable_details = {"error": "Could not fetch job details"}

        end_time = time.monotonic()
        logger.info(f"Finished /match-jobs for {file.filename} in {end_time - start_time:.4f}s")
        return {
            "top_matching_job": serializable_details,
            "similarity_score": float(top_match['similarity_score']), # Ensure float
            "evaluation": evaluation
        }
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /match-jobs for {file.filename} after {end_time - start_time:.4f}s")
        if 'temp_path' in locals() and os.path.exists(temp_path):
            try: os.remove(temp_path)
            except Exception as cleanup_e: logger.error(f"Failed to cleanup temp file on error: {cleanup_e}")
        raise HTTPException(status_code=500, detail=f"Error matching jobs: {e}")


@app.get("/jobs")
async def get_jobs_endpoint(
    query: Optional[str] = Query(None),
    limit: int = Query(25) # Default limit
):
    start_time = time.monotonic()
    logger.info(f"Starting /jobs query='{query}', limit={limit}")
    if not job_matcher:
         raise HTTPException(status_code=500, detail="JobMatcher not available")
    try:
        # Use job_matcher's internal data or reload if needed
        jobs_df = job_matcher.jobs_df
        if query:
            # Simple text search (case-insensitive)
            mask = jobs_df['title'].str.contains(query, case=False, na=False) | \
                   jobs_df['description'].str.contains(query, case=False, na=False) | \
                   jobs_df['company'].str.contains(query, case=False, na=False)
            filtered_jobs = jobs_df[mask]
        else:
            filtered_jobs = jobs_df

        # Ensure 'similarityScore' exists, add if not
        if 'similarityScore' not in filtered_jobs.columns:
             filtered_jobs['similarityScore'] = 0.0 # Default score

        # Limit and convert to dicts
        results = filtered_jobs.head(limit).to_dict(orient='records')

        # Convert potential numpy types
        serializable_results = []
        serialize_start = time.monotonic()
        for job in results:
            serializable_job = {}
            for k, v in job.items():
                if isinstance(v, (pd.np.integer, pd.np.floating)):
                     serializable_job[k] = float(v) if isinstance(v, pd.np.floating) else int(v)
                elif pd.isna(v):
                     serializable_job[k] = None # Handle NaN
                else:
                     serializable_job[k] = v
            serializable_results.append(serializable_job)
        logger.debug(f"Serialization took {time.monotonic() - serialize_start:.4f}s")

        end_time = time.monotonic()
        logger.info(f"Finished /jobs query='{query}' in {end_time - start_time:.4f}s, returning {len(serializable_results)} results")
        return serializable_results # Return list directly as per frontend expectation

    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /jobs after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error fetching jobs: {e}")


@app.get("/jobs/{job_id}")
async def get_job_endpoint(job_id: str):
     if not job_matcher:
         raise HTTPException(status_code=500, detail="JobMatcher not available")
     try:
        job = job_matcher.get_job_details(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")

        # Convert potential numpy types
        serializable_job = {}
        for k, v in job.items():
             if isinstance(v, (pd.np.integer, pd.np.floating)):
                 serializable_job[k] = float(v) if isinstance(v, pd.np.floating) else int(v)
             elif pd.isna(v):
                 serializable_job[k] = None
             else:
                 serializable_job[k] = v
        return serializable_job

     except Exception as e:
        logger.exception(f"Error fetching job {job_id}")
        raise HTTPException(status_code=500, detail=f"Error fetching job: {e}")


@app.post("/improvement-plan")
async def generate_improvement_plan_endpoint(request: EvaluationRequest):
    if not resume_improver:
        raise HTTPException(status_code=500, detail="ResumeImprover not available")
    # Simplified logic
    try:
        plan = resume_improver.generate_improvement_plan({"gaps": request.gaps})
        return {"improvement_plan": plan}
    except Exception as e:
        logger.exception("Error generating improvement plan")
        raise HTTPException(status_code=500, detail=f"Error generating improvement plan: {e}")

# --- New Endpoints from previous context ---

@app.post("/jobs/{job_id}/match")
async def match_single_job(job_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting single job match for job_id={job_id}")
    if not job_matcher or not resume_parser:
        raise HTTPException(status_code=500, detail="Required components not available")
    try:
        # Assuming the latest resume is needed - logic to find it required here
        # Placeholder: find latest .pdf in uploads
        uploads_dir = os.path.join(backend_dir, '..', 'uploads') # Relative to backend dir
        if not os.path.exists(uploads_dir): os.makedirs(uploads_dir)
        resumes = sorted(
            [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if f.lower().endswith('.pdf')],
            key=os.path.getmtime,
            reverse=True
        )
        if not resumes:
            raise HTTPException(status_code=404, detail="No resume found to match against.")
        latest_resume_path = resumes[0]

        resume_data = resume_parser.parse_resume(latest_resume_path)
        resume_text = resume_data.get("text", "")

        match_start = time.monotonic()
        score = job_matcher.calculate_similarity(resume_text, job_id)
        logger.debug(f"Calculated similarity score in {time.monotonic() - match_start:.4f}s")

        end_time = time.monotonic()
        logger.info(f"Finished single job match for job_id={job_id} in {end_time - start_time:.4f}s")
        return {"job_id": job_id, "similarity_score": float(score)} # Ensure float
    except HTTPException:
        raise
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error matching single job {job_id} after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error matching job {job_id}: {e}")


@app.get("/resumes")
async def get_resumes_endpoint():
    resumes_json_path = os.path.join(backend_dir, '..', 'resumes.json') # Look in root
    if not os.path.exists(resumes_json_path):
        # Fallback or create default
        return {"resumes": []}
    try:
        with open(resumes_json_path, "r") as f:
            resumes_data = json.load(f)
        # Format as needed by frontend
        formatted_resumes = [
            {"id": r.get("id"), "name": r.get("filename"), "uploadedAt": r.get("upload_date"), "size": r.get("size", 0)}
            for r in resumes_data
        ]
        return {"resumes": formatted_resumes}
    except Exception as e:
        logger.exception("Error reading resumes.json")
        return {"resumes": [], "error": str(e)}

@app.get("/resumes/{resume_id}")
async def get_resume_endpoint(resume_id: str):
    resumes_json_path = os.path.join(backend_dir, '..', 'resumes.json')
    if not os.path.exists(resumes_json_path):
        raise HTTPException(status_code=404, detail="Resumes metadata not found")
    try:
        with open(resumes_json_path, "r") as f:
            resumes_data = json.load(f)
        resume_meta = next((r for r in resumes_data if r.get("id") == resume_id), None)
        if not resume_meta:
            raise HTTPException(status_code=404, detail="Resume metadata not found")

        resume_file_path = os.path.join(backend_dir, '..', resume_meta.get("path", "")) # Path relative to root
        if not os.path.exists(resume_file_path):
             raise HTTPException(status_code=404, detail=f"Resume file not found at {resume_file_path}")

        # If PDF, parse it, otherwise return metadata
        if resume_file_path.lower().endswith(".pdf") and resume_parser:
             parsed_data = resume_parser.parse_resume(resume_file_path)
             return {**resume_meta, **parsed_data} # Combine metadata and parsed data
        else:
             # Return basic metadata if not PDF or parser unavailable
             return resume_meta

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error fetching resume {resume_id}")
        raise HTTPException(status_code=500, detail=f"Error fetching resume: {e}")


@app.delete("/resumes/{resume_id}")
async def delete_resume_endpoint(resume_id: str):
    resumes_json_path = os.path.join(backend_dir, '..', 'resumes.json')
    uploads_dir = os.path.join(backend_dir, '..', 'uploads')
    file_deleted = False
    record_updated = False

    if not os.path.exists(resumes_json_path):
        # If no JSON, maybe just try deleting the file directly if pattern matches
        for ext in ['.pdf', '.docx']:
            potential_path = os.path.join(uploads_dir, f"{resume_id}{ext}")
            if os.path.exists(potential_path):
                 try:
                     os.remove(potential_path)
                     logger.info(f"Deleted resume file: {potential_path}")
                     return {"message": "Resume file deleted (metadata not found)"}
                 except Exception as e:
                     logger.exception(f"Error deleting resume file {potential_path}")
                     raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")
        raise HTTPException(status_code=404, detail="Resume metadata and file not found")

    try:
        with open(resumes_json_path, "r") as f:
            resumes_data = json.load(f)

        original_count = len(resumes_data)
        resume_to_delete = None
        updated_resumes = []
        for r in resumes_data:
            if r.get("id") == resume_id:
                resume_to_delete = r
            else:
                updated_resumes.append(r)

        if resume_to_delete:
            # Delete the physical file
            file_path_relative = resume_to_delete.get("path")
            if file_path_relative:
                 file_path_absolute = os.path.join(backend_dir, '..', file_path_relative)
                 if os.path.exists(file_path_absolute):
                     try:
                         os.remove(file_path_absolute)
                         logger.info(f"Deleted resume file: {file_path_absolute}")
                         file_deleted = True
                     except Exception as e:
                         logger.warning(f"Could not delete file {file_path_absolute}: {e}")
                 else:
                     logger.warning(f"File path in JSON does not exist: {file_path_absolute}")
            else:
                 logger.warning(f"No file path found in metadata for resume {resume_id}")

            # Update the JSON file
            with open(resumes_json_path, "w") as f:
                json.dump(updated_resumes, f, indent=2)
            record_updated = True
            logger.info(f"Removed resume record {resume_id} from resumes.json")

            return {"message": "Resume deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Resume record not found in metadata")

    except Exception as e:
        logger.exception(f"Error deleting resume {resume_id}")
        raise HTTPException(status_code=500, detail=f"Error deleting resume: {e}")


@app.post("/upload-resume")
async def upload_resume_endpoint(file: UploadFile = File(...)):
    # Basic file validation (can be enhanced)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx']:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF/DOCX allowed.")

    # Ensure uploads directory exists (relative to root)
    uploads_dir = os.path.join(backend_dir, '..', 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    # Create unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_id = f"resume_{timestamp}"
    file_path = os.path.join(uploads_dir, f"{resume_id}{file_ext}")

    try:
        # Save file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded resume to {file_path}")

        # Update resumes.json (relative to root)
        resumes_json_path = os.path.join(backend_dir, '..', 'resumes.json')
        resumes_data = []
        if os.path.exists(resumes_json_path):
            try:
                with open(resumes_json_path, "r") as f:
                    resumes_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning("resumes.json is corrupted, starting fresh.")

        new_resume_meta = {
            "id": resume_id,
            "filename": file.filename,
            "path": os.path.relpath(file_path, os.path.join(backend_dir, '..')), # Path relative to project root
            "upload_date": datetime.now().isoformat(),
            "size": os.path.getsize(file_path)
        }
        resumes_data.append(new_resume_meta)

        with open(resumes_json_path, "w") as f:
            json.dump(resumes_data, f, indent=2)

        return {"message": "Resume uploaded successfully", "resume_id": resume_id, "file_info": new_resume_meta}

    except Exception as e:
        logger.exception("Error uploading resume")
        # Clean up saved file if error occurred
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=f"Error uploading resume: {e}")


@app.post("/jobs")
async def add_job_endpoint(job: AddJobRequest):
     if not job_matcher:
         raise HTTPException(status_code=500, detail="JobMatcher not available")
     try:
         new_job_data = job.dict()
         new_job_data['id'] = str(uuid.uuid4()) # Generate new ID
         new_job_data['postedDate'] = datetime.now().strftime('%Y-%m-%d')
         # Convert requirements list to semicolon-separated string if needed by JobMatcher
         new_job_data['requirements'] = ";".join(job.requirements)

         # Add to the DataFrame in JobMatcher
         success = job_matcher.add_job(new_job_data)

         if success:
             # Also need to save back to the CSV file
             job_matcher.save_jobs(csv_file)
             # Format for frontend response (convert requirements back to list maybe?)
             response_job = new_job_data.copy()
             response_job['requirements'] = job.requirements # Send back as list
             response_job['similarityScore'] = 0.0 # Default score for new job
             return response_job
         else:
             raise HTTPException(status_code=500, detail="Failed to add job to internal list.")

     except Exception as e:
         logger.exception("Error adding job")
         raise HTTPException(status_code=500, detail=f"Error adding job: {e}")


# --- Gemini Endpoints (Simplified) ---
@app.post("/jobs/{job_id}/tailor-resume")
async def tailor_resume_endpoint(job_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting /tailor-resume for job_id={job_id}")
    if not job_matcher or not resume_parser:
         raise HTTPException(status_code=500, detail="Required components not available")
    try:
        # Find latest resume (logic needed)
        uploads_dir = os.path.join(backend_dir, '..', 'uploads')
        resumes = sorted(
            [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if f.lower().endswith('.pdf')],
            key=os.path.getmtime,
            reverse=True
        )
        if not resumes: raise HTTPException(status_code=404, detail="No resume found")
        resume_path = resumes[0]
        resume_text = resume_parser.parse_resume(resume_path).get("text", "")

        job = job_matcher.get_job_details(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job not found")

        # Call Gemini service (if imported)
        if 'tailor_resume' in globals() and callable(tailor_resume):
             tailored = tailor_resume(resume_text, job['description'])
             end_time = time.monotonic()
             logger.info(f"Finished /tailor-resume for job_id={job_id} in {end_time - start_time:.4f}s")
             return {"tailored_resume": tailored}
        else:
             raise HTTPException(status_code=501, detail="Tailor resume feature not available")
    except HTTPException:
        raise
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /tailor-resume after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error tailoring resume: {e}")

@app.post("/jobs/{job_id}/generate-cover-letter")
async def generate_cover_letter_endpoint(job_id: str):
    # Similar logic to tailor-resume: get resume text, job details
    if not job_matcher or not resume_parser:
        raise HTTPException(status_code=500, detail="Required components not available")
    try:
        uploads_dir = os.path.join(backend_dir, '..', 'uploads')
        resumes = sorted(
            [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if f.lower().endswith('.pdf')],
            key=os.path.getmtime,
            reverse=True
        )
        if not resumes: raise HTTPException(status_code=404, detail="No resume found")
        resume_path = resumes[0]
        resume_text = resume_parser.parse_resume(resume_path).get("text", "")

        job = job_matcher.get_job_details(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job not found")

        # Call Gemini service (if imported)
        if 'generate_cover_letter' in globals() and callable(generate_cover_letter):
             cover_letter = generate_cover_letter(resume_text, job['description'])
             return {"cover_letter": cover_letter}
        else:
             raise HTTPException(status_code=501, detail="Generate cover letter feature not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generating cover letter for job {job_id}")
        raise HTTPException(status_code=500, detail=f"Error generating cover letter: {e}")


@app.post("/jobs/{job_id}/generate-roadmap")
async def generate_roadmap_endpoint(job_id: str):
     # Similar logic: get resume text, job details
    if not job_matcher or not resume_parser:
        raise HTTPException(status_code=500, detail="Required components not available")
    try:
        uploads_dir = os.path.join(backend_dir, '..', 'uploads')
        resumes = sorted(
            [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if f.lower().endswith('.pdf')],
            key=os.path.getmtime,
            reverse=True
        )
        if not resumes: raise HTTPException(status_code=404, detail="No resume found")
        resume_path = resumes[0]
        resume_text = resume_parser.parse_resume(resume_path).get("text", "")

        job = job_matcher.get_job_details(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job not found")

        # Call Gemini service (if imported)
        if 'generate_roadmap_suggestions' in globals() and callable(generate_roadmap_suggestions):
             roadmap = generate_roadmap_suggestions(resume_text, job['description'])
             return {"roadmap": roadmap}
        else:
             raise HTTPException(status_code=501, detail="Generate roadmap feature not available")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generating roadmap for job {job_id}")
        raise HTTPException(status_code=500, detail=f"Error generating roadmap: {e}")


# --- Main Execution Guard ---
if __name__ == "__main__":
    # Read PORT from environment variable, default to 8080 for Render/Cloud Run compatibility
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting server on host 0.0.0.0, port {port}")
    # Disable reload in production deployment
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)