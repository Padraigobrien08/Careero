from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Request, Response
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
import time
import asyncio # <-- Import asyncio
from contextlib import asynccontextmanager

# --- Define backend_dir at module level ---
backend_dir = os.path.dirname(os.path.abspath(__file__))

# --- Logging Setup ---
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())
logger = logging.getLogger(__name__)

# --- Global Variables ---
jobs_data: List[Dict] = []
job_matcher: Optional[Any] = None # Use 'Any' or the actual class type
resume_parser: Optional[Any] = None
llm_evaluator: Optional[Any] = None
resume_improver: Optional[Any] = None
csv_processor: Optional[Any] = None
# Define placeholders for potentially missing modules/classes if needed
JobMatcher = None
LLMEvaluator = None
ResumeParser = None
PDFProcessor = None
ResumeImprover = None
JobDataProcessor = None
gemini_tailor_resume = None
gemini_generate_cover_letter = None
gemini_generate_roadmap = None

# --- Attempt Imports ---
try:
    from .job_matcher import JobMatcher
except ImportError:
    print("Warning: job_matcher not found.")
try:
    from .llm_evaluator import LLMEvaluator
except ImportError:
    print("Warning: llm_evaluator not found.")
try:
    from .resume_parser import ResumeParser
except ImportError:
    print("Warning: resume_parser not found.")
try:
    from .pdf_processor import PDFProcessor
except ImportError:
    print("Warning: pdf_processor not found.")
try:
    from .gemini_service import tailor_resume as gemini_tailor_resume, \
                                generate_cover_letter as gemini_generate_cover_letter, \
                                generate_roadmap as gemini_generate_roadmap
except ImportError:
    print("Warning: gemini_service not found.")
    # Define placeholders if Gemini service is critical but might be missing
    def gemini_tailor_resume(*args, **kwargs): return None
    def gemini_generate_cover_letter(*args, **kwargs): return None
    def gemini_generate_roadmap(*args, **kwargs): return None
try:
    from .resume_improver import ResumeImprover
except ImportError:
    print("Warning: resume_improver not found.")
try:
    from .csv_processor import JobDataProcessor
except ImportError:
    print("Warning: csv_processor not found.")


from dotenv import load_dotenv
from fastapi.responses import FileResponse, JSONResponse

# --- FastAPI App Setup ---
from contextlib import asynccontextmanager

# --- Lifespan Event Handler (Replaces on_event) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Code to run on startup
    global jobs_data, job_matcher, resume_parser, llm_evaluator, resume_improver, csv_processor
    logger.info("Application startup: Initializing components and loading data via lifespan...")
    start_time = time.monotonic()
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '.env'))
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    # Look for CSV *only* within the backend directory
    csv_file = os.path.join(backend_dir, 'job_title_des.csv')

    if not os.path.exists(csv_file):
        logger.error(f"CRITICAL: job_title_des.csv not found in backend directory: {csv_file}")
        # Optionally create default, but log error regardless
        logger.warning(f"Creating default job_title_des.csv at {csv_file}")
        try:
            pd.DataFrame({
                'id': [str(uuid.uuid4())], 'title': ['Sample Job'], 'company': ['Sample Co'],
                'location': ['Remote'], 'salary': [0], 'description': ['Sample Desc'],
                'requirements': ['None'], 'postedDate': [datetime.now().strftime('%Y-%m-%d')]
            }).to_csv(csv_file, index=False)
        except Exception as e:
            logger.error(f"Failed to create default CSV: {e}")
            csv_file = None # Ensure it's None if creation failed
    else:
        logger.info(f"Found job_title_des.csv at: {csv_file}")

    # --- Initialize components ---
    if ResumeParser:
        try:
            resume_parser = ResumeParser()
            logger.info("ResumeParser initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ResumeParser: {e}")

    if JobMatcher and csv_file and os.path.exists(csv_file):
        try:
            # This initialization now calls load_jobs() internally
            job_matcher = JobMatcher(csv_path=csv_file)
            logger.info(f"JobMatcher initialized with data from {csv_file}.")
            # Now correctly check the attribute AFTER initialization
            if hasattr(job_matcher, 'jobs_df') and job_matcher.jobs_df is not None and not job_matcher.jobs_df.empty:
                 # Simplified loading logic (ensure serialization if needed)
                 jobs_data = job_matcher.jobs_df.where(pd.notna(job_matcher.jobs_df), None).to_dict(orient='records')
                 logger.info(f"Loaded {len(jobs_data)} jobs into memory.")
            else:
                 logger.warning("JobMatcher initialized, but its DataFrame is empty or None after load.")
                 jobs_data = []
        except Exception as e:
            logger.error(f"Failed to initialize JobMatcher: {e}")
            jobs_data = []
            job_matcher = None # Ensure job_matcher is None on error

    # Initialize other optional components
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if LLMEvaluator and gemini_api_key:
        try:
            llm_evaluator = LLMEvaluator(gemini_api_key)
            logger.info("LLMEvaluator initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize LLMEvaluator: {e}")
            llm_evaluator = None
    elif LLMEvaluator and not gemini_api_key:
        logger.warning("LLMEvaluator class found, but GEMINI_API_KEY not set.")


    if ResumeImprover and gemini_api_key:
        try:
            resume_improver = ResumeImprover(gemini_api_key)
            logger.info("ResumeImprover initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize ResumeImprover: {e}")
            resume_improver = None
    elif ResumeImprover and not gemini_api_key:
         logger.warning("ResumeImprover class found, but GEMINI_API_KEY not set.")


    if JobDataProcessor and csv_file:
        try:
            csv_processor = JobDataProcessor(csv_file)
            logger.info("JobDataProcessor initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize JobDataProcessor: {e}")
            csv_processor = None
    elif JobDataProcessor and not csv_file:
         logger.warning("JobDataProcessor class found, but CSV file path is invalid.")


    end_time = time.monotonic()
    logger.info(f"Application startup completed in {end_time - start_time:.4f}s")

    yield # Application runs here

    # Code to run on shutdown (optional)
    logger.info("Application shutdown.")


app = FastAPI(title="CareeroOS API", lifespan=lifespan)

# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Request Logging Middleware ---
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
async def root_get():
    """
    Returns the list of jobs loaded into memory at startup (GET request).
    """
    logger.debug("Handling GET /")
    return jobs_data

@app.head("/")
async def root_head():
    """
    Handles HEAD requests for the root path. Returns empty response with headers.
    """
    logger.debug("Handling HEAD /")
    # FastAPI/Starlette usually handles HEAD correctly based on GET,
    # but an explicit empty response is also valid.
    return Response(status_code=200)

@app.post("/process-resume")
async def process_uploaded_resume(file: UploadFile = File(...)):
    start_time = time.monotonic()
    logger.info(f"Starting /process-resume for {file.filename}")
    if not resume_parser:
         raise HTTPException(status_code=500, detail="ResumeParser not available")

    temp_path = f"temp_{uuid.uuid4()}_{file.filename}" # Unique temp name
    try:
        logger.debug("Saving temporary file...")
        save_start = time.monotonic()
        with open(temp_path, "wb") as buffer:
             shutil.copyfileobj(file.file, buffer)
        logger.debug(f"Temporary file saved in {time.monotonic() - save_start:.4f}s")

        logger.debug("Parsing resume...")
        parse_start = time.monotonic()
        result = resume_parser.parse_resume(temp_path)
        logger.debug(f"Resume parsing completed in {time.monotonic() - parse_start:.4f}s")

    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error during /process-resume for {file.filename} after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error processing resume: {e}")
    finally:
        # Ensure temp file is always removed
        if os.path.exists(temp_path):
            try:
                cleanup_start = time.monotonic()
                os.remove(temp_path)
                logger.debug(f"Temporary file cleaned up in {time.monotonic() - cleanup_start:.4f}s")
            except Exception as cleanup_e:
                logger.error(f"Failed to cleanup temp file {temp_path}: {cleanup_e}")

    end_time = time.monotonic()
    logger.info(f"Finished /process-resume for {file.filename} in {end_time - start_time:.4f}s")
    return result


@app.post("/match-jobs")
async def match_jobs_endpoint(file: UploadFile = File(...)):
    start_time = time.monotonic()
    logger.info(f"Starting /match-jobs for {file.filename}")
    if not job_matcher:
         raise HTTPException(status_code=500, detail="JobMatcher not available")
    if not resume_parser:
         raise HTTPException(status_code=500, detail="ResumeParser not available")
    if not llm_evaluator:
         logger.warning("LLMEvaluator not available, proceeding without evaluation.")
         # Decide if this should be an error or just proceed without evaluation
         # raise HTTPException(status_code=500, detail="LLMEvaluator not available")


    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
    try:
        logger.debug("Saving temporary file...")
        save_start = time.monotonic()
        with open(temp_path, "wb") as buffer:
             shutil.copyfileobj(file.file, buffer)
        logger.debug(f"Temp file saved in {time.monotonic() - save_start:.4f}s")

        logger.debug("Parsing resume...")
        parse_start = time.monotonic()
        resume_data = resume_parser.parse_resume(temp_path)
        resume_text = resume_data.get("text", "")
        logger.debug(f"Resume parsing took {time.monotonic() - parse_start:.4f}s")

        logger.debug("Finding matching jobs...")
        match_start = time.monotonic()
        # Use the initialized job_matcher instance
        matching_jobs = job_matcher.find_matching_jobs(resume_text)
        logger.debug(f"Job matching took {time.monotonic() - match_start:.4f}s")

        if not matching_jobs:
            logger.warning(f"No matching jobs found for {file.filename}")
            end_time = time.monotonic()
            logger.info(f"Finished /match-jobs for {file.filename} (no matches) in {end_time - start_time:.4f}s")
            return {"error": "No matching jobs found"}

        evaluation = {"strengths": [], "gaps": [], "overall_fit": "N/A"}
        top_match_details = None
        if llm_evaluator and matching_jobs:
             top_match_id = matching_jobs[0]['id']
             # Use the initialized job_matcher instance
             top_match_details = job_matcher.get_job_details(top_match_id)
             if top_match_details:
                 logger.debug("Evaluating candidate with LLM...")
                 eval_start = time.monotonic()
                 evaluation = llm_evaluator.evaluate_candidate(resume_text, top_match_details.get('title', ''), top_match_details.get('description', ''))
                 logger.debug(f"LLM evaluation took {time.monotonic() - eval_start:.4f}s")
             else:
                 logger.warning(f"Could not get details for top job {top_match_id} to perform evaluation.")
        elif not llm_evaluator:
             logger.info("LLM evaluation skipped as evaluator is not available.")


        # Use the globally loaded jobs_data list to find details if not fetched above
        if not top_match_details:
             top_match_id = matching_jobs[0]['id']
             top_match_details = next((job for job in jobs_data if str(job.get('id')) == str(top_match_id)), None)
             if not top_match_details:
                  logger.error(f"Could not find details for top job {top_match_id} in loaded data.")
                  top_match_details = {"error": "Could not fetch job details"}


        end_time = time.monotonic()
        logger.info(f"Finished /match-jobs for {file.filename} in {end_time - start_time:.4f}s")
        return {
            "top_matching_job": top_match_details, # Already serialized during startup load
            "similarity_score": float(matching_jobs[0]['similarity_score']), # Ensure float
            "evaluation": evaluation
        }
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /match-jobs for {file.filename} after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error matching jobs: {e}")
    finally:
        if os.path.exists(temp_path):
             try: os.remove(temp_path)
             except Exception as cleanup_e: logger.error(f"Failed to cleanup temp file {temp_path} on error: {cleanup_e}")


@app.get("/jobs")
async def get_jobs_endpoint(
    query: Optional[str] = Query(None),
    limit: int = Query(25)
):
    """Returns jobs, optionally filtered by query, from the in-memory list."""
    start_time = time.monotonic()
    logger.info(f"Starting /jobs query='{query}', limit={limit}")

    if not jobs_data: # Check if data was loaded
        logger.warning("Job data not loaded or empty.")
        # Decide whether to return empty or error
        # return []
        raise HTTPException(status_code=503, detail="Job data not available")

    try:
        # Filter the in-memory list
        if query:
            query_lower = query.lower()
            filtered_jobs = [
                job for job in jobs_data if
                (job.get('title') and query_lower in str(job['title']).lower()) or
                (job.get('description') and query_lower in str(job['description']).lower()) or
                (job.get('company') and query_lower in str(job['company']).lower())
            ]
        else:
            filtered_jobs = jobs_data # Use the full list

        # Apply limit
        results = filtered_jobs[:limit]

        end_time = time.monotonic()
        logger.info(f"Finished /jobs query='{query}' in {end_time - start_time:.4f}s, returning {len(results)} results")
        return results # Return list directly

    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /jobs after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error fetching jobs: {e}")


@app.get("/jobs/{job_id}")
async def get_job_endpoint(job_id: str):
    """Gets a single job by ID from the in-memory list."""
    start_time = time.monotonic()
    logger.info(f"Starting /jobs/{job_id}")

    if not jobs_data:
        raise HTTPException(status_code=503, detail="Job data not available")

    try:
        # Find job in the global list
        job = next((job for job in jobs_data if str(job.get('id')) == str(job_id)), None)

        if job is None:
             logger.warning(f"Job with ID {job_id} not found in memory.")
             raise HTTPException(status_code=404, detail="Job not found")

        end_time = time.monotonic()
        logger.info(f"Finished /jobs/{job_id} in {end_time - start_time:.4f}s")
        return job # Return the dictionary directly

    except HTTPException:
        raise # Re-raise specific HTTP exceptions
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error fetching job {job_id} after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error fetching job: {e}")


@app.post("/improvement-plan")
async def generate_improvement_plan_endpoint(request: EvaluationRequest):
    start_time = time.monotonic()
    logger.info("Starting /improvement-plan")
    if not resume_improver:
        raise HTTPException(status_code=501, detail="ResumeImprover feature not available")
    try:
        plan = resume_improver.generate_improvement_plan({"gaps": request.gaps})
        end_time = time.monotonic()
        logger.info(f"Finished /improvement-plan in {end_time - start_time:.4f}s")
        return {"improvement_plan": plan}
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /improvement-plan after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error generating improvement plan: {e}")

# --- Helper for finding latest resume ---
def _find_latest_resume_path(base_dir: str) -> Optional[str]:
    uploads_dir = os.path.join(base_dir, '..', 'uploads') # Relative to backend dir
    if not os.path.exists(uploads_dir):
        logger.warning(f"Uploads directory not found at {uploads_dir}")
        return None
    try:
        resumes = sorted(
            [os.path.join(uploads_dir, f) for f in os.listdir(uploads_dir) if f.lower().endswith(('.pdf', '.docx'))],
            key=os.path.getmtime,
            reverse=True
        )
        if resumes:
            logger.debug(f"Found latest resume: {resumes[0]}")
            return resumes[0]
        else:
            logger.warning("No resume files found in uploads directory.")
            return None
    except Exception as e:
        logger.error(f"Error finding latest resume: {e}")
        return None

@app.post("/jobs/{job_id}/match")
async def match_single_job(job_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting single job match for job_id={job_id}")
    if not job_matcher:
        raise HTTPException(status_code=500, detail="JobMatcher not available")
    if not resume_parser:
        raise HTTPException(status_code=500, detail="ResumeParser not available")

    try:
        latest_resume_path = _find_latest_resume_path(backend_dir)
        if not latest_resume_path:
            raise HTTPException(status_code=404, detail="No resume found to match against.")

        parse_start = time.monotonic()
        resume_data = resume_parser.parse_resume(latest_resume_path)
        resume_text = resume_data.get("text", "")
        logger.debug(f"Parsed resume for single match in {time.monotonic() - parse_start:.4f}s")

        match_start = time.monotonic()
        # Use the initialized job_matcher
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
    start_time = time.monotonic()
    logger.info("Starting /resumes GET")
    # Use the global backend_dir
    resumes_json_path = os.path.join(backend_dir, '..', 'resumes.json')
    if not os.path.exists(resumes_json_path):
        logger.warning("resumes.json not found in project root.")
        return {"resumes": []}
    try:
        with open(resumes_json_path, "r") as f:
            resumes_data = json.load(f)
        formatted_resumes = [
            {"id": r.get("id"), "name": r.get("filename"), "uploadedAt": r.get("upload_date"), "size": r.get("size", 0)}
            for r in resumes_data
        ]
        end_time = time.monotonic()
        logger.info(f"Finished /resumes GET in {end_time - start_time:.4f}s, returning {len(formatted_resumes)} resumes")
        return {"resumes": formatted_resumes}
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error reading resumes.json after {end_time - start_time:.4f}s")
        return {"resumes": [], "error": str(e)}

@app.get("/resumes/{resume_id}")
async def get_resume_endpoint(resume_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting /resumes/{resume_id} GET")
    resumes_json_path = os.path.join(backend_dir, '..', 'resumes.json')
    if not os.path.exists(resumes_json_path):
        raise HTTPException(status_code=404, detail="Resumes metadata file not found")

    try:
        with open(resumes_json_path, "r") as f:
            resumes_data = json.load(f)
        resume_meta = next((r for r in resumes_data if r.get("id") == resume_id), None)
        if not resume_meta:
            raise HTTPException(status_code=404, detail="Resume metadata not found")

        file_path_relative = resume_meta.get("path")
        if not file_path_relative:
             logger.warning(f"No path found in metadata for resume {resume_id}")
             raise HTTPException(status_code=404, detail="Resume file path missing in metadata")

        resume_file_path = os.path.join(backend_dir, '..', file_path_relative) # Path relative to project root
        logger.debug(f"Attempting to access resume file at: {resume_file_path}")

        if not os.path.exists(resume_file_path):
             logger.error(f"Resume file not found at resolved path: {resume_file_path}")
             raise HTTPException(status_code=404, detail=f"Resume file not found at specified path")

        # Try parsing if PDF and parser is available
        if resume_file_path.lower().endswith(".pdf") and resume_parser:
             parse_start = time.monotonic()
             parsed_data = resume_parser.parse_resume(resume_file_path)
             logger.debug(f"Parsed resume PDF in {time.monotonic() - parse_start:.4f}s")
             combined_data = {**resume_meta, **parsed_data}
             end_time = time.monotonic()
             logger.info(f"Finished /resumes/{resume_id} (parsed) in {end_time - start_time:.4f}s")
             return combined_data
        else:
             # Return basic metadata if not PDF or parser unavailable
             end_time = time.monotonic()
             logger.info(f"Finished /resumes/{resume_id} (metadata only) in {end_time - start_time:.4f}s")
             return resume_meta

    except HTTPException:
        raise
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error fetching resume {resume_id} after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error fetching resume: {e}")


@app.delete("/resumes/{resume_id}")
async def delete_resume_endpoint(resume_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting DELETE /resumes/{resume_id}")
    resumes_json_path = os.path.join(backend_dir, '..', 'resumes.json')
    uploads_dir = os.path.join(backend_dir, '..', 'uploads')
    file_deleted = False
    record_updated = False

    if not os.path.exists(resumes_json_path):
        logger.warning("resumes.json not found, attempting direct file deletion if pattern matches...")
        for ext in ['.pdf', '.docx']:
            potential_path = os.path.join(uploads_dir, f"{resume_id}{ext}")
            if os.path.exists(potential_path):
                 try:
                     os.remove(potential_path)
                     logger.info(f"Deleted resume file directly: {potential_path}")
                     end_time = time.monotonic()
                     logger.info(f"Finished DELETE /resumes/{resume_id} (file only) in {end_time - start_time:.4f}s")
                     return {"message": "Resume file deleted (metadata not found)"}
                 except Exception as e:
                     logger.exception(f"Error deleting resume file {potential_path}")
                     raise HTTPException(status_code=500, detail=f"Error deleting file: {e}")
        raise HTTPException(status_code=404, detail="Resume metadata and matching file not found")

    try:
        with open(resumes_json_path, "r") as f:
            resumes_data = json.load(f)

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
                         del_start = time.monotonic()
                         os.remove(file_path_absolute)
                         logger.info(f"Deleted resume file: {file_path_absolute} in {time.monotonic() - del_start:.4f}s")
                         file_deleted = True
                     except Exception as e:
                         logger.warning(f"Could not delete file {file_path_absolute}: {e}")
                 else:
                     logger.warning(f"File path in JSON does not exist: {file_path_absolute}")
            else:
                 logger.warning(f"No file path found in metadata for resume {resume_id}")

            # Update the JSON file
            update_start = time.monotonic()
            with open(resumes_json_path, "w") as f:
                json.dump(updated_resumes, f, indent=2)
            record_updated = True
            logger.info(f"Removed resume record {resume_id} from resumes.json in {time.monotonic() - update_start:.4f}s")

            end_time = time.monotonic()
            logger.info(f"Finished DELETE /resumes/{resume_id} (record updated, file deleted: {file_deleted}) in {end_time - start_time:.4f}s")
            return {"message": "Resume deleted successfully"}
        else:
            logger.warning(f"Resume record {resume_id} not found in resumes.json")
            raise HTTPException(status_code=404, detail="Resume record not found in metadata")

    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error deleting resume {resume_id} after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error deleting resume: {e}")


@app.post("/upload-resume")
async def upload_resume_endpoint(file: UploadFile = File(...)):
    start_time = time.monotonic()
    logger.info(f"Starting /upload-resume for {file.filename}")
    # Basic file validation
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
    # Save file path relative to project root for consistency
    file_path_relative = os.path.join('uploads', f"{resume_id}{file_ext}")
    file_path_absolute = os.path.join(backend_dir, '..', file_path_relative)

    try:
        # Save file
        save_start = time.monotonic()
        with open(file_path_absolute, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded resume to {file_path_absolute} in {time.monotonic() - save_start:.4f}s")

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
            "path": file_path_relative, # Path relative to project root
            "upload_date": datetime.now().isoformat(),
            "size": os.path.getsize(file_path_absolute)
        }
        resumes_data.append(new_resume_meta)

        update_start = time.monotonic()
        with open(resumes_json_path, "w") as f:
            json.dump(resumes_data, f, indent=2)
        logger.info(f"Updated resumes.json in {time.monotonic() - update_start:.4f}s")

        end_time = time.monotonic()
        logger.info(f"Finished /upload-resume for {file.filename} in {end_time - start_time:.4f}s")
        return {"message": "Resume uploaded successfully", "resume_id": resume_id, "file_info": new_resume_meta}

    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error uploading resume after {end_time - start_time:.4f}s")
        # Clean up saved file if error occurred
        if os.path.exists(file_path_absolute):
            os.remove(file_path_absolute)
        raise HTTPException(status_code=500, detail=f"Error uploading resume: {e}")


@app.post("/jobs")
async def add_job_endpoint(job: AddJobRequest):
    start_time = time.monotonic()
    logger.info(f"Starting POST /jobs for title: {job.title}")
    if not job_matcher:
         raise HTTPException(status_code=500, detail="JobMatcher not available")
    try:
        new_job_data = job.dict()
        new_job_data['id'] = str(uuid.uuid4()) # Generate new ID
        new_job_data['postedDate'] = datetime.now().strftime('%Y-%m-%d')
        # Ensure requirements is a string if needed by JobMatcher
        if isinstance(job.requirements, list):
             new_job_data['requirements'] = ";".join(job.requirements)
        else:
             new_job_data['requirements'] = str(job.requirements) # Fallback

        # Add to the DataFrame in JobMatcher
        add_start = time.monotonic()
        success = job_matcher.add_job(new_job_data)
        logger.debug(f"Adding job to internal DataFrame took {time.monotonic() - add_start:.4f}s")

        if success:
            # Save back to the CSV file using the path known by job_matcher
            save_start = time.monotonic()
            job_matcher.save_jobs() # Assuming save_jobs uses the path it was initialized with
            logger.info(f"Saved updated jobs CSV in {time.monotonic() - save_start:.4f}s")

            # Also update the in-memory list `jobs_data`
            global jobs_data
            # Convert added job requirements back to list for response consistency
            new_job_data['requirements'] = job.requirements
            new_job_data['similarityScore'] = 0.0 # Default score
            jobs_data.append(new_job_data)

            end_time = time.monotonic()
            logger.info(f"Finished POST /jobs for title: {job.title} in {end_time - start_time:.4f}s")
            return new_job_data
        else:
             raise HTTPException(status_code=500, detail="Failed to add job to internal list.")

    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error adding job after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error adding job: {e}")


# --- Gemini Endpoints (Simplified with Timing) ---
async def _call_gemini_timed(func, *args, func_name="Gemini Call"):
    start_time = time.monotonic()
    logger.info(f"Starting {func_name}...")
    try:
        # Ensure the function exists before calling
        if func is None or not callable(func):
             raise NotImplementedError(f"{func_name} function is not available.")
        result = func(*args)
        # If the underlying function is async, await it
        if asyncio.iscoroutine(result):
             result = await result
        duration = time.monotonic() - start_time
        logger.info(f"{func_name} completed in {duration:.4f}s")
        return result
    except Exception as e:
        duration = time.monotonic() - start_time
        logger.exception(f"{func_name} failed after {duration:.4f}s")
        raise # Re-raise the exception


@app.post("/jobs/{job_id}/tailor-resume")
async def tailor_resume_endpoint(job_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting /tailor-resume for job_id={job_id}")
    if not job_matcher or not resume_parser:
         raise HTTPException(status_code=500, detail="Required components not available")

    try:
        latest_resume_path = _find_latest_resume_path(backend_dir)
        if not latest_resume_path: raise HTTPException(status_code=404, detail="No resume found")
        resume_text = resume_parser.parse_resume(latest_resume_path).get("text", "")

        job = job_matcher.get_job_details(job_id) # Uses initialized job_matcher
        if not job: raise HTTPException(status_code=404, detail="Job not found")

        tailored = await _call_gemini_timed(gemini_tailor_resume, resume_text, job.get('description', ''), func_name="Tailor Resume")
        end_time = time.monotonic()
        logger.info(f"Finished /tailor-resume for job_id={job_id} in {end_time - start_time:.4f}s")
        return {"tailored_resume": tailored}

    except HTTPException:
        raise
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /tailor-resume after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error tailoring resume: {e}")

@app.post("/jobs/{job_id}/generate-cover-letter")
async def generate_cover_letter_endpoint(job_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting /generate-cover-letter for job_id={job_id}")
    if not job_matcher or not resume_parser:
        raise HTTPException(status_code=500, detail="Required components not available")
    try:
        latest_resume_path = _find_latest_resume_path(backend_dir)
        if not latest_resume_path: raise HTTPException(status_code=404, detail="No resume found")
        resume_text = resume_parser.parse_resume(latest_resume_path).get("text", "")

        job = job_matcher.get_job_details(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job not found")

        cover_letter = await _call_gemini_timed(gemini_generate_cover_letter, resume_text, job.get('description', ''), func_name="Generate Cover Letter")
        end_time = time.monotonic()
        logger.info(f"Finished /generate-cover-letter for job_id={job_id} in {end_time - start_time:.4f}s")
        return {"cover_letter": cover_letter}

    except HTTPException:
        raise
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /generate-cover-letter after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error generating cover letter: {e}")


@app.post("/jobs/{job_id}/generate-roadmap")
async def generate_roadmap_endpoint(job_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting /generate-roadmap for job_id={job_id}")
    if not job_matcher or not resume_parser:
        raise HTTPException(status_code=500, detail="Required components not available")
    try:
        latest_resume_path = _find_latest_resume_path(backend_dir)
        if not latest_resume_path: raise HTTPException(status_code=404, detail="No resume found")
        resume_text = resume_parser.parse_resume(latest_resume_path).get("text", "")

        job = job_matcher.get_job_details(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job not found")

        roadmap = await _call_gemini_timed(gemini_generate_roadmap, resume_text, job.get('description', ''), func_name="Generate Roadmap")
        end_time = time.monotonic()
        logger.info(f"Finished /generate-roadmap for job_id={job_id} in {end_time - start_time:.4f}s")
        return {"roadmap": roadmap}

    except HTTPException:
        raise
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /generate-roadmap after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error generating roadmap: {e}")


# --- Main Execution Guard ---
if __name__ == "__main__":
    # This block is primarily for local development runs
    port = int(os.environ.get("PORT", 8080)) # Use 8080 default for Render/Cloud Run compatibility
    logger.info(f"Starting Uvicorn development server on host 0.0.0.0, port {port} with reload")
    # Enable reload only for local dev runs
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)