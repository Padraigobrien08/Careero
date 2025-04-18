import logging
import logging.config
import os
import sys
import json
import traceback
import uuid
import secrets
import base64
import shutil
import time
import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
# Make sure datetime is imported if used for default dates
from datetime import datetime

import magic
import pandas as pd
# Need numpy for safe type checks if pandas uses it internally
try:
    import numpy as np
except ImportError:
    class DummyNp: # Basic fallback if numpy isn't installed (pandas usually needs it)
        number = (int, float); integer = (int,); floating = (float,)
    np = DummyNp()

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

# --- Define backend_dir at module level ---
backend_dir = os.path.dirname(os.path.abspath(__file__))

# --- Logging Setup ---
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Logging configured with level: {logging.getLevelName(logger.getEffectiveLevel())}")


# --- Global Placeholders & Attempt Imports ---
JobMatcher = None; LLMEvaluator = None; ResumeParser = None; PDFProcessor = None; ResumeImprover = None; JobDataProcessor = None;
gemini_tailor_resume = None; gemini_generate_cover_letter = None; gemini_generate_roadmap = None

# Attempt imports (keep existing try/except blocks, logging success/failure)
try: from .job_matcher import JobMatcher; logger.info("Successfully imported JobMatcher")
except ImportError: logger.warning("job_matcher module not found.")
try: from .llm_evaluator import LLMEvaluator; logger.info("Successfully imported LLMEvaluator")
except ImportError: logger.warning("llm_evaluator module not found.")
try: from .resume_parser import ResumeParser; logger.info("Successfully imported ResumeParser")
except ImportError: logger.warning("resume_parser module not found.")
try: from .pdf_processor import PDFProcessor; logger.info("Successfully imported PDFProcessor")
except ImportError: logger.warning("pdf_processor module not found.")
try:
    from .gemini_service import tailor_resume as gemini_tailor_resume, generate_cover_letter as gemini_generate_cover_letter, generate_roadmap as gemini_generate_roadmap
    logger.info("Successfully imported gemini_service functions")
except ImportError: logger.warning("gemini_service module not found.")
try: from .resume_improver import ResumeImprover; logger.info("Successfully imported ResumeImprover")
except ImportError: logger.warning("resume_improver module not found.")
try: from .csv_processor import JobDataProcessor; logger.info("Successfully imported JobDataProcessor")
except ImportError: logger.warning("csv_processor module not found.")


# --- Helper Function for Direct CSV Loading ---
def _load_jobs_directly(csv_path: str) -> Optional[List[Dict]]:
    """Loads jobs directly from CSV, processes, and returns a list of dicts or None on critical error."""
    if not csv_path or not os.path.exists(csv_path):
         logger.error(f"Direct load failed: CSV path is invalid or file doesn't exist ('{csv_path}').")
         return None
    try:
        logger.info(f"Directly reading CSV: {csv_path}")
        df = pd.read_csv(csv_path)
        logger.info(f"Direct load successful. DF shape: {df.shape}. Processing for app state...")

        # Ensure essential columns exist, add if missing
        essential_cols = ['id', 'title', 'company', 'location', 'description', 'requirements', 'postedDate', 'salary', 'similarityScore']
        for col in essential_cols:
            if col not in df.columns:
                logger.warning(f"Direct load: Column '{col}' missing. Adding with default.")
                if col == 'similarityScore': df[col] = 0.0
                # Check postedDate specifically
                elif col == 'postedDate': df[col] = datetime.now().strftime('%Y-%m-%d')
                elif col == 'id': pass # Handled below
                else: df[col] = None

        # Ensure 'id' is string and handle potential NaNs before conversion
        if 'id' not in df.columns or df['id'].isnull().any():
            logger.warning("'id' column missing or contains nulls. Generating UUIDs.")
            df['id'] = df.apply(lambda row: str(uuid.uuid4()) if pd.isna(row.get('id')) else str(row['id']), axis=1)
        else:
             df['id'] = df['id'].astype(str)

        # Safe serialization (handle numpy types, NaN/NaT -> None)
        temp_df = df.copy() # Work on a copy

        # Convert numeric types safely
        for col in temp_df.select_dtypes(include=[np.number]).columns:
            temp_df[col] = temp_df[col].apply(
                lambda x: None if pd.isna(x) else
                          float(x) if isinstance(x, (np.floating, float)) else
                          int(x) if isinstance(x, (np.integer, int)) else
                          x # Keep other types as is
            )
        # Convert object/other types safely to handle potential NaNs or other issues
        for col in temp_df.columns:
             if col not in temp_df.select_dtypes(include=[np.number]).columns: # Avoid re-processing numeric
                 temp_df[col] = temp_df[col].apply(lambda x: None if pd.isna(x) else x)

        processed_jobs_list = temp_df.to_dict(orient='records')
        logger.info(f"Direct load: Successfully processed {len(processed_jobs_list)} jobs into list.")
        return processed_jobs_list

    except Exception as e:
        logger.exception(f"CRITICAL ERROR during direct CSV load/processing from '{csv_path}': {e}")
        return None


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize state variables
    app.state.jobs_data = []
    app.state.job_matcher = None
    app.state.resume_parser = None
    app.state.llm_evaluator = None
    app.state.resume_improver = None
    # app.state.csv_processor = None # Likely redundant

    logger.info("Lifespan: Startup sequence initiated...")
    start_time = time.monotonic()
    load_dotenv(dotenv_path=os.path.join(backend_dir, '.env'))

    # --- Determine CSV Path ---
    csv_file = os.path.join(backend_dir, 'job_title_des.csv')
    if not os.path.exists(csv_file):
        logger.warning(f"job_title_des.csv not found in {backend_dir}. Trying root...")
        root_csv_file = os.path.join(backend_dir, '..', 'job_title_des.csv')
        if os.path.exists(root_csv_file):
             logger.info(f"Found job_title_des.csv in root: {root_csv_file}")
             csv_file = root_csv_file
        else:
            logger.error("job_title_des.csv not found in root either. Cannot load primary job data.")
            csv_file = None

    # --- Load Job Data DIRECTLY into app.state ---
    # Wrap this critical part separately
    jobs_loaded_directly = False
    if csv_file:
        try:
            logger.info(f"Attempting direct load into app.state.jobs_data from {csv_file}")
            jobs_list = _load_jobs_directly(csv_file)
            if jobs_list is not None:
                app.state.jobs_data = jobs_list
                jobs_loaded_directly = True
                logger.info(f"[startup] DIRECT LOAD SUCCESS: Stored {len(app.state.jobs_data)} jobs in app.state.jobs_data, id={id(app.state.jobs_data)}")
            else:
                logger.error("[startup] DIRECT LOAD FAILED: Helper returned None. app.state.jobs_data remains empty.")
                app.state.jobs_data = [] # Ensure empty
        except Exception as e:
             logger.exception(f"[startup] UNEXPECTED EXCEPTION during direct job load/assignment: {e}")
             app.state.jobs_data = [] # Ensure empty on any exception
    else:
        logger.error("[startup] Cannot load jobs, CSV file path is None.")
        app.state.jobs_data = [] # Ensure empty

    # --- Initialize other components (separately) ---
    if JobMatcher and csv_file:
        try:
            logger.info("Initializing JobMatcher instance...")
            # Initialize it, its internal load will happen (or fail)
            app.state.job_matcher = JobMatcher(csv_path=csv_file)
            if hasattr(app.state.job_matcher, 'jobs_df') and isinstance(app.state.job_matcher.jobs_df, pd.DataFrame) and not app.state.job_matcher.jobs_df.empty:
                 logger.info(f"JobMatcher instance created and its internal jobs_df appears valid. Shape: {app.state.job_matcher.jobs_df.shape}")
            else:
                 logger.warning("JobMatcher instance created, but its internal jobs_df is missing, not a DataFrame, or empty.")
        except Exception as e:
            logger.exception(f"Failed to initialize JobMatcher instance: {e}")
            app.state.job_matcher = None
    else:
         logger.warning("JobMatcher class not imported or CSV file missing. JobMatcher instance not created.")

    if ResumeParser:
        try: app.state.resume_parser = ResumeParser(); logger.info("ResumeParser initialized and stored.")
        except Exception as e: logger.exception("Failed to initialize ResumeParser")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key: logger.warning("GEMINI_API_KEY env var not set.")

    if LLMEvaluator and gemini_api_key:
        try: app.state.llm_evaluator = LLMEvaluator(gemini_api_key); logger.info("LLMEvaluator initialized.")
        except Exception as e: logger.exception("Failed to initialize LLMEvaluator")

    if ResumeImprover and gemini_api_key:
        try: app.state.resume_improver = ResumeImprover(gemini_api_key); logger.info("ResumeImprover initialized.")
        except Exception as e: logger.exception("Failed to initialize ResumeImprover")


    end_time = time.monotonic()
    # Log final state count explicitly
    final_job_count = len(getattr(app.state, 'jobs_data', []))
    logger.info(f"Lifespan: Startup sequence finished in {end_time - start_time:.4f}s. Final jobs count in app.state: {final_job_count}")

    yield # Application runs here

    logger.info("Lifespan: Shutdown.")


# Create FastAPI app instance with lifespan manager
app = FastAPI(title="CareeroOS API", lifespan=lifespan)

# CORS Middleware (keep as is)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# Request Logging Middleware (keep as is)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.monotonic()
    logger.info(f"Request Start: {request.method} {request.url}")
    try: response = await call_next(request)
    except Exception as e:
        process_time = time.monotonic() - start_time; logger.exception(f"Request Failed: {request.method} {request.url} - Took: {process_time:.4f}s")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})
    process_time = time.monotonic() - start_time
    logger.info(f"Request End: {request.method} {request.url} - Status: {response.status_code} - Took: {process_time:.4f}s")
    return response

# --- Pydantic Models --- (keep as is)
class EvaluationRequest(BaseModel): gaps: List[str] = []
class AddJobRequest(BaseModel): title: str; description: str; company: str; location: str; salary: Optional[str] = None; requirements: List[str] = []

# --- API Endpoints ---
@app.get("/") # Implicitly handles HEAD
async def root_get(request: Request):
    logger.debug("Handling GET /")
    jobs = getattr(request.app.state, 'jobs_data', [])
    logger.info(f"Root endpoint returning {len(jobs)} jobs from app.state, id={id(jobs)}")
    return jobs

@app.head("/")
async def root_head():
    logger.debug("Handling HEAD /")
    return Response(status_code=200)

@app.get("/jobs")
async def get_jobs_endpoint(
    request: Request,
    query: Optional[str] = Query(None),
    limit: int = Query(25)
):
    start_time = time.monotonic()
    # Directly use app.state.jobs_data loaded at startup
    jobs_in_memory = getattr(request.app.state, 'jobs_data', [])
    logger.info(f"[handler] /jobs sees {len(jobs_in_memory)} jobs in app.state, id={id(jobs_in_memory)}")

    # If state is empty after startup, return empty list (200 OK)
    # REMOVED LAZY LOAD - Rely solely on startup
    if not jobs_in_memory:
        logger.warning("/jobs handler: Job data in app.state is empty. Returning empty list.")
        return []

    # Proceed with filtering/returning jobs_in_memory
    try:
        if query:
            query_lower = query.lower()
            filtered_jobs = [
                job for job in jobs_in_memory if job and (
                (job.get('title') and query_lower in str(job['title']).lower()) or
                (job.get('description') and query_lower in str(job['description']).lower()) or
                (job.get('company') and query_lower in str(job['company']).lower())
                )
            ]
        else:
            filtered_jobs = jobs_in_memory

        results = filtered_jobs[:limit]
        end_time = time.monotonic()
        logger.info(f"Finished /jobs query='{query}' in {end_time - start_time:.4f}s, returning {len(results)} results")
        return results # Return 200 OK with the (potentially empty) list

    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /jobs handler processing after {end_time - start_time:.4f}s")
        # Don't return 503, return 500 for unexpected processing error
        raise HTTPException(status_code=500, detail=f"Error processing jobs: {e}")


# --- Other Endpoints ---
# Ensure ALL endpoints get components like job_matcher, resume_parser via
# `local_component = getattr(request.app.state, 'component_name', None)`
# and check `if not local_component:` raising an appropriate error (e.g., 503 Service Unavailable)

@app.get("/jobs/{job_id}")
async def get_job_endpoint(request: Request, job_id: str):
    start_time = time.monotonic()
    jobs_in_memory = getattr(request.app.state, 'jobs_data', [])
    logger.info(f"Starting /jobs/{job_id}. Handler sees {len(jobs_in_memory)} jobs in app.state, id={id(jobs_in_memory)}")
    if not jobs_in_memory: raise HTTPException(status_code=404, detail="Job data not available")
    try:
        job = next((j for j in jobs_in_memory if j and str(j.get('id')) == str(job_id)), None)
        if job is None: raise HTTPException(status_code=404, detail="Job not found")
        end_time = time.monotonic(); logger.info(f"Finished /jobs/{job_id} in {end_time - start_time:.4f}s")
        return job
    except HTTPException: raise
    except Exception as e: end_time = time.monotonic(); logger.exception(f"Error fetching job {job_id} after {end_time - start_time:.4f}s"); raise HTTPException(status_code=500, detail=f"Error fetching job: {e}")


@app.post("/process-resume")
async def process_uploaded_resume(request: Request, file: UploadFile = File(...)):
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)
    if not local_resume_parser: raise HTTPException(503, "ResumeParser unavailable")
    # Read file content
    # ... (Implementation needed based on ResumeParser usage)
    logger.info("Processing uploaded resume...")
    # Example: Parse and return data
    try:
        # Assuming ResumeParser has a parse method that takes file content
        content = await file.read()
        parsed_data = local_resume_parser.parse(content, file.filename) # Adjust method name if needed
        return {"filename": file.filename, "parsed_data": parsed_data}
    except Exception as e:
        logger.exception(f"Error processing resume {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process resume: {e}")


@app.post("/match-jobs")
async def match_jobs_endpoint(request: Request, file: UploadFile = File(...)):
     local_job_matcher = getattr(request.app.state, 'job_matcher', None)
     local_resume_parser = getattr(request.app.state, 'resume_parser', None)
     # local_llm_evaluator = getattr(request.app.state, 'llm_evaluator', None) # Not used directly here?
     if not local_job_matcher: raise HTTPException(503, "JobMatcher unavailable.")
     if not local_resume_parser: raise HTTPException(503, "ResumeParser unavailable.")
     logger.info("Matching jobs for uploaded resume...")
     try:
        content = await file.read()
        resume_text = local_resume_parser.parse_to_text(content, file.filename) # Assuming a method to get text
        if not resume_text:
             raise HTTPException(status_code=400, detail="Could not extract text from resume")

        matched_jobs = local_job_matcher.find_matching_jobs(resume_text)
        logger.info(f"Found {len(matched_jobs)} matches for resume {file.filename}")
        return matched_jobs
     except Exception as e:
        logger.exception(f"Error matching jobs for resume {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to match jobs: {e}")


@app.post("/improvement-plan")
async def generate_improvement_plan_endpoint(request: Request):
     local_resume_improver = getattr(request.app.state, 'resume_improver', None)
     if not local_resume_improver: raise HTTPException(501, "ResumeImprover unavailable.")
     # This endpoint likely needs the resume content and maybe target job details
     # Needs request body definition (e.g., Pydantic model)
     logger.warning("/improvement-plan endpoint needs implementation details (request body, logic)")
     # Placeholder implementation
     try:
        # Assuming request body contains resume_text and maybe job_description
        # body = await request.json()
        # resume_text = body.get("resume_text")
        # job_description = body.get("job_description") # Optional?
        # if not resume_text: raise HTTPException(400, "resume_text missing in request")
        # plan = local_resume_improver.generate_plan(resume_text, job_description) # Adjust method/args
        # return {"improvement_plan": plan}
        raise HTTPException(501, "Improvement plan endpoint not fully implemented")
     except Exception as e:
        logger.exception(f"Error generating improvement plan: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate plan: {e}")


@app.post("/jobs/{job_id}/match")
async def match_single_job(request: Request, job_id: str):
    local_job_matcher = getattr(request.app.state, 'job_matcher', None)
    local_resume_parser = getattr(request.app.state, 'resume_parser', None) # Assume resume context is available/needed
    jobs_in_memory = getattr(request.app.state, 'jobs_data', [])

    if not local_job_matcher: raise HTTPException(503, "JobMatcher unavailable.")
    if not local_resume_parser: raise HTTPException(503, "ResumeParser unavailable.") # Or handle differently if no resume needed

    # Find the specific job
    job = next((j for j in jobs_in_memory if j and str(j.get('id')) == str(job_id)), None)
    if job is None: raise HTTPException(status_code=404, detail="Job not found")

    logger.info(f"Matching single job {job_id}...")
    try:
        # Needs resume text - how is it provided? Request body? Session?
        # Placeholder: Assume resume_text comes from request body
        body = await request.json()
        resume_text = body.get("resume_text")
        if not resume_text: raise HTTPException(400, "resume_text missing in request")

        # Assuming JobMatcher has a method to calculate similarity for a single job
        similarity_score = local_job_matcher.calculate_similarity_for_job(resume_text, job) # Adjust method/args
        return {"job_id": job_id, "similarity_score": similarity_score}
    except Exception as e:
        logger.exception(f"Error matching single job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to match job: {e}")


@app.get("/resumes")
async def get_resumes_endpoint():
    resumes_dir = os.path.join(backend_dir, 'resumes')
    os.makedirs(resumes_dir, exist_ok=True)
    try:
        resumes = [f for f in os.listdir(resumes_dir) if os.path.isfile(os.path.join(resumes_dir, f))]
        return {"resumes": resumes}
    except Exception as e:
        logger.exception(f"Error listing resumes: {e}")
        raise HTTPException(status_code=500, detail="Failed to list resumes")


@app.get("/resumes/{resume_id}")
async def get_resume_endpoint(request: Request, resume_id: str):
    # Note: resume_id is likely the filename here
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)
    if not local_resume_parser: raise HTTPException(503, "ResumeParser unavailable.")

    resumes_dir = os.path.join(backend_dir, 'resumes')
    file_path = os.path.join(resumes_dir, resume_id)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Resume not found")

    try:
        # Could return file directly or parsed content
        # Returning parsed content as example
        with open(file_path, "rb") as f:
            content = f.read()
        # Pass content AND filename (resume_id) to parse method
        parsed_data = local_resume_parser.parse(content, resume_id) 
        # Return a structure consistent with what getCurrentResumeText expects
        return {"filename": resume_id, "parsed_data": parsed_data}
    except Exception as e:
        logger.exception(f"Error getting resume {resume_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get resume")


@app.delete("/resumes/{resume_id}")
async def delete_resume_endpoint(resume_id: str):
    # Note: resume_id is likely the filename here
    resumes_dir = os.path.join(backend_dir, 'resumes')
    file_path = os.path.join(resumes_dir, resume_id)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Resume not found")

    try:
        os.remove(file_path)
        logger.info(f"Deleted resume: {resume_id}")
        return {"message": f"Resume '{resume_id}' deleted successfully."}
    except Exception as e:
        logger.exception(f"Error deleting resume {resume_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete resume")


@app.post("/upload-resume")
async def upload_resume_endpoint(file: UploadFile = File(...)):
    resumes_dir = os.path.join(backend_dir, 'resumes')
    os.makedirs(resumes_dir, exist_ok=True)
    # Sanitize filename (important for security)
    safe_filename = secrets.token_hex(8) + "_" + "".join(c for c in file.filename if c.isalnum() or c in ('-', '_', '.'))
    file_path = os.path.join(resumes_dir, safe_filename)

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Uploaded resume saved as: {safe_filename}")
        return {"filename": safe_filename, "message": "Resume uploaded successfully"}
    except Exception as e:
        logger.exception(f"Error uploading resume {file.filename}: {e}")
        # Clean up partial file if upload failed
        if os.path.exists(file_path):
             try: os.remove(file_path); logger.info(f"Cleaned up partial file: {safe_filename}")
             except Exception as clean_e: logger.error(f"Error cleaning up partial file {safe_filename}: {clean_e}")
        raise HTTPException(status_code=500, detail="Failed to save uploaded resume")
    finally:
         # Ensure the file object is closed (and awaited)
         if hasattr(file, 'close') and callable(file.close):
             await file.close()


@app.post("/jobs")
async def add_job_endpoint(request: Request, job: AddJobRequest):
     local_job_matcher = getattr(request.app.state, 'job_matcher', None)
     jobs_in_memory = getattr(request.app.state, 'jobs_data', None) # Get the list itself

     if not local_job_matcher: raise HTTPException(503, "JobMatcher unavailable.")
     if jobs_in_memory is None: raise HTTPException(503, "Job data state unavailable.") # Check if list exists

     logger.info(f"Adding new job: {job.title}")
     try:
        # Add to JobMatcher (assuming it has an add_job method)
        new_job_data = job.dict()
        new_job_data['id'] = str(uuid.uuid4()) # Assign a new ID
        # Add default values if missing
        new_job_data.setdefault('postedDate', datetime.now().strftime('%Y-%m-%d'))
        new_job_data.setdefault('similarityScore', 0.0)

        # --- State Update Strategy ---
        # Option 1: Add to both JobMatcher's internal df AND app.state.jobs_data
        # This requires JobMatcher to expose an 'add_job' that also returns the dict
        added_job_dict = local_job_matcher.add_job(new_job_data) # Assumes this method exists & returns dict
        if added_job_dict:
            jobs_in_memory.append(added_job_dict) # Append to the list in app.state
            logger.info(f"Added job {added_job_dict['id']} to JobMatcher and app.state.jobs_data")
            # Consider saving back to CSV - requires JobMatcher method
            # local_job_matcher.save_jobs()
            return added_job_dict
        else:
            logger.error("Failed to add job via JobMatcher method.")
            raise HTTPException(500, "Failed to add job internally")

        # Option 2: Only add to app.state.jobs_data, JobMatcher becomes stale until restart
        # jobs_in_memory.append(new_job_data)
        # logger.warning(f"Added job {new_job_data['id']} to app.state.jobs_data ONLY. JobMatcher is now potentially stale.")
        # return new_job_data

     except Exception as e:
        logger.exception(f"Error adding job {job.title}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add job: {e}")


# Gemini Endpoints using app.state components
@app.post("/jobs/{job_id}/tailor-resume")
async def tailor_resume_endpoint(request: Request, job_id: str):
    local_job_matcher = getattr(request.app.state, 'job_matcher', None)
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)
    jobs_in_memory = getattr(request.app.state, 'jobs_data', [])

    if not local_job_matcher: raise HTTPException(503, "JobMatcher unavailable.")
    if not local_resume_parser: raise HTTPException(503, "ResumeParser unavailable.")
    if not gemini_tailor_resume: raise HTTPException(501, "Tailor Resume feature unavailable.")

    job = next((j for j in jobs_in_memory if j and str(j.get('id')) == str(job_id)), None)
    if job is None: raise HTTPException(status_code=404, detail="Job not found")

    logger.info(f"Tailoring resume for job {job_id}...")
    try:
        # Needs resume text - assuming from request body
        body = await request.json()
        resume_text = body.get("resume_text")
        if not resume_text: raise HTTPException(400, "resume_text missing in request")

        # Call the imported Gemini service function
        tailored_resume_text = await gemini_tailor_resume(resume_text, job) # Pass job dict
        return {"job_id": job_id, "tailored_resume": tailored_resume_text}

    except Exception as e:
        logger.exception(f"Error tailoring resume for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to tailor resume: {e}")


@app.post("/jobs/{job_id}/generate-cover-letter")
async def generate_cover_letter_endpoint(request: Request, job_id: str):
    local_job_matcher = getattr(request.app.state, 'job_matcher', None)
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)
    jobs_in_memory = getattr(request.app.state, 'jobs_data', [])

    if not local_job_matcher: raise HTTPException(503, "JobMatcher unavailable.") # Needed for job details? Yes.
    if not local_resume_parser: raise HTTPException(503, "ResumeParser unavailable.")
    if not gemini_generate_cover_letter: raise HTTPException(501, "Cover Letter feature unavailable.")

    job = next((j for j in jobs_in_memory if j and str(j.get('id')) == str(job_id)), None)
    if job is None: raise HTTPException(status_code=404, detail="Job not found")

    logger.info(f"Generating cover letter for job {job_id}...")
    try:
        body = await request.json()
        resume_text = body.get("resume_text")
        if not resume_text: raise HTTPException(400, "resume_text missing in request")

        cover_letter_text = await gemini_generate_cover_letter(resume_text, job)
        return {"job_id": job_id, "cover_letter": cover_letter_text}

    except Exception as e:
        logger.exception(f"Error generating cover letter for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate cover letter: {e}")


@app.post("/jobs/{job_id}/generate-roadmap")
async def generate_roadmap_endpoint(request: Request, job_id: str):
    local_job_matcher = getattr(request.app.state, 'job_matcher', None)
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)
    jobs_in_memory = getattr(request.app.state, 'jobs_data', [])

    if not local_job_matcher: raise HTTPException(503, "JobMatcher unavailable.") # Needed for job details
    if not local_resume_parser: raise HTTPException(503, "ResumeParser unavailable.")
    if not gemini_generate_roadmap: raise HTTPException(501, "Roadmap feature unavailable.")

    job = next((j for j in jobs_in_memory if j and str(j.get('id')) == str(job_id)), None)
    if job is None: raise HTTPException(status_code=404, detail="Job not found")

    logger.info(f"Generating roadmap for job {job_id}...")
    try:
        body = await request.json()
        resume_text = body.get("resume_text")
        if not resume_text: raise HTTPException(400, "resume_text missing in request")

        roadmap_text = await gemini_generate_roadmap(resume_text, job)
        return {"job_id": job_id, "roadmap": roadmap_text}

    except Exception as e:
        logger.exception(f"Error generating roadmap for job {job_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate roadmap: {e}")


# --- Main Execution Guard ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting Uvicorn development server on host 0.0.0.0, port {port} with reload")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)