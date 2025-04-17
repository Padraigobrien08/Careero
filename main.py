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
import datetime
import magic  # For file type validation
import pandas as pd
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query, Request # Added Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response # Added Response
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

# --- Define backend_dir at module level ---
backend_dir = os.path.dirname(os.path.abspath(__file__))

# --- Logging Setup ---
# Configure logging basic setup - adjust level as needed
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO) # Default to INFO if invalid level
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.info(f"Logging configured with level: {logging.getLevelName(logger.getEffectiveLevel())}")


# --- Global Placeholders & Attempt Imports ---
# Initialize all potential classes/functions to None first
JobMatcher = None
LLMEvaluator = None
ResumeParser = None
PDFProcessor = None
ResumeImprover = None
JobDataProcessor = None
gemini_tailor_resume = None
gemini_generate_cover_letter = None
gemini_generate_roadmap = None

# Attempt to import local modules using relative imports
try: from .job_matcher import JobMatcher; logger.info("Successfully imported JobMatcher")
except ImportError: logger.warning("job_matcher module not found.")
try: from .llm_evaluator import LLMEvaluator; logger.info("Successfully imported LLMEvaluator")
except ImportError: logger.warning("llm_evaluator module not found.")
try: from .resume_parser import ResumeParser; logger.info("Successfully imported ResumeParser")
except ImportError: logger.warning("resume_parser module not found.")
try: from .pdf_processor import PDFProcessor; logger.info("Successfully imported PDFProcessor")
except ImportError: logger.warning("pdf_processor module not found.")
try:
    from .gemini_service import tailor_resume as gemini_tailor_resume, \
                                generate_cover_letter as gemini_generate_cover_letter, \
                                generate_roadmap as gemini_generate_roadmap
    logger.info("Successfully imported gemini_service functions")
except ImportError: logger.warning("gemini_service module not found.")
try: from .resume_improver import ResumeImprover; logger.info("Successfully imported ResumeImprover")
except ImportError: logger.warning("resume_improver module not found.")
try: from .csv_processor import JobDataProcessor; logger.info("Successfully imported JobDataProcessor")
except ImportError: logger.warning("csv_processor module not found.")


# --- Helper Function for Loading Jobs (More Robust) ---
def _load_and_process_jobs_robustly(csv_path: str) -> Tuple[Optional[List[Dict]], Optional[JobMatcher]]:
    """
    Loads jobs from CSV via JobMatcher, processes, handles errors,
    and returns a tuple (list_of_job_dicts | None, job_matcher_instance | None).
    """
    if not JobMatcher:
        logger.error("Cannot load jobs: JobMatcher class was not imported.")
        return None, None
    if not csv_path or not os.path.exists(csv_path):
         logger.error(f"Cannot load jobs: CSV path is invalid or file doesn't exist ('{csv_path}').")
         return None, None

    local_job_matcher = None
    processed_jobs_list = None

    try:
        logger.info(f"Attempting to initialize JobMatcher with CSV: {csv_path}")
        # Initialize the class - its __init__ should handle internal loading
        local_job_matcher = JobMatcher(csv_path=csv_path)
        logger.info("JobMatcher instance created.")

        # *** Explicitly check the state AFTER initialization ***
        if hasattr(local_job_matcher, 'jobs_df') and isinstance(local_job_matcher.jobs_df, pd.DataFrame) and not local_job_matcher.jobs_df.empty:
            logger.info(f"JobMatcher internal jobs_df loaded successfully. Shape: {local_job_matcher.jobs_df.shape}. Processing for app state...")
            temp_df = local_job_matcher.jobs_df.copy()

            # Ensure essential columns exist for frontend compatibility, add if missing
            essential_cols = ['id', 'title', 'company', 'location', 'description', 'requirements', 'postedDate', 'salary']
            for col in essential_cols:
                if col not in temp_df.columns:
                    logger.warning(f"Column '{col}' missing in loaded CSV DataFrame. Adding it with default values (None).")
                    temp_df[col] = None

            # Add similarityScore if missing (default to 0.0)
            if 'similarityScore' not in temp_df.columns:
                 temp_df['similarityScore'] = 0.0

            # Safe serialization (handle numpy types, NaN/NaT)
            for col in temp_df.select_dtypes(include=[pd.np.number]).columns:
                 temp_df[col] = temp_df[col].apply(lambda x: float(x) if pd.notna(x) and hasattr(x, 'item') and isinstance(x, (pd.np.floating, float)) else (int(x) if pd.notna(x) and hasattr(x, 'item') and isinstance(x, (pd.np.integer, int)) else (None if pd.isna(x) else x)))

            processed_jobs_list = temp_df.where(pd.notna(temp_df), None).to_dict(orient='records')
            logger.info(f"Successfully processed {len(processed_jobs_list)} jobs into list.")
            # Both list and matcher instance are valid
            return processed_jobs_list, local_job_matcher
        else:
            # This case means JobMatcher initialized but its internal df is bad
            logger.error("JobMatcher initialized, BUT its internal 'jobs_df' is missing, None, not a DataFrame, or empty AFTER its internal load attempt.")
            return [], None # Return empty list, no valid matcher instance

    except Exception as e:
        # Catch any error during JobMatcher init or the processing above
        logger.exception(f"CRITICAL ERROR during JobMatcher init or processing data from '{csv_path}': {e}")
        return None, None # Indicate failure clearly


# --- Helper Function for Direct CSV Loading ---
def _load_jobs_directly(csv_path: str) -> Optional[List[Dict]]:
    """Loads jobs directly from CSV, processes, and returns a list of dicts."""
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
                if col == 'similarityScore':
                    df[col] = 0.0 # Default score
                else:
                    df[col] = None # Default value

        # Ensure 'id' is string and handle potential NaNs before conversion
        if 'id' in df.columns:
             df['id'] = df['id'].fillna(pd.Series([str(uuid.uuid4()) for _ in range(len(df))]))
             df['id'] = df['id'].astype(str)
        else: # Should not happen if added above, but safety check
             df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]


        # Safe serialization (handle numpy types, NaN/NaT -> None)
        temp_df = df.copy()
        for col in temp_df.select_dtypes(include=[pd.np.number]).columns:
             temp_df[col] = temp_df[col].apply(lambda x: float(x) if pd.notna(x) and isinstance(x, (pd.np.floating, float)) else (int(x) if pd.notna(x) and isinstance(x, (pd.np.integer, int)) else (None if pd.isna(x) else x)))
        # Handle object columns that might contain NaN/NaT
        for col in temp_df.select_dtypes(include=['object']).columns:
             temp_df[col] = temp_df[col].apply(lambda x: None if pd.isna(x) else x)

        processed_jobs_list = temp_df.where(pd.notna(temp_df), None).to_dict(orient='records')
        logger.info(f"Direct load: Successfully processed {len(processed_jobs_list)} jobs into list.")
        return processed_jobs_list

    except Exception as e:
        logger.exception(f"CRITICAL ERROR during direct CSV load/processing from '{csv_path}': {e}")
        return None


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize state variables
    app.state.jobs_data = [] # Default to empty list
    app.state.job_matcher = None
    app.state.resume_parser = None
    app.state.llm_evaluator = None
    app.state.resume_improver = None
    app.state.csv_processor = None # Likely redundant now

    logger.info("Lifespan: Startup sequence initiated...")
    start_time = time.monotonic()
    load_dotenv(dotenv_path=os.path.join(backend_dir, '.env'))

    # --- Determine CSV Path ---
    csv_file = os.path.join(backend_dir, 'job_title_des.csv')
    if not os.path.exists(csv_file):
        logger.warning(f"job_title_des.csv not found in backend directory: {csv_file}. Trying root...")
        root_csv_file = os.path.join(backend_dir, '..', 'job_title_des.csv')
        if os.path.exists(root_csv_file):
             logger.info(f"Found job_title_des.csv in root directory: {root_csv_file}")
             csv_file = root_csv_file
        else:
            logger.error("job_title_des.csv not found in root either. Cannot load primary job data.")
            csv_file = None # Set to None if not found

    # --- Load Job Data DIRECTLY into app.state ---
    if csv_file:
        jobs_list = _load_jobs_directly(csv_file)
        if jobs_list is not None:
            app.state.jobs_data = jobs_list
            logger.info(f"[startup] DIRECTLY loaded {len(app.state.jobs_data)} jobs into app.state.jobs_data, id={id(app.state.jobs_data)}")
        else:
            logger.error("[startup] Failed to load jobs directly from CSV. app.state.jobs_data remains empty.")
            app.state.jobs_data = [] # Ensure empty
    else:
        logger.error("[startup] Cannot load jobs, CSV file path is None.")
        app.state.jobs_data = [] # Ensure empty

    # --- Initialize other components ---
    # Store instances directly in app.state
    if ResumeParser:
        try: app.state.resume_parser = ResumeParser(); logger.info("ResumeParser initialized and stored in app.state.")
        except Exception as e: logger.exception("Failed to initialize ResumeParser")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key: logger.warning("GEMINI_API_KEY env var not set. LLM features will be unavailable.")

    if LLMEvaluator and gemini_api_key:
        try: app.state.llm_evaluator = LLMEvaluator(gemini_api_key); logger.info("LLMEvaluator initialized and stored in app.state.")
        except Exception as e: logger.exception("Failed to initialize LLMEvaluator")

    if ResumeImprover and gemini_api_key:
        try: app.state.resume_improver = ResumeImprover(gemini_api_key); logger.info("ResumeImprover initialized and stored in app.state.")
        except Exception as e: logger.exception("Failed to initialize ResumeImprover")

    # csv_processor might be redundant
    # if JobDataProcessor and csv_file:
    #     try: app.state.csv_processor = JobDataProcessor(csv_file); logger.info("JobDataProcessor initialized.")
    #     except Exception as e: logger.exception("Failed to initialize JobDataProcessor")

    end_time = time.monotonic()
    logger.info(f"Lifespan: Startup sequence finished in {end_time - start_time:.4f}s. Final jobs in state: {len(app.state.jobs_data)}")

    yield # Application runs here

    logger.info("Lifespan: Shutdown.")


# Create FastAPI app instance with lifespan manager
app = FastAPI(title="CareeroOS API", lifespan=lifespan)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        logger.exception(f"Request Failed: {request.method} {request.url} - Took: {process_time:.4f}s")
        return JSONResponse(status_code=500, content={"detail": "Internal Server Error"})


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
@app.get("/") # Implicitly handles HEAD
async def root_get(request: Request):
    logger.debug("Handling GET /")
    # Safely access jobs_data from app.state
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
    start_time = time.monotonic()
    logger.info(f"Starting /process-resume for {file.filename}")
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)
    if not local_resume_parser:
         logger.error("ResumeParser not available in app state.")
         raise HTTPException(status_code=503, detail="ResumeParser service not available")

    temp_path = f"temp_{uuid.uuid4()}_{file.filename}" # Unique temp name
    try:
        logger.debug("Saving temporary file...")
        save_start = time.monotonic()
        with open(temp_path, "wb") as buffer:
             shutil.copyfileobj(file.file, buffer)
        logger.debug(f"Temporary file saved in {time.monotonic() - save_start:.4f}s")

        logger.debug("Parsing resume...")
        parse_start = time.monotonic()
        result = local_resume_parser.parse_resume(temp_path)
        logger.debug(f"Resume parsing completed in {time.monotonic() - parse_start:.4f}s")

    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error during /process-resume for {file.filename} after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error processing resume: {e}")
    finally:
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
async def match_jobs_endpoint(request: Request, file: UploadFile = File(...)):
    start_time = time.monotonic()
    logger.info(f"Starting /match-jobs for {file.filename}")
    # Get instances from app.state
    local_job_matcher = getattr(request.app.state, 'job_matcher', None)
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)
    local_llm_evaluator = getattr(request.app.state, 'llm_evaluator', None)

    if not local_job_matcher:
         raise HTTPException(status_code=503, detail="JobMatcher service not available")
    if not local_resume_parser:
         raise HTTPException(status_code=503, detail="ResumeParser service not available")
    if not local_llm_evaluator:
         logger.warning("LLMEvaluator not available, proceeding without evaluation.")

    temp_path = f"temp_{uuid.uuid4()}_{file.filename}"
    try:
        logger.debug("Saving temporary file...")
        save_start = time.monotonic()
        with open(temp_path, "wb") as buffer:
             shutil.copyfileobj(file.file, buffer)
        logger.debug(f"Temp file saved in {time.monotonic() - save_start:.4f}s")

        logger.debug("Parsing resume...")
        parse_start = time.monotonic()
        resume_data = local_resume_parser.parse_resume(temp_path)
        resume_text = resume_data.get("text", "")
        logger.debug(f"Resume parsing took {time.monotonic() - parse_start:.4f}s")

        logger.debug("Finding matching jobs...")
        match_start = time.monotonic()
        matching_jobs = local_job_matcher.find_matching_jobs(resume_text)
        logger.debug(f"Job matching took {time.monotonic() - match_start:.4f}s")

        if not matching_jobs:
            logger.warning(f"No matching jobs found for {file.filename}")
            end_time = time.monotonic()
            logger.info(f"Finished /match-jobs for {file.filename} (no matches) in {end_time - start_time:.4f}s")
            return {"error": "No matching jobs found"}

        evaluation = {"strengths": [], "gaps": [], "overall_fit": "N/A"}
        top_match_details = None
        if local_llm_evaluator and matching_jobs:
             top_match_id = matching_jobs[0]['id']
             top_match_details = local_job_matcher.get_job_details(top_match_id)
             if top_match_details:
                 logger.debug("Evaluating candidate with LLM...")
                 eval_start = time.monotonic()
                 evaluation = local_llm_evaluator.evaluate_candidate(resume_text, top_match_details.get('title', ''), top_match_details.get('description', ''))
                 logger.debug(f"LLM evaluation took {time.monotonic() - eval_start:.4f}s")
             else:
                 logger.warning(f"Could not get details for top job {top_match_id} to perform evaluation.")
        elif not local_llm_evaluator:
             logger.info("LLM evaluation skipped as evaluator is not available.")

        # Use the app.state jobs_data list as fallback if needed
        if not top_match_details:
             top_match_id = matching_jobs[0]['id']
             jobs_in_memory = getattr(request.app.state, 'jobs_data', [])
             top_match_details = next((job for job in jobs_in_memory if str(job.get('id')) == str(top_match_id)), None)
             if not top_match_details:
                  logger.error(f"Could not find details for top job {top_match_id} in loaded app state.")
                  top_match_details = {"error": "Could not fetch job details"}


        end_time = time.monotonic()
        logger.info(f"Finished /match-jobs for {file.filename} in {end_time - start_time:.4f}s")
        return {
            "top_matching_job": top_match_details,
            "similarity_score": float(matching_jobs[0]['similarity_score']),
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

@app.post("/improvement-plan")
async def generate_improvement_plan_endpoint(request: Request):
    start_time = time.monotonic()
    logger.info("Starting /improvement-plan")
    # Get instance from app.state
    local_resume_improver = getattr(request.app.state, 'resume_improver', None)
    if not local_resume_improver:
        raise HTTPException(status_code=501, detail="ResumeImprover service not available")

    try:
        # Need to parse the request body correctly
        try:
             eval_request = EvaluationRequest.parse_raw(await request.body())
        except Exception as parse_error:
             logger.error(f"Failed to parse request body for improvement plan: {parse_error}")
             raise HTTPException(status_code=400, detail="Invalid request body")

        if not eval_request.gaps:
             logger.warning("No gaps provided in improvement plan request.")
             # Return empty plan or specific message?
             return {"improvement_plan": []}

        plan = local_resume_improver.generate_improvement_plan({"gaps": eval_request.gaps})
        end_time = time.monotonic()
        logger.info(f"Finished /improvement-plan in {end_time - start_time:.4f}s")
        return {"improvement_plan": plan}
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /improvement-plan after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error generating improvement plan: {e}")

@app.post("/jobs/{job_id}/match")
async def match_single_job(request: Request, job_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting single job match for job_id={job_id}")
    # Get instances from app.state
    local_job_matcher = getattr(request.app.state, 'job_matcher', None)
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)

    if not local_job_matcher: raise HTTPException(status_code=503, detail="JobMatcher not available")
    if not local_resume_parser: raise HTTPException(status_code=503, detail="ResumeParser not available")

    try:
        latest_resume_path = _find_latest_resume_path(backend_dir)
        if not latest_resume_path:
            raise HTTPException(status_code=404, detail="No resume found to match against.")

        parse_start = time.monotonic()
        resume_data = local_resume_parser.parse_resume(latest_resume_path)
        resume_text = resume_data.get("text", "")
        logger.debug(f"Parsed resume for single match in {time.monotonic() - parse_start:.4f}s")

        match_start = time.monotonic()
        score = local_job_matcher.calculate_similarity(resume_text, job_id)
        logger.debug(f"Calculated similarity score in {time.monotonic() - match_start:.4f}s")

        end_time = time.monotonic()
        logger.info(f"Finished single job match for job_id={job_id} in {end_time - start_time:.4f}s")
        return {"job_id": job_id, "similarity_score": float(score)}
    except HTTPException:
        raise
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error matching single job {job_id} after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error matching job {job_id}: {e}")

@app.get("/resumes")
async def get_resumes_endpoint():
    # Logic remains mostly the same, using backend_dir defined globally
    start_time = time.monotonic()
    logger.info("Starting /resumes GET")
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
async def get_resume_endpoint(request: Request, resume_id: str):
    # Logic remains mostly the same, uses global backend_dir
    # Access resume_parser via request.app.state
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)
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

        resume_file_path = os.path.join(backend_dir, '..', file_path_relative)
        logger.debug(f"Attempting to access resume file at: {resume_file_path}")

        if not os.path.exists(resume_file_path):
             logger.error(f"Resume file not found at resolved path: {resume_file_path}")
             raise HTTPException(status_code=404, detail=f"Resume file not found at specified path")

        if resume_file_path.lower().endswith(".pdf") and local_resume_parser:
             parse_start = time.monotonic()
             parsed_data = local_resume_parser.parse_resume(resume_file_path)
             logger.debug(f"Parsed resume PDF in {time.monotonic() - parse_start:.4f}s")
             combined_data = {**resume_meta, **parsed_data}
             end_time = time.monotonic()
             logger.info(f"Finished /resumes/{resume_id} (parsed) in {end_time - start_time:.4f}s")
             return combined_data
        else:
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
    # Logic remains mostly the same, uses global backend_dir
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
    # Logic remains mostly the same, uses global backend_dir
    start_time = time.monotonic()
    logger.info(f"Starting /upload-resume for {file.filename}")
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ['.pdf', '.docx']:
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF/DOCX allowed.")

    uploads_dir = os.path.join(backend_dir, '..', 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    resume_id = f"resume_{timestamp}"
    file_path_relative = os.path.join('uploads', f"{resume_id}{file_ext}")
    file_path_absolute = os.path.join(backend_dir, '..', file_path_relative)

    try:
        save_start = time.monotonic()
        with open(file_path_absolute, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"Saved uploaded resume to {file_path_absolute} in {time.monotonic() - save_start:.4f}s")

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
        if os.path.exists(file_path_absolute):
            os.remove(file_path_absolute)
        raise HTTPException(status_code=500, detail=f"Error uploading resume: {e}")

@app.post("/jobs")
async def add_job_endpoint(request: Request, job: AddJobRequest):
     start_time = time.monotonic()
     logger.info(f"Starting POST /jobs for title: {job.title}")
     # Get job_matcher instance from app.state
     local_job_matcher = getattr(request.app.state, 'job_matcher', None)
     if not local_job_matcher:
         raise HTTPException(status_code=503, detail="JobMatcher service not available")
     try:
         new_job_data = job.dict()
         new_job_data['id'] = str(uuid.uuid4())
         new_job_data['postedDate'] = datetime.now().strftime('%Y-%m-%d')
         if isinstance(job.requirements, list):
              new_job_data['requirements'] = ";".join(job.requirements)
         else:
              new_job_data['requirements'] = str(job.requirements)

         add_start = time.monotonic()
         success = local_job_matcher.add_job(new_job_data)
         logger.debug(f"Adding job to internal DataFrame took {time.monotonic() - add_start:.4f}s")

         if success:
             save_start = time.monotonic()
             saved_csv = local_job_matcher.save_jobs()
             if saved_csv:
                 logger.info(f"Saved updated jobs CSV in {time.monotonic() - save_start:.4f}s")
             else:
                 logger.error("Failed to save jobs CSV after adding new job.")
                 # Decide if this is a critical error

             # Update app.state.jobs_data
             if hasattr(request.app.state, 'jobs_data'):
                 response_job_state = new_job_data.copy()
                 response_job_state['requirements'] = job.requirements # Use original list for state
                 response_job_state['similarityScore'] = 0.0
                 request.app.state.jobs_data.append(response_job_state)
                 logger.info(f"Added new job to app.state.jobs_data. Total jobs now: {len(request.app.state.jobs_data)}")

             response_job_final = new_job_data.copy()
             response_job_final['requirements'] = job.requirements # Send list back to frontend
             response_job_final['similarityScore'] = 0.0

             end_time = time.monotonic()
             logger.info(f"Finished POST /jobs for title: {job.title} in {end_time - start_time:.4f}s")
             return response_job_final
         else:
             logger.error("Failed to add job to internal JobMatcher list.")
             raise HTTPException(status_code=500, detail="Failed to add job to internal list.")

     except Exception as e:
         end_time = time.monotonic()
         logger.exception(f"Error adding job after {end_time - start_time:.4f}s")
         raise HTTPException(status_code=500, detail=f"Error adding job: {e}")

@app.post("/jobs/{job_id}/tailor-resume")
async def tailor_resume_endpoint(request: Request, job_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting /tailor-resume for job_id={job_id}")
    # Get instances from app.state
    local_job_matcher = getattr(request.app.state, 'job_matcher', None)
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)

    if not local_job_matcher: raise HTTPException(status_code=503, detail="JobMatcher not available")
    if not local_resume_parser: raise HTTPException(status_code=503, detail="ResumeParser not available")

    try:
        latest_resume_path = _find_latest_resume_path(backend_dir)
        if not latest_resume_path: raise HTTPException(status_code=404, detail="No resume found")
        resume_text = local_resume_parser.parse_resume(latest_resume_path).get("text", "")

        job = local_job_matcher.get_job_details(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job not found")

        # Use gemini function potentially loaded at startup
        tailored = await _call_gemini_timed(gemini_tailor_resume, resume_text, job.get('description', ''), func_name="Tailor Resume")
        end_time = time.monotonic()
        logger.info(f"Finished /tailor-resume for job_id={job_id} in {end_time - start_time:.4f}s")
        return {"tailored_resume": tailored}

    except HTTPException:
        raise
    except NotImplementedError as nie: # Catch specific error if Gemini func unavailable
         logger.error(f"Tailor resume feature not implemented/available: {nie}")
         raise HTTPException(status_code=501, detail=str(nie))
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /tailor-resume after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error tailoring resume: {e}")

@app.post("/jobs/{job_id}/generate-cover-letter")
async def generate_cover_letter_endpoint(request: Request, job_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting /generate-cover-letter for job_id={job_id}")
    # Get instances from app.state
    local_job_matcher = getattr(request.app.state, 'job_matcher', None)
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)

    if not local_job_matcher: raise HTTPException(status_code=503, detail="JobMatcher not available")
    if not local_resume_parser: raise HTTPException(status_code=503, detail="ResumeParser not available")

    try:
        latest_resume_path = _find_latest_resume_path(backend_dir)
        if not latest_resume_path: raise HTTPException(status_code=404, detail="No resume found")
        resume_text = local_resume_parser.parse_resume(latest_resume_path).get("text", "")

        job = local_job_matcher.get_job_details(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job not found")

        cover_letter = await _call_gemini_timed(gemini_generate_cover_letter, resume_text, job.get('description', ''), func_name="Generate Cover Letter")
        end_time = time.monotonic()
        logger.info(f"Finished /generate-cover-letter for job_id={job_id} in {end_time - start_time:.4f}s")
        return {"cover_letter": cover_letter}

    except HTTPException:
        raise
    except NotImplementedError as nie:
        logger.error(f"Generate cover letter feature not implemented/available: {nie}")
        raise HTTPException(status_code=501, detail=str(nie))
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /generate-cover-letter after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error generating cover letter: {e}")

@app.post("/jobs/{job_id}/generate-roadmap")
async def generate_roadmap_endpoint(request: Request, job_id: str):
    start_time = time.monotonic()
    logger.info(f"Starting /generate-roadmap for job_id={job_id}")
    # Get instances from app.state
    local_job_matcher = getattr(request.app.state, 'job_matcher', None)
    local_resume_parser = getattr(request.app.state, 'resume_parser', None)

    if not local_job_matcher: raise HTTPException(status_code=503, detail="JobMatcher not available")
    if not local_resume_parser: raise HTTPException(status_code=503, detail="ResumeParser not available")

    try:
        latest_resume_path = _find_latest_resume_path(backend_dir)
        if not latest_resume_path: raise HTTPException(status_code=404, detail="No resume found")
        resume_text = local_resume_parser.parse_resume(latest_resume_path).get("text", "")

        job = local_job_matcher.get_job_details(job_id)
        if not job: raise HTTPException(status_code=404, detail="Job not found")

        roadmap = await _call_gemini_timed(gemini_generate_roadmap, resume_text, job.get('description', ''), func_name="Generate Roadmap")
        end_time = time.monotonic()
        logger.info(f"Finished /generate-roadmap for job_id={job_id} in {end_time - start_time:.4f}s")
        return {"roadmap": roadmap}

    except HTTPException:
        raise
    except NotImplementedError as nie:
        logger.error(f"Generate roadmap feature not implemented/available: {nie}")
        raise HTTPException(status_code=501, detail=str(nie))
    except Exception as e:
        end_time = time.monotonic()
        logger.exception(f"Error in /generate-roadmap after {end_time - start_time:.4f}s")
        raise HTTPException(status_code=500, detail=f"Error generating roadmap: {e}")

# --- Main Execution Guard ---
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting Uvicorn development server on host 0.0.0.0, port {port} with reload")
    # Use "main:app" string for reload to work correctly
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)