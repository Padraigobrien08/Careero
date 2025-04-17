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
from typing import List, Dict, Any, Optional
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
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Added format
logger = logging.getLogger(__name__)


# --- Global Placeholders & Attempt Imports ---
# Define placeholders first
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
try:
    from .job_matcher import JobMatcher
    logger.debug("Successfully imported JobMatcher")
except ImportError:
    logger.warning("job_matcher module not found.")
try:
    from .llm_evaluator import LLMEvaluator
    logger.debug("Successfully imported LLMEvaluator")
except ImportError:
    logger.warning("llm_evaluator module not found.")
try:
    from .resume_parser import ResumeParser
    logger.debug("Successfully imported ResumeParser")
except ImportError:
    logger.warning("resume_parser module not found.")
try:
    from .pdf_processor import PDFProcessor
    logger.debug("Successfully imported PDFProcessor")
except ImportError:
    logger.warning("pdf_processor module not found.")
try:
    from .gemini_service import tailor_resume as gemini_tailor_resume, \
                                generate_cover_letter as gemini_generate_cover_letter, \
                                generate_roadmap as gemini_generate_roadmap
    logger.debug("Successfully imported gemini_service functions")
except ImportError:
    logger.warning("gemini_service module not found.")
    # Define placeholders if Gemini service is critical but might be missing
    def gemini_tailor_resume(*args, **kwargs): logger.warning("gemini_tailor_resume called but not available"); return None
    def gemini_generate_cover_letter(*args, **kwargs): logger.warning("gemini_generate_cover_letter called but not available"); return None
    def gemini_generate_roadmap(*args, **kwargs): logger.warning("gemini_generate_roadmap called but not available"); return None
try:
    from .resume_improver import ResumeImprover
    logger.debug("Successfully imported ResumeImprover")
except ImportError:
    logger.warning("resume_improver module not found.")
try:
    from .csv_processor import JobDataProcessor
    logger.debug("Successfully imported JobDataProcessor")
except ImportError:
    logger.warning("csv_processor module not found.")

# --- Component Instances (Initialized in lifespan) ---
# Using Optional typing and initializing to None
job_matcher_instance: Optional[JobMatcher] = None
resume_parser_instance: Optional[ResumeParser] = None
llm_evaluator_instance: Optional[LLMEvaluator] = None
resume_improver_instance: Optional[ResumeImprover] = None
csv_processor_instance: Optional[JobDataProcessor] = None


# --- Lifespan Event Handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize state variables
    app.state.jobs_data = []
    app.state.job_matcher = None
    app.state.resume_parser = None
    app.state.llm_evaluator = None
    app.state.resume_improver = None
    app.state.csv_processor = None

    logger.info("Application startup: Initializing components and loading data via lifespan...")
    start_time = time.monotonic()
    load_dotenv(dotenv_path=os.path.join(backend_dir, '.env')) # Use global backend_dir

    # --- Path and File Setup ---
    # Look for CSV *only* within the backend directory
    csv_file = os.path.join(backend_dir, 'job_title_des.csv')
    found_csv = False

    if not os.path.exists(csv_file):
        logger.error(f"CRITICAL: job_title_des.csv not found in backend directory: {csv_file}")
        # Optionally create default, but log error regardless
        logger.warning(f"Attempting to create default job_title_des.csv at {csv_file}")
        try:
            pd.DataFrame({
                'id': [str(uuid.uuid4())], 'title': ['Sample Job - Default'], 'company': ['Default Co'],
                'location': ['Default Location'], 'salary': [0], 'description': ['Default job description.'],
                'requirements': ['None'], 'postedDate': [datetime.now().strftime('%Y-%m-%d')]
            }).to_csv(csv_file, index=False)
            logger.info(f"Successfully created default CSV: {csv_file}")
            found_csv = True
        except Exception as e:
            logger.exception(f"Failed to create default CSV: {e}") # Use logger.exception
            csv_file = None # Ensure it's None if creation failed
    else:
        logger.info(f"Found job_title_des.csv at: {csv_file}")
        found_csv = True

    # --- Initialize components & Load Job Data ---
    jobs_loaded_successfully = False
    if JobMatcher and csv_file and found_csv:
        try:
            logger.info("Initializing JobMatcher...")
            # Use a local variable first
            local_job_matcher = JobMatcher(csv_path=csv_file)
            logger.info(f"JobMatcher initialized with data from {csv_file}.")

            # Check if jobs_df was loaded correctly inside JobMatcher
            if hasattr(local_job_matcher, 'jobs_df') and local_job_matcher.jobs_df is not None and not local_job_matcher.jobs_df.empty:
                 logger.info("JobMatcher has a non-empty jobs_df. Converting to app.state.jobs_data...")
                 # Handle potential numpy types during conversion more safely
                 temp_df = local_job_matcher.jobs_df.copy()
                 for col in temp_df.select_dtypes(include=[pd.np.number]).columns:
                     temp_df[col] = temp_df[col].apply(lambda x: float(x) if pd.notna(x) and hasattr(x, 'item') and isinstance(x, (pd.np.floating, float)) else (int(x) if pd.notna(x) and hasattr(x, 'item') and isinstance(x, (pd.np.integer, int)) else x))
                 # Convert NaN/NaT to None before to_dict
                 app.state.jobs_data = temp_df.where(pd.notna(temp_df), None).to_dict(orient='records')
                 app.state.job_matcher = local_job_matcher # Assign successful instance to app.state
                 jobs_loaded_successfully = True
                 logger.info(f"Loaded {len(app.state.jobs_data)} jobs into app.state.jobs_data.")
            else:
                 logger.warning("JobMatcher initialized, but its jobs_df attribute is empty or None after internal load.")
                 app.state.jobs_data = [] # Ensure it's an empty list

        except Exception as e:
            # Use logger.exception to include traceback
            logger.exception(f"Failed during JobMatcher initialization or data loading: {e}")
            app.state.jobs_data = [] # Ensure empty list on failure
            app.state.job_matcher = None
    elif JobMatcher and not (csv_file and found_csv):
         logger.error("JobMatcher class found, but CSV file is missing or path is invalid. Cannot initialize.")
    elif not JobMatcher:
         logger.error("JobMatcher class not found. Cannot initialize.")


    # Initialize other components and store in app.state
    if ResumeParser:
        try:
            app.state.resume_parser = ResumeParser()
            logger.info("ResumeParser initialized.")
        except Exception as e: logger.exception("Failed to initialize ResumeParser")

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
         logger.warning("GEMINI_API_KEY environment variable not set. LLM features will be unavailable.")

    if LLMEvaluator and gemini_api_key:
        try:
            app.state.llm_evaluator = LLMEvaluator(gemini_api_key)
            logger.info("LLMEvaluator initialized.")
        except Exception as e: logger.exception("Failed to initialize LLMEvaluator")
    elif LLMEvaluator and not gemini_api_key:
         logger.warning("LLMEvaluator class found, but cannot initialize without GEMINI_API_KEY.")

    if ResumeImprover and gemini_api_key:
        try:
            app.state.resume_improver = ResumeImprover(gemini_api_key)
            logger.info("ResumeImprover initialized.")
        except Exception as e: logger.exception("Failed to initialize ResumeImprover")
    elif ResumeImprover and not gemini_api_key:
        logger.warning("ResumeImprover class found, but cannot initialize without GEMINI_API_KEY.")

    if JobDataProcessor and csv_file and found_csv:
        try:
            app.state.csv_processor = JobDataProcessor(csv_file)
            logger.info("JobDataProcessor initialized.")
        except Exception as e: logger.exception("Failed to initialize JobDataProcessor")
    elif JobDataProcessor and not (csv_file and found_csv):
        logger.warning("JobDataProcessor class found, but CSV file path is invalid.")

    end_time = time.monotonic()
    logger.info(f"Application startup sequence finished in {end_time - start_time:.4f}s. Jobs loaded: {jobs_loaded_successfully}")

    yield # Application runs here

    # Code to run on shutdown (optional)
    logger.info("Application shutdown.")


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
        logger.exception(f"Request Failed: {request.method} {request.url} - Took: {process_time:.4f}s") # Log full exception
        # Return a generic 500 response instead of re-raising maybe?
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
async def root_get(request: Request): # Add request parameter
    """
    Returns the list of jobs loaded into memory at startup (GET request).
    """
    logger.debug("Handling GET /")
    # Access jobs from app.state
    jobs = request.app.state.jobs_data if hasattr(request.app.state, 'jobs_data') else []
    logger.info(f"Root endpoint returning {len(jobs)} jobs.")
    return jobs

@app.head("/")
async def root_head():
    """
    Handles HEAD requests for the root path. Returns empty response with headers.
    """
    logger.debug("Handling HEAD /")
    return Response(status_code=200) # Empty response is sufficient


@app.post("/process-resume")
async def process_uploaded_resume(request: Request, file: UploadFile = File(...)):
    start_time = time.monotonic()
    logger.info(f"Starting /process-resume for {file.filename}")
    local_resume_parser = request.app.state.resume_parser if hasattr(request.app.state, 'resume_parser') else None
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
    local_job_matcher = request.app.state.job_matcher if hasattr(request.app.state, 'job_matcher') else None
    local_resume_parser = request.app.state.resume_parser if hasattr(request.app.state, 'resume_parser') else None
    local_llm_evaluator = request.app.state.llm_evaluator if hasattr(request.app.state, 'llm_evaluator') else None

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
             jobs_in_memory = request.app.state.jobs_data if hasattr(request.app.state, 'jobs_data') else []
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


@app.get("/jobs")
async def get_jobs_endpoint(
    request: Request, # Add request parameter
    query: Optional[str] = Query(None),
    limit: int = Query(25)
):
    """Returns jobs, optionally filtered by query, from the in-memory list."""
    start_time = time.monotonic()
    # Access jobs from app.state via request
    jobs_in_memory = request.app.state.jobs_data if hasattr(request.app.state, 'jobs_data') else []
    logger.info(f"Starting /jobs query='{query}', limit={limit}. Handler sees {len(jobs_in_memory)} jobs in memory.")

    if not jobs_in_memory: # Check if data was loaded
        logger.warning("Job data in app.state is empty. Returning empty list.")
        # Return 200 OK with empty list instead of 503
        return []

    try:
        # Filter the in-memory list
        if query:
            query_lower = query.lower()
            filtered_jobs = [
                job for job in jobs_in_memory if job and ( # Check if job is not None
                (job.get('title') and query_lower in str(job['title']).lower()) or
                (job.get('description') and query_lower in str(job['description']).lower()) or
                (job.get('company') and query_lower in str(job['company']).lower())
                )
            ]
        else:
            filtered_jobs = jobs_in_memory # Use the full list

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
async def get_job_endpoint(request: Request, job_id: str): # Add request parameter
    """Gets a single job by ID from the in-memory list."""
    start_time = time.monotonic()
    # Access jobs from app.state
    jobs_in_memory = request.app.state.jobs_data if hasattr(request.app.state, 'jobs_data') else []
    logger.info(f"Starting /jobs/{job_id}. Handler sees {len(jobs_in_memory)} jobs in memory.")


    if not jobs_in_memory:
        logger.warning(f"Job data in app.state is empty when requesting job {job_id}.")
        raise HTTPException(status_code=404, detail="Job data not available") # 404 might be more appropriate

    try:
        # Find job in the app.state list
        job = next((job for job in jobs_in_memory if job and str(job.get('id')) == str(job_id)), None)

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
async def generate_improvement_plan_endpoint(request: Request): # Add request parameter
    start_time = time.monotonic()
    logger.info("Starting /improvement-plan")
    # Get instance from app.state
    local_resume_improver = request.app.state.resume_improver if hasattr(request.app.state, 'resume_improver') else None
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


# --- Helper for finding latest resume --- (Keep as is)
def _find_latest_resume_path(base_dir: str) -> Optional[str]:
    # ... (implementation remains the same)
    pass

@app.post("/jobs/{job_id}/match")
async def match_single_job(request: Request, job_id: str): # Add request parameter
    start_time = time.monotonic()
    logger.info(f"Starting single job match for job_id={job_id}")
    # Get instances from app.state
    local_job_matcher = request.app.state.job_matcher if hasattr(request.app.state, 'job_matcher') else None
    local_resume_parser = request.app.state.resume_parser if hasattr(request.app.state, 'resume_parser') else None

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
async def get_resume_endpoint(request: Request, resume_id: str): # Add request parameter
    # Logic remains mostly the same, uses global backend_dir
    # Access resume_parser via request.app.state
    local_resume_parser = request.app.state.resume_parser if hasattr(request.app.state, 'resume_parser') else None
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
async def add_job_endpoint(request: Request, job: AddJobRequest): # Add request parameter
     start_time = time.monotonic()
     logger.info(f"Starting POST /jobs for title: {job.title}")
     # Get job_matcher instance from app.state
     local_job_matcher = request.app.state.job_matcher if hasattr(request.app.state, 'job_matcher') else None
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


# --- Gemini Endpoints (Simplified with Timing & app.state) ---
async def _call_gemini_timed(func, *args, func_name="Gemini Call"):
    start_time = time.monotonic()
    logger.info(f"Starting {func_name}...")
    try:
        if func is None or not callable(func):
             raise NotImplementedError(f"{func_name} function is not available.")
        result = func(*args)
        if asyncio.iscoroutine(result): result = await result # Await if necessary
        duration = time.monotonic() - start_time
        logger.info(f"{func_name} completed in {duration:.4f}s")
        return result
    except Exception as e:
        duration = time.monotonic() - start_time
        logger.exception(f"{func_name} failed after {duration:.4f}s")
        raise # Re-raise the exception


@app.post("/jobs/{job_id}/tailor-resume")
async def tailor_resume_endpoint(request: Request, job_id: str): # Add request parameter
    start_time = time.monotonic()
    logger.info(f"Starting /tailor-resume for job_id={job_id}")
    # Get instances from app.state
    local_job_matcher = request.app.state.job_matcher if hasattr(request.app.state, 'job_matcher') else None
    local_resume_parser = request.app.state.resume_parser if hasattr(request.app.state, 'resume_parser') else None

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
async def generate_cover_letter_endpoint(request: Request, job_id: str): # Add request parameter
    start_time = time.monotonic()
    logger.info(f"Starting /generate-cover-letter for job_id={job_id}")
    # Get instances from app.state
    local_job_matcher = request.app.state.job_matcher if hasattr(request.app.state, 'job_matcher') else None
    local_resume_parser = request.app.state.resume_parser if hasattr(request.app.state, 'resume_parser') else None

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
async def generate_roadmap_endpoint(request: Request, job_id: str): # Add request parameter
    start_time = time.monotonic()
    logger.info(f"Starting /generate-roadmap for job_id={job_id}")
    # Get instances from app.state
    local_job_matcher = request.app.state.job_matcher if hasattr(request.app.state, 'job_matcher') else None
    local_resume_parser = request.app.state.resume_parser if hasattr(request.app.state, 'resume_parser') else None

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
    # This block is primarily for local development runs
    port = int(os.environ.get("PORT", 8080)) # Use 8080 default for Render/Cloud Run compatibility
    logger.info(f"Starting Uvicorn development server on host 0.0.0.0, port {port} with reload")
    # Enable reload only for local dev runs - use "main:app" string to allow reload to work
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)