import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
import uvicorn
from pdf_processor import PDFProcessor
from csv_processor import JobDataProcessor
from job_matcher import JobMatcher
from llm_evaluator import LLMEvaluator
from resume_improver import ResumeImprover
import os
import json
import numpy as np
from dotenv import load_dotenv
from pydantic import BaseModel
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

app = FastAPI(title="Resume and Job Description Processor")

# Initialize processors
csv_processor = JobDataProcessor("backend/job_title_des.csv")
job_matcher = JobMatcher("backend/job_title_des.csv")
llm_evaluator = LLMEvaluator(os.getenv("GEMINI_API_KEY"))  # Make sure to set this environment variable
resume_improver = ResumeImprover(os.getenv("GEMINI_API_KEY"))

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class EvaluationRequest(BaseModel):
    gaps: List[str]

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming {request.method} request to {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status code: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise

@app.get("/")
async def root():
    logger.info("Root endpoint called")
    return {"message": "Welcome to the Resume and Job Description Processor API"}

@app.post("/process-resume")
async def process_resume(file: UploadFile = File(...)):
    """Process uploaded resume PDF and return extracted text"""
    logger.info(f"Processing resume: {file.filename}")
    try:
        # Save uploaded file temporarily
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"Saved temporary file to {temp_path}")
        
        # Process PDF
        text = PDFProcessor().extract_text(temp_path)
        logger.info("Successfully extracted text from PDF")
        
        # Clean up temporary file
        os.remove(temp_path)
        logger.info("Cleaned up temporary file")
        
        return {"text": text}
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/match-jobs")
async def match_jobs(file: UploadFile = File(...)):
    """Process resume and find matching jobs with LLM evaluation"""
    logger.info(f"Matching jobs for resume: {file.filename}")
    try:
        # Process resume
        temp_path = f"temp_{file.filename}"
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info("Processing resume text")
        resume_text = PDFProcessor().extract_text(temp_path)
        os.remove(temp_path)
        
        # Find matching jobs
        logger.info("Finding matching jobs")
        matching_jobs = job_matcher.find_matching_jobs(resume_text)
        
        if not matching_jobs:
            logger.warning("No matching jobs found")
            return {"error": "No matching jobs found"}
        
        # Get the top matching job
        top_job = matching_jobs[0]
        logger.info(f"Found top matching job with ID: {top_job['id']}")
        
        # Convert numpy types to Python native types for JSON serialization
        top_job = {
            'id': int(top_job['id']),
            'similarity_score': float(top_job['similarity_score'])
        }
        
        # Get full job details
        logger.info(f"Getting full job details for ID: {top_job['id']}")
        job_details = job_matcher.get_job_details(top_job['id'])
        
        # Evaluate candidate with LLM
        logger.info("Evaluating candidate with LLM")
        evaluation = llm_evaluator.evaluate_candidate(
            resume_text=resume_text,
            job_title=job_details['title'],
            job_description=job_details['description']
        )
        logger.info("LLM evaluation completed")
        
        return {
            "top_matching_job": job_details,
            "similarity_score": top_job['similarity_score'],
            "evaluation": evaluation
        }
        
    except Exception as e:
        logger.error(f"Error matching jobs: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs")
async def get_jobs(
    query: Optional[str] = Query(None, description="Search query for jobs"),
    limit: int = Query(10, description="Maximum number of jobs to return")
):
    try:
        jobs = csv_processor.search_jobs(query, limit)
        return {"jobs": jobs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    try:
        job = csv_processor.get_job_by_id(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/improvement-plan")
async def generate_improvement_plan(request: EvaluationRequest):
    """Generate an improvement plan based on the evaluation"""
    logger.info("Generating improvement plan")
    try:
        if not request.gaps:
            logger.warning("No gaps provided in request")
            raise HTTPException(
                status_code=400,
                detail="No gaps provided for improvement plan generation. Please ensure the evaluation contains gaps."
            )
            
        logger.info(f"Received gaps: {json.dumps(request.gaps, indent=2)}")
        improvement_plan = resume_improver.generate_improvement_plan({"gaps": request.gaps})
        
        if not improvement_plan:
            logger.error("Failed to generate improvement plan - no plan returned")
            raise HTTPException(
                status_code=500,
                detail="Failed to generate improvement plan. Please try again."
            )
            
        logger.info(f"Generated improvement plan: {json.dumps(improvement_plan, indent=2)}")
        return {"improvement_plan": improvement_plan}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating improvement plan: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while generating the improvement plan"
        )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

# Example usage of the process-resume endpoint using requests
url = "http://localhost:8000/process-resume"
files = {'file': open('POB.pdf', 'rb')}