from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import uvicorn
from job_matcher import JobMatcher
from llm_evaluator import LLMEvaluator
from resume_parser import ResumeParser
import os
from dotenv import load_dotenv
import pandas as pd
import json
from pdf_processor import PDFProcessor
import shutil
from datetime import datetime
from fastapi.responses import FileResponse, JSONResponse
import base64
from pathlib import Path
import secrets
import magic  # For file type validation
import uuid
from gemini_service import tailor_resume, generate_cover_letter, generate_roadmap as generate_roadmap_suggestions
import logging
import logging.config

# Load environment variables
load_dotenv()

# Security settings
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "").split(",")
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB
ALLOWED_FILE_TYPES = {
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'application/msword': '.doc'
}

app = FastAPI(title="CareeroOS API")

# Security middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security dependencies
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def validate_file_type(file: UploadFile) -> bool:
    """Validate file type using magic numbers"""
    try:
        file_content = file.file.read(2048)
        file.file.seek(0)
        mime_type = magic.from_buffer(file_content, mime=True)
        return mime_type in ALLOWED_FILE_TYPES
    except Exception:
        return False

def validate_file_size(file: UploadFile) -> bool:
    """Validate file size"""
    try:
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        file.file.seek(0)  # Reset file pointer
        return size <= MAX_FILE_SIZE
    except Exception:
        return False

# Initialize components
resume_parser = ResumeParser()
# Use the CSV file from the backend directory
job_matcher = JobMatcher()
job_matcher.load_jobs('backend/job_title_des.csv')

# Load job data
try:
    jobs_df = pd.read_csv('backend/job_title_des.csv')
except FileNotFoundError:
    jobs_df = pd.DataFrame(columns=['title', 'description'])

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "uploads"
if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

class ResumeAnalysis(BaseModel):
    text: str
    skills: List[str]
    experience: List[Dict[str, Any]]
    education: List[Dict[str, Any]]

class JobMatchRequest(BaseModel):
    resume_path: str

class JobMatch(BaseModel):
    title: str
    description: str
    similarity_score: float

class JobMatchResponse(BaseModel):
    matches: List[JobMatch]

class EvaluationRequest(BaseModel):
    evaluation: Dict[str, Any]

class AddJobRequest(BaseModel):
    title: str
    description: str
    company: str
    location: str
    salary: str
    requirements: List[str]

@app.post("/parse-resume")
async def parse_resume(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
            
        file_extension = os.path.splitext(file.filename)[1].lower()
        if file_extension not in ['.pdf', '.docx']:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF or DOCX file.")

        # Ensure uploads directory exists
        if not os.path.exists(UPLOAD_DIR):
            os.makedirs(UPLOAD_DIR)

        # Save the uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(UPLOAD_DIR, f"resume_{timestamp}{file_extension}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Parse the resume
        parser = ResumeParser()
        result = parser.parse_resume(file_path)
        
        # Add file info to the result
        file_info = {
            "id": f"resume_{timestamp}",
            "name": file.filename,
            "path": file_path,
            "uploadedAt": datetime.now().isoformat(),
            "size": os.path.getsize(file_path)
        }
        
        return {
            "success": True,
            "file_info": file_info,
            "parsed_data": result
        }
    except HTTPException:
        raise
    except Exception as e:
        # Clean up the file if parsing failed
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/resumes")
async def get_resumes():
    try:
        # Read from the resumes.json file
        resumes_file = "resumes.json"
        if not os.path.exists(resumes_file):
            return {"resumes": []}
            
        try:
            with open(resumes_file, "r") as f:
                resumes_data = json.load(f)
        except json.JSONDecodeError:
            resumes_data = []
        
        # Format the resumes for the frontend
        resumes = []
        for resume in resumes_data:
            resumes.append({
                "id": resume["id"],
                "name": resume["filename"],
                "uploadedAt": resume["upload_date"],
                "size": os.path.getsize(resume["path"]) if os.path.exists(resume["path"]) else 0
            })
            
        # Fallback to the old method if no resumes found in the json file
        if not resumes:
            if not os.path.exists(UPLOAD_DIR):
                return {"resumes": []}
                
            for filename in os.listdir(UPLOAD_DIR):
                if filename.startswith("resume_"):
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    file_info = {
                        "id": os.path.splitext(filename)[0],
                        "name": filename,
                        "uploadedAt": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                        "size": os.path.getsize(file_path)
                    }
                    resumes.append(file_info)
                    
        return {"resumes": sorted(resumes, key=lambda x: x["uploadedAt"], reverse=True)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/resumes/{resume_id}")
async def get_resume(resume_id: str):
    try:
        # First try to find resume in resumes.json
        resumes_file = "resumes.json"
        if os.path.exists(resumes_file):
            try:
                with open(resumes_file, "r") as f:
                    resumes_data = json.load(f)
                
                # Find the resume with the given ID
                resume_record = next((r for r in resumes_data if r["id"] == resume_id), None)
                
                if resume_record and os.path.exists(resume_record["path"]):
                    # Get the file content as base64
                    with open(resume_record["path"], "rb") as f:
                        file_content = f.read()
                        base64_content = base64.b64encode(file_content).decode('utf-8')
                    
                    # Use parsed content from the record or re-parse if needed
                    parsed_content = {
                        "text": resume_record.get("text", ""),
                        "skills": resume_record.get("skills", []),
                        "education": resume_record.get("education", []),
                        "experience": resume_record.get("experience", [])
                    }
                    
                    # If parsed content is empty, try to parse again
                    if not parsed_content["text"]:
                        parser = ResumeParser()
                        parsed_content = parser.parse_resume(resume_record["path"])
                    
                    ext = os.path.splitext(resume_record["path"])[1].lower()
                    content_type = "application/pdf" if ext == '.pdf' else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    
                    return {
                        "file": {
                            "content": base64_content,
                            "filename": resume_record["filename"],
                            "content_type": content_type
                        },
                        "parsed_content": parsed_content
                    }
            except json.JSONDecodeError:
                # Continue to fallback method if JSON is invalid
                pass
        
        # Fallback to the old method: look for the file with any supported extension
        for ext in ['.pdf', '.docx']:
            file_path = os.path.join(UPLOAD_DIR, f"{resume_id}{ext}")
            if os.path.exists(file_path):
                # Get the file content as base64
                with open(file_path, "rb") as f:
                    file_content = f.read()
                    base64_content = base64.b64encode(file_content).decode('utf-8')
                
                # Parse the resume content
                parser = ResumeParser()
                parsed_content = parser.parse_resume(file_path)
                
                return {
                    "file": {
                        "content": base64_content,
                        "filename": f"{resume_id}{ext}",
                        "content_type": "application/pdf" if ext == '.pdf' else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    },
                    "parsed_content": parsed_content
                }
        
        raise HTTPException(status_code=404, detail="Resume not found")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: str):
    try:
        # Look for the file with any supported extension
        for ext in ['.pdf', '.docx']:
            file_path = os.path.join(UPLOAD_DIR, f"{resume_id}{ext}")
            if os.path.exists(file_path):
                os.remove(file_path)
                return {"message": "Resume deleted successfully"}
        
        raise HTTPException(status_code=404, detail="Resume not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):
    try:
        # Read file content
        content = await file.read()
        
        # Parse resume
        parsed_data = resume_parser.parse_resume(content, file.content_type)
        
        return {
            "success": True,
            "data": parsed_data
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/match-jobs")
async def match_jobs(request: JobMatchRequest):
    try:
        # Parse the resume
        parser = ResumeParser()
        resume_data = parser.parse_resume(request.resume_path)
        
        # Get job matches
        matches = job_matcher.match_jobs(resume_data["text"])
        
        return {
            "matches": matches
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def migrate_to_uuids(df):
    """Migrate existing numeric IDs to UUIDs"""
    # Create a new id column with UUIDs for rows without them
    for idx, row in df.iterrows():
        if pd.isna(row.get('id')) or str(row['id']).replace('.', '').isdigit():
            df.at[idx, 'id'] = str(uuid.uuid4())
    return df

@app.get("/jobs")
async def get_jobs():
    try:
        resume_text = None
        
        # Try to get resume text from resumes.json first
        resumes_file = "resumes.json"
        if os.path.exists(resumes_file):
            try:
                with open(resumes_file, "r") as f:
                    resumes_data = json.load(f)
                
                if resumes_data:
                    # Sort resumes by upload date
                    sorted_resumes = sorted(
                        resumes_data, 
                        key=lambda x: x.get("upload_date", ""), 
                        reverse=True
                    )
                    
                    # Use the most recent resume
                    if sorted_resumes:
                        latest_resume = sorted_resumes[0]
                        resume_text = latest_resume.get("text", "")
                        
                        # If the text is empty but we have the path, try to parse the file
                        if not resume_text and os.path.exists(latest_resume.get("path", "")):
                            parser = ResumeParser()
                            resume_data = parser.parse_resume(latest_resume["path"])
                            resume_text = resume_data.get("text", "")
            except json.JSONDecodeError:
                # Continue to fallback method if JSON is invalid
                pass
        
        # Fallback to the old method if no resume text found
        if not resume_text:
            # Find the most recent resume file
            resume_files = []
            for filename in os.listdir(UPLOAD_DIR):
                if filename.startswith("resume_") and (filename.endswith(".pdf") or filename.endswith(".docx")):
                    file_path = os.path.join(UPLOAD_DIR, filename)
                    resume_files.append((file_path, os.path.getmtime(file_path)))
            
            if not resume_files:
                raise HTTPException(status_code=404, detail="No resume found. Please upload a resume first.")
            
            # Get the most recent resume
            latest_resume = max(resume_files, key=lambda x: x[1])[0]
            
            # Parse the resume
            parser = ResumeParser()
            resume_data = parser.parse_resume(latest_resume)
            
            # Extract the text from the parsed resume
            if not isinstance(resume_data, dict) or 'text' not in resume_data:
                raise HTTPException(status_code=500, detail="Failed to parse resume text")
                
            resume_text = resume_data['text']
        
        # If we still don't have resume text, raise an error
        if not resume_text:
            raise HTTPException(status_code=500, detail="Failed to extract resume text")
        
        # Get job matches
        matches = job_matcher.match_jobs(resume_text)
        
        # Load job data
        try:
            df = pd.read_csv('backend/job_title_des.csv', encoding='utf-8')
            print(f"Loaded {len(df)} jobs from CSV")
        except Exception as e:
            print(f"Error loading jobs CSV: {str(e)}")
            # Return empty list if file doesn't exist or is corrupt
            return {"matches": []}
        
        # Clean up the DataFrame
        # Remove duplicate columns and keep the most complete ones
        if 'Job Title' in df.columns and 'title' in df.columns:
            df['title'] = df['title'].fillna(df['Job Title'])
            df = df.drop('Job Title', axis=1)
        if 'Job Description' in df.columns and 'description' in df.columns:
            df['description'] = df['description'].fillna(df['Job Description'])
            df = df.drop('Job Description', axis=1)
        
        # Convert index to string if it exists and no id column
        if 'index' in df.columns and 'id' not in df.columns:
            df['id'] = df['index'].astype(str)
            df = df.drop('index', axis=1)
        
        # Migrate numeric IDs to UUIDs
        df = migrate_to_uuids(df)
        
        # Save the updated DataFrame with UUIDs
        df.to_csv('backend/job_title_des.csv', index=False)
        
        # Ensure required columns exist
        required_columns = ['id', 'title', 'description', 'company', 'location', 'salary', 'requirements', 'posted_date']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        
        # Build jobs list with proper error handling
        jobs = []
        for _, row in df.iterrows():
            try:
                # Skip if title or description is empty
                if pd.isna(row['title']) or pd.isna(row['description']):
                    continue
                
                # Get match score from matches dictionary
                job_id = str(row['id'])
                
                # Calculate match score if not in matches
                if job_id not in matches:
                    job_description = str(row['description'])
                    match_score = job_matcher.calculate_similarity(resume_text, job_description)
                    matches[job_id] = match_score
                else:
                    match_score = matches[job_id]
                
                # Convert requirements to array if it's a string
                requirements = row['requirements']
                if isinstance(requirements, str):
                    requirements = [req.strip() for req in requirements.split('\n') if req.strip()]
                elif pd.isna(requirements):
                    requirements = []
                elif not isinstance(requirements, list):
                    requirements = []
                
                # Default for posted_date if missing
                posted_date = row.get('posted_date', None)
                if pd.isna(posted_date):
                    posted_date = datetime.now().strftime("%Y-%m-%d")
                
                jobs.append({
                    "id": job_id,
                    "title": row['title'],
                    "company": row['company'] if not pd.isna(row['company']) else "Unknown Company",
                    "location": row['location'] if not pd.isna(row['location']) else "Remote",
                    "salary": row['salary'] if not pd.isna(row['salary']) else "Competitive",
                    "description": row['description'],
                    "requirements": requirements,
                    "postedDate": posted_date,
                    "similarityScore": float(match_score)
                })
            except Exception as e:
                print(f"Error processing job row: {str(e)}")
                continue
        
        # Sort jobs by similarity score (descending)
        jobs.sort(key=lambda x: x["similarityScore"], reverse=True)
        
        return jobs
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error in get_jobs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/jobs/{job_id}/match")
async def match_job(job_id: str):
    try:
        print(f"Starting job match for job_id: {job_id}")
        
        # Get the most recent resume
        resume_files = []
        for filename in os.listdir(UPLOAD_DIR):
            if filename.startswith("resume_") and (filename.endswith(".pdf") or filename.endswith(".docx")):
                file_path = os.path.join(UPLOAD_DIR, filename)
                resume_files.append((file_path, os.path.getmtime(file_path)))
        
        print(f"Found {len(resume_files)} resume files")
        
        if not resume_files:
            raise HTTPException(status_code=404, detail="No resume found. Please upload a resume first.")
        
        # Get the most recent resume
        latest_resume = max(resume_files, key=lambda x: x[1])[0]
        print(f"Using resume: {latest_resume}")
        
        # Parse the resume
        parser = ResumeParser()
        print("Parsing resume...")
        resume_data = parser.parse_resume(latest_resume)
        print("Resume parsed successfully")
        
        # Extract the text from the parsed resume
        if not isinstance(resume_data, dict) or 'text' not in resume_data:
            raise HTTPException(status_code=500, detail="Failed to parse resume text")
            
        resume_text = resume_data['text']
        print(f"Extracted resume text length: {len(resume_text)}")
        
        # Get job matches
        print("Getting job matches...")
        matches = job_matcher.match_jobs(resume_text)
        print(f"Found {len(matches)} matches")
        
        # If job_id not in matches, calculate it directly
        if job_id not in matches:
            print(f"Job {job_id} not in matches, calculating directly...")
            # Load job data
            df = pd.read_csv('backend/job_title_des.csv', encoding='utf-8')
            print(f"Loaded job data with {len(df)} rows")
            
            # Try to find the job by UUID first, then by numeric ID if not found
            job_row = df[df['id'] == job_id]
            if job_row.empty and job_id.replace('.', '').isdigit():
                # Try finding by numeric ID
                job_row = df[df['id'].astype(str).str.replace('.', '').str.strip() == job_id.replace('.', '')]
            
            print(f"Found {len(job_row)} rows matching job_id {job_id}")
            
            if job_row.empty:
                raise HTTPException(status_code=404, detail="Job not found")
                
            job_description = job_row.iloc[0]['description']
            print(f"Calculating similarity for job description length: {len(job_description)}")
            match_score = job_matcher.calculate_similarity(resume_text, job_description)
            
            # Update the job's ID to UUID if it's numeric
            if str(job_row.iloc[0]['id']).replace('.', '').isdigit():
                new_id = str(uuid.uuid4())
                df.loc[job_row.index[0], 'id'] = new_id
                df.to_csv('backend/job_title_des.csv', index=False)
                print(f"Updated job ID from {job_id} to {new_id}")
        else:
            match_score = matches[job_id]
        
        print(f"Final match score: {match_score}")
        return {"similarityScore": float(match_score)}
    except Exception as e:
        print(f"Error in match_job: {str(e)}")
        print(f"Error type: {type(e)}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/search")
async def search_jobs(query: str):
    try:
        # Load job data from CSV
        df = pd.read_csv('backend/job_title_des.csv')
        
        # Filter jobs based on search query
        search_results = []
        for _, row in df.iterrows():
            if pd.isna(row['title']) or pd.isna(row['description']):
                continue
                
            # Check if the job matches the search query
            title = str(row['title']).lower() if not pd.isna(row['title']) else ''
            description = str(row['description']).lower() if not pd.isna(row['description']) else ''
            company = str(row['company']).lower() if not pd.isna(row['company']) else ''
            query_lower = query.lower()
            
            if (query_lower in title or 
                query_lower in description or
                query_lower in company):
                
                job_id = str(row['id']) if not pd.isna(row['id']) else str(uuid.uuid4())
                job = {
                    "id": job_id,
                    "title": row['title'],
                    "company": row['company'] if not pd.isna(row['company']) else "Unknown Company",
                    "location": row['location'] if not pd.isna(row['location']) else "Remote",
                    "salary": str(row['salary']) if not pd.isna(row['salary']) else "",
                    "description": row['description'],
                    "requirements": row['requirements'].split('\n') if not pd.isna(row['requirements']) else [],
                    "postedDate": row['posted_date'] if not pd.isna(row['posted_date']) else datetime.now().isoformat(),
                    "similarityScore": 0.0  # Search results don't have match scores
                }
                search_results.append(job)
        
        # Sort by title to ensure consistent ordering
        search_results.sort(key=lambda x: x['title'])
        
        return {"jobs": search_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/jobs")
async def add_job(job: AddJobRequest):
    try:
        # Initialize DataFrame with correct columns if file doesn't exist
        if not os.path.exists('backend/job_title_des.csv'):
            jobs_df = pd.DataFrame(columns=[
                'id', 'title', 'description', 'company', 
                'location', 'salary', 'requirements', 'posted_date', 
                'similarity_score'
            ])
        else:
            jobs_df = pd.read_csv('backend/job_title_des.csv')
        
        # Create new job entry with UUID and proper data handling
        new_job = {
            'id': str(uuid.uuid4()),  # Generate a unique UUID for the job
            'title': job.title.strip(),
            'description': job.description.strip(),
            'company': job.company.strip() if job.company else 'Unknown Company',
            'location': job.location.strip() if job.location else 'Remote',
            'salary': job.salary.strip() if job.salary else '',
            'requirements': ';'.join(job.requirements) if job.requirements else '',
            'posted_date': datetime.now().isoformat(),
            'similarity_score': 0.0  # Default score if resume matching fails
        }
        
        # Try to get match score from resume if available
        try:
            # First, get the most recent resume
            resumes = []
            if os.path.exists(UPLOAD_DIR):
                for filename in os.listdir(UPLOAD_DIR):
                    if filename.startswith("resume_"):
                        file_path = os.path.join(UPLOAD_DIR, filename)
                        file_info = {
                            "id": os.path.splitext(filename)[0],
                            "path": file_path,
                            "uploadedAt": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat(),
                        }
                        resumes.append(file_info)
            
            if resumes:
                # Get the most recent resume
                most_recent_resume = sorted(resumes, key=lambda x: x["uploadedAt"], reverse=True)[0]
                
                # Parse the resume
                parser = ResumeParser()
                resume_data = parser.parse_resume(most_recent_resume["path"])
                
                # Calculate match score
                match_score = job_matcher.calculate_similarity(resume_data["text"], job.description)
                if not pd.isna(match_score):
                    new_job['similarity_score'] = float(match_score)
        except Exception as e:
            print(f"Error calculating match score: {str(e)}")
            # Continue with default score of 0.0
            pass
        
        # Append new job to DataFrame
        jobs_df = pd.concat([jobs_df, pd.DataFrame([new_job])], ignore_index=True)
        
        # Save updated DataFrame
        jobs_df.to_csv('backend/job_title_des.csv', index=False)
        
        # Return the job with the correct field names for the frontend
        return {
            "message": "Job added successfully", 
            "job": {
                "id": new_job['id'],
                "title": new_job['title'],
                "description": new_job['description'],
                "company": new_job['company'],
                "location": new_job['location'],
                "salary": new_job['salary'],
                "requirements": new_job['requirements'].split(';') if new_job['requirements'] else [],
                "postedDate": new_job['posted_date'],
                "similarityScore": new_job['similarity_score']
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-resume")
async def upload_resume(
    file: UploadFile = File(...),
    token: str = Depends(oauth2_scheme)
):
    try:
        # Validate file
        if not validate_file_type(file):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. Only PDF, DOC, and DOCX files are allowed."
            )
        
        if not validate_file_size(file):
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE/1024/1024}MB"
            )

        # Create upload directory if it doesn't exist
        os.makedirs(UPLOAD_DIR, exist_ok=True)
        
        # Generate a unique filename with timestamp
        file_extension = file.filename.split('.')[-1].lower()
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"resume_{timestamp}.{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, filename)
        
        # Save the file
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Parse the resume
        parser = ResumeParser()
        resume_data = parser.parse_resume(file_path)
        
        # Create a resume record in the database
        resume_id = f"resume_{timestamp}"
        resume_record = {
            "id": resume_id,
            "filename": filename,
            "path": file_path,
            "upload_date": datetime.now().isoformat(),
            "text": resume_data.get("text", ""),
            "skills": resume_data.get("skills", []),
            "education": resume_data.get("education", []),
            "experience": resume_data.get("experience", [])
        }
        
        # Save resume record
        resumes_file = "resumes.json"
        resumes = []
        if os.path.exists(resumes_file):
            with open(resumes_file, "r") as f:
                try:
                    resumes = json.load(f)
                except json.JSONDecodeError:
                    resumes = []
        
        resumes.append(resume_record)
        with open(resumes_file, "w") as f:
            json.dump(resumes, f)
        
        return {
            "id": resume_id,
            "filename": filename,
            "uploadDate": resume_record["upload_date"],
            "skills": resume_data.get("skills", []),
            "message": "Resume uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/jobs/{job_id}/tailor-resume")
async def tailor_resume_endpoint(job_id: str):
    logger.info(f"Received resume tailoring request for job_id: {job_id}")
    
    try:
        # Get most recent resume
        logger.info("Looking for most recent resume")
        resume_files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith("resume_")]
        if not resume_files:
            logger.error("No resume files found")
            raise HTTPException(status_code=404, detail="No resume found. Please upload a resume first.")
        
        latest_resume = max(resume_files, key=lambda x: os.path.getctime(os.path.join(UPLOAD_DIR, x)))
        resume_path = os.path.join(UPLOAD_DIR, latest_resume)
        logger.info(f"Using resume file: {latest_resume}")
        
        # Parse resume
        logger.info("Parsing resume")
        parser = ResumeParser()
        parsed_data = parser.parse_resume(resume_path)
        resume_text = parsed_data["text"]
        logger.debug(f"Resume text length: {len(resume_text)}")
        
        # Get job description
        logger.info("Loading job data")
        jobs_df = pd.read_csv('backend/job_title_des.csv')
        
        try:
            # Convert job_id to string for comparison
            job = jobs_df[jobs_df['id'].astype(str) == str(job_id)].iloc[0]
            job_description = job['description']
            logger.debug(f"Job description length: {len(job_description)}")
        except IndexError:
            logger.error(f"Job with ID {job_id} not found in database")
            raise HTTPException(status_code=404, detail="Job not found")
        except Exception as e:
            logger.error(f"Error accessing job data: {e}")
            raise HTTPException(status_code=500, detail="Error accessing job data")
        
        # Generate tailored resume suggestions
        logger.info("Generating tailored resume suggestions")
        try:
            from gemini_service import tailor_resume as generate_tailored_resume
            
            suggestions = generate_tailored_resume(resume_text, job_description)
            
            # Transform the suggestions into a more structured format for the frontend
            sections = [
                {
                    "title": "Key Changes to Make",
                    "content": "\n".join([f"• {edit}" for edit in suggestions["specific_edits"]])
                },
                {
                    "title": "Sections to Focus On",
                    "content": "\n".join([f"• {section}" for section in suggestions["sections_to_focus"]])
                },
                {
                    "title": "Keywords to Include",
                    "content": "\n".join([f"• {keyword}" for keyword in suggestions["keywords"]])
                },
                {
                    "title": "Skills to Emphasize",
                    "content": "\n".join([f"• {skill}" for skill in suggestions["skills_to_emphasize"]])
                },
                {
                    "title": "Experience to Highlight",
                    "content": "\n".join([f"• {exp}" for exp in suggestions["experience_to_highlight"]])
                }
            ]
            
            logger.info("Successfully generated tailored resume suggestions")
            return JSONResponse(content={"sections": sections})
        except Exception as e:
            logger.error(f"Error in tailor_resume function: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in tailor_resume endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/jobs/{job_id}/generate-cover-letter")
async def generate_cover_letter_endpoint(job_id: str):
    logger.info(f"Received cover letter generation request for job_id: {job_id}")
    
    try:
        # Get most recent resume
        logger.info("Looking for most recent resume")
        resume_files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith("resume_")]
        if not resume_files:
            logger.error("No resume files found")
            raise HTTPException(status_code=404, detail="No resume found. Please upload a resume first.")
        
        latest_resume = max(resume_files, key=lambda x: os.path.getctime(os.path.join(UPLOAD_DIR, x)))
        resume_path = os.path.join(UPLOAD_DIR, latest_resume)
        logger.info(f"Using resume file: {latest_resume}")
        
        # Parse resume
        logger.info("Parsing resume")
        parser = ResumeParser()
        parsed_data = parser.parse_resume(resume_path)
        resume_text = parsed_data["text"]
        logger.debug(f"Resume text length: {len(resume_text)}")
        
        # Get job description
        logger.info("Loading job data")
        jobs_df = pd.read_csv('backend/job_title_des.csv')
        
        try:
            # Convert job_id to string for comparison
            job = jobs_df[jobs_df['id'].astype(str) == str(job_id)].iloc[0]
            job_description = job['description']
            job_title = job['title']
            company = job['company'] if 'company' in job and not pd.isna(job['company']) else "The Company"
            logger.debug(f"Job description length: {len(job_description)}")
        except IndexError:
            logger.error(f"Job with ID {job_id} not found in database")
            raise HTTPException(status_code=404, detail="Job not found")
        except Exception as e:
            logger.error(f"Error accessing job data: {e}")
            raise HTTPException(status_code=500, detail="Error accessing job data")
        
        # Generate cover letter
        logger.info("Generating cover letter")
        try:
            from gemini_service import generate_cover_letter as gemini_generate_cover_letter
            
            cover_letter = gemini_generate_cover_letter(resume_text, job_description)
            logger.info("Successfully generated cover letter")
            
            return JSONResponse(content={
                "coverLetter": cover_letter,
                "jobTitle": job_title,
                "company": company
            })
        except Exception as e:
            logger.error(f"Error in generate_cover_letter function: {e}")
            raise HTTPException(status_code=500, detail=str(e))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_cover_letter endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Configure logging
try:
    logging.config.fileConfig('logging.ini')
except Exception as e:
    # Fallback to basic configuration if logging.ini is missing or invalid
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
logger = logging.getLogger('app')

@app.post("/jobs/{job_id}/generate-roadmap")
async def generate_roadmap(job_id: str):
    logger.info(f"Received roadmap generation request for job_id: {job_id}")
    
    try:
        # Get most recent resume
        logger.info("Looking for most recent resume")
        resume_files = [f for f in os.listdir(UPLOAD_DIR) if f.startswith("resume_")]
        if not resume_files:
            logger.error("No resume files found")
            raise HTTPException(status_code=404, detail="No resume found. Please upload a resume first.")
        
        latest_resume = max(resume_files, key=lambda x: os.path.getctime(os.path.join(UPLOAD_DIR, x)))
        resume_path = os.path.join(UPLOAD_DIR, latest_resume)
        logger.info(f"Using resume file: {latest_resume}")
        
        # Parse resume
        logger.info("Parsing resume")
        parser = ResumeParser()
        parsed_data = parser.parse_resume(resume_path)
        resume_text = parsed_data["text"]
        logger.debug(f"Resume text length: {len(resume_text)}")
        logger.debug(f"First 100 chars of resume: {resume_text[:100]}")
        
        # Get job description
        logger.info("Loading job data")
        jobs_df = pd.read_csv('backend/job_title_des.csv')
        logger.debug(f"Found {len(jobs_df)} jobs in database")
        logger.debug(f"Columns in jobs_df: {jobs_df.columns.tolist()}")
        logger.debug(f"Sample job IDs: {jobs_df['id'].head().tolist()}")
        
        try:
            # Convert job_id to string for comparison
            job = jobs_df[jobs_df['id'].astype(str) == str(job_id)].iloc[0]
            job_description = job['description']
            logger.debug(f"Job description length: {len(job_description)}")
            logger.debug(f"First 100 chars of job description: {job_description[:100]}")
            logger.debug(f"Job data: {job.to_dict()}")
        except IndexError:
            logger.error(f"Job with ID {job_id} not found in database")
            logger.error(f"Available job IDs: {jobs_df['id'].astype(str).tolist()}")
            raise HTTPException(status_code=404, detail="Job not found")
        except Exception as e:
            logger.error(f"Error accessing job data: {e}")
            logger.error(f"Error type: {type(e)}")
            raise HTTPException(status_code=500, detail="Error accessing job data")
        
        # Generate roadmap
        logger.info("Generating roadmap")
        try:
            roadmap = generate_roadmap_suggestions(resume_text, job_description)
            logger.info("Successfully generated roadmap")
            logger.debug(f"Roadmap content: {roadmap}")
            return JSONResponse(content={"roadmap": roadmap})
        except Exception as e:
            logger.error(f"Error in generate_roadmap function: {e}")
            logger.error(f"Error type: {type(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in generate_roadmap endpoint: {e}")
        logger.error(f"Error type: {type(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 