import re
from typing import Any, Dict, List
import html
from datetime import datetime

def sanitize_input(data: Any) -> Any:
    """Sanitize input data to prevent XSS and injection attacks"""
    if isinstance(data, str):
        return html.escape(data.strip())
    elif isinstance(data, dict):
        return {k: sanitize_input(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_input(item) for item in data]
    return data

def validate_job_data(job_data: Dict[str, Any]) -> List[str]:
    """Validate job data and return list of errors"""
    errors = []
    
    # Title validation
    if not job_data.get('title'):
        errors.append("Job title is required")
    elif len(job_data['title']) > 200:
        errors.append("Job title is too long")
    
    # Description validation
    if not job_data.get('description'):
        errors.append("Job description is required")
    elif len(job_data['description']) > 5000:
        errors.append("Job description is too long")
    
    # Company validation
    if job_data.get('company') and len(job_data['company']) > 100:
        errors.append("Company name is too long")
    
    # Location validation
    if job_data.get('location') and len(job_data['location']) > 100:
        errors.append("Location is too long")
    
    # Salary validation
    if job_data.get('salary'):
        if not re.match(r'^\$?\d+(,\d{3})*(\.\d{2})?$', job_data['salary']):
            errors.append("Invalid salary format")
    
    # Requirements validation
    if job_data.get('requirements'):
        if not isinstance(job_data['requirements'], list):
            errors.append("Requirements must be a list")
        else:
            for req in job_data['requirements']:
                if len(req) > 500:
                    errors.append("Requirement is too long")
    
    return errors

def validate_resume_data(resume_data: Dict[str, Any]) -> List[str]:
    """Validate resume data and return list of errors"""
    errors = []
    
    # Text validation
    if not resume_data.get('text'):
        errors.append("Resume text is required")
    elif len(resume_data['text']) > 100000:
        errors.append("Resume text is too long")
    
    # Skills validation
    if resume_data.get('skills'):
        if not isinstance(resume_data['skills'], list):
            errors.append("Skills must be a list")
        else:
            for skill in resume_data['skills']:
                if len(skill) > 100:
                    errors.append("Skill name is too long")
    
    # Experience validation
    if resume_data.get('experience'):
        if not isinstance(resume_data['experience'], list):
            errors.append("Experience must be a list")
        else:
            for exp in resume_data['experience']:
                if not isinstance(exp, dict):
                    errors.append("Experience entry must be an object")
                else:
                    if 'title' in exp and len(exp['title']) > 200:
                        errors.append("Job title in experience is too long")
                    if 'company' in exp and len(exp['company']) > 100:
                        errors.append("Company name in experience is too long")
    
    # Education validation
    if resume_data.get('education'):
        if not isinstance(resume_data['education'], list):
            errors.append("Education must be a list")
        else:
            for edu in resume_data['education']:
                if not isinstance(edu, dict):
                    errors.append("Education entry must be an object")
                else:
                    if 'degree' in edu and len(edu['degree']) > 200:
                        errors.append("Degree name is too long")
                    if 'institution' in edu and len(edu['institution']) > 200:
                        errors.append("Institution name is too long")
    
    return errors

def validate_date(date_str: str) -> bool:
    """Validate date string format"""
    try:
        datetime.fromisoformat(date_str.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False 