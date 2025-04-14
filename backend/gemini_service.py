import google.generativeai as genai
import os
import json
import logging
from typing import List, Dict, Any
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
model = genai.GenerativeModel('models/gemini-1.5-pro-001')

def tailor_resume(resume_text: str, job_description: str) -> Dict[str, Any]:
    """Generate specific resume tailoring suggestions based on the job description."""
    logger.info("Starting resume tailoring")
    logger.debug(f"Resume text length: {len(resume_text)}")
    logger.debug(f"Job description length: {len(job_description)}")
    
    prompt = f"""
    Analyze this resume and job description to provide specific, actionable suggestions for tailoring the resume.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Provide:
    1. Specific edits to make to the resume
    2. Which sections need the most attention
    3. Keywords to include
    4. Skills to emphasize
    5. Experience to highlight
    
    Format the response as a JSON object with these keys:
    - specific_edits: List of specific changes to make
    - sections_to_focus: List of sections needing attention
    - keywords: List of keywords to include
    - skills_to_emphasize: List of skills to emphasize
    - experience_to_highlight: List of experience points to highlight
    """
    
    try:
        logger.info("Generating content with Gemini")
        response = model.generate_content(prompt)
        logger.debug(f"Raw response: {response.text}")
        
        # Extract JSON from the response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        elif response_text.startswith('```'):
            # Handle case where language isn't specified in markdown
            code_block_parts = response_text.split('```')
            if len(code_block_parts) >= 3:  # At least one code block
                response_text = code_block_parts[1].strip()
        
        # Clean the response text to ensure valid JSON
        # Remove any non-JSON text before and after the JSON object
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        if json_start >= 0 and json_end > json_start:
            response_text = response_text[json_start:json_end]
        
        logger.info("Attempting to parse JSON response")
        result = json.loads(response_text)
        logger.info("Successfully parsed JSON response")
        
        # Ensure all expected keys are present
        default_keys = {
            "specific_edits": [],
            "sections_to_focus": [],
            "keywords": [],
            "skills_to_emphasize": [],
            "experience_to_highlight": []
        }
        
        for key, default_value in default_keys.items():
            if key not in result or not isinstance(result[key], list):
                result[key] = default_value
        
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Response text: {response_text}")
        return {
            "specific_edits": ["Highlight relevant skills and experience", 
                              "Quantify achievements where possible",
                              "Tailor your professional summary to match job requirements"],
            "sections_to_focus": ["Skills section", "Professional experience", "Summary/Objective"],
            "keywords": ["relevant keywords from the job description"],
            "skills_to_emphasize": ["key skills matching the job requirements"],
            "experience_to_highlight": ["relevant experience that matches the job"]
        }
    except Exception as e:
        logger.error(f"Unexpected error in tailor_resume: {e}")
        raise

def generate_cover_letter(resume_text: str, job_description: str) -> str:
    """Generate a professional cover letter based on the resume and job description."""
    logger.info("Starting cover letter generation")
    logger.debug(f"Resume text length: {len(resume_text)}")
    logger.debug(f"Job description length: {len(job_description)}")
    
    prompt = f"""
    Write a professional cover letter that:
    1. Matches the candidate's experience from their resume with the job requirements
    2. Highlights specific strengths and achievements
    3. Demonstrates enthusiasm for the position
    4. Is tailored to the specific company and role
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Format the response as a professional cover letter with proper salutation and closing.
    Do not include placeholder text like [Your Name] or [Company Name]. Use appropriate generic text instead.
    """
    
    try:
        logger.info("Generating content with Gemini")
        response = model.generate_content(prompt)
        logger.info("Successfully generated cover letter")
        
        # Clean the response if it's enclosed in quotation marks or markdown
        cover_letter = response.text.strip()
        if cover_letter.startswith('```') and cover_letter.endswith('```'):
            cover_letter = cover_letter[3:-3].strip()
        if cover_letter.startswith('"') and cover_letter.endswith('"'):
            cover_letter = cover_letter[1:-1].strip()
            
        return cover_letter
    except Exception as e:
        logger.error(f"Error in generate_cover_letter: {e}")
        raise

def generate_roadmap(resume_text: str, job_description: str) -> List[Dict[str, Any]]:
    """Generate a development roadmap with specific milestones."""
    logger.info("Starting roadmap generation")
    logger.debug(f"Resume text length: {len(resume_text)}")
    logger.debug(f"Job description length: {len(job_description)}")
    
    prompt = f"""
    Analyze the gaps between this resume and job description to create a development roadmap.
    
    Resume:
    {resume_text}
    
    Job Description:
    {job_description}
    
    Generate 5 specific, actionable milestones that would help bridge the gaps. For each milestone:
    1. Provide a clear title
    2. Give a detailed description
    3. Suggest a timeframe for completion
    4. List specific skills or knowledge to be gained
    5. Include potential resources or methods to achieve it
    
    Format the response as a JSON array of objects, each with these keys:
    - title: String
    - description: String
    - timeframe: String (e.g., "2-3 months")
    - skills: List of strings
    - resources: List of strings
    """
    
    try:
        logger.info("Generating content with Gemini")
        response = model.generate_content(prompt)
        logger.debug(f"Raw response: {response.text}")
        
        # Extract JSON from the response
        response_text = response.text.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:-3].strip()
        
        logger.info("Attempting to parse JSON response")
        result = json.loads(response_text)
        logger.info("Successfully parsed JSON response")
        return result
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {e}")
        logger.error(f"Response text: {response_text}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error in generate_roadmap: {e}")
        raise 