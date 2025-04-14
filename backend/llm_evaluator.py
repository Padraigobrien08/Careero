import google.generativeai as genai
from typing import Dict, Any
import json

class LLMEvaluator:
    def __init__(self, api_key: str):
        """Initialize the LLM evaluator with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

    def evaluate_candidate(self, resume_text: str, job_title: str, job_description: str) -> Dict[str, Any]:
        """Evaluate candidate's qualifications using Gemini"""
        try:
            # Construct the prompt
            prompt = f"""
            Rate how qualified the candidate is for this job.
            
            Job Title: {job_title}
            Job Description: {job_description}
            
            Candidate's Resume:
            {resume_text}
            
            Please provide a structured evaluation in JSON format with the following fields:
            - score: A number between 0 and 100 indicating overall qualification
            - explanation: A detailed explanation of the rating
            - strengths: List of candidate's strengths for this role
            - gaps: List of areas where the candidate falls short
            """
            
            # Get response from Gemini
            response = self.model.generate_content(prompt)
            
            # Parse the response
            try:
                evaluation = json.loads(response.text)
                return evaluation
            except json.JSONDecodeError:
                # If response is not valid JSON, create a structured response
                return {
                    "score": 0,
                    "explanation": "Unable to parse LLM response",
                    "strengths": [],
                    "gaps": []
                }
                
        except Exception as e:
            print(f"Error in LLM evaluation: {str(e)}")
            return {
                "score": 0,
                "explanation": f"Error during evaluation: {str(e)}",
                "strengths": [],
                "gaps": []
            } 