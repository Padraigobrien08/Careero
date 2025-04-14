import google.generativeai as genai
import json
from typing import Dict, Optional

class LLMEvaluator:
    def __init__(self, api_key: str):
        """Initialize the Gemini model with API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def evaluate_candidate(self, resume_text: str, job_title: str, job_description: str) -> Dict:
        """
        Evaluate how qualified the candidate is for the job using Gemini.
        
        Args:
            resume_text (str): The candidate's resume text
            job_title (str): The job title
            job_description (str): The job description
            
        Returns:
            Dict: Evaluation results including score and explanation
        """
        try:
            # Create the prompt
            prompt = f"""
            You are an expert hiring manager. Please evaluate how qualified the candidate is for this job.
            
            Job Title: {job_title}
            Job Description: {job_description}
            
            Candidate's Resume:
            {resume_text}
            
            Please provide:
            1. A score from 0-100 indicating how qualified the candidate is
            2. A detailed explanation of your evaluation
            3. Key strengths that match the job requirements
            4. Any potential gaps or areas for improvement
            
            Format your response as a JSON object with these keys:
            - score (integer)
            - explanation (string)
            - strengths (list of strings)
            - gaps (list of strings)
            """
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Write response to output.txt
            with open('output.txt', 'w') as f:
                f.write(response.text)
            
            # Parse the response
            try:
                # Extract JSON from the response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                
                # Parse JSON safely
                evaluation = json.loads(response_text)
                
                # Ensure all required fields are present with default values
                evaluation.setdefault('score', 0)
                evaluation.setdefault('explanation', 'No explanation provided')
                evaluation.setdefault('strengths', [])
                evaluation.setdefault('gaps', [])
                
                # Ensure gaps is a list
                if not isinstance(evaluation['gaps'], list):
                    evaluation['gaps'] = [str(evaluation['gaps'])]
                
                return evaluation
                
            except json.JSONDecodeError as e:
                print(f"Failed to parse LLM response: {str(e)}")
                print(f"Raw response: {response.text}")
                return {
                    'score': 0,
                    'explanation': 'Failed to parse evaluation',
                    'strengths': [],
                    'gaps': ['Failed to generate evaluation']
                }
                
        except Exception as e:
            print(f"Error in LLM evaluation: {str(e)}")
            return {
                'score': 0,
                'explanation': 'Error in evaluation',
                'strengths': [],
                'gaps': ['Error in evaluation']
            } 