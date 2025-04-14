import google.generativeai as genai
import json
import os
from typing import Dict, List, Any

class ResumeImprover:
    def __init__(self, api_key: str):
        """Initialize the ResumeImprover with Gemini API key"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')
        
    def generate_improvement_plan(self, evaluation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate a detailed improvement plan based on the evaluation"""
        try:
            # Extract gaps from the evaluation
            gaps = evaluation.get('gaps', [])
            
            # Construct the prompt
            prompt = f"""
            Based on the following gaps in a candidate's resume, provide detailed, actionable recommendations 
            to improve their profile. For each gap, suggest:
            1. Specific projects they could work on
            2. Relevant certifications or courses
            3. Practical steps to gain experience
            4. Resources or platforms to use
            5. Timeline estimates for each recommendation
            
            Gaps identified:
            {json.dumps(gaps, indent=2)}
            
            Provide your response in the following JSON format:
            [
                {{
                    "gap": "original gap text",
                    "recommendations": [
                        {{
                            "type": "project/certification/experience",
                            "title": "recommendation title",
                            "description": "detailed description",
                            "resources": ["list of resources"],
                            "timeline": "estimated time to complete",
                            "priority": "high/medium/low"
                        }}
                    ]
                }}
            ]
            """
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Write response to output.txt
            with open('output.txt', 'a') as f:
                f.write('\n\nImprovement Plan:\n')
                f.write(response.text)
            
            # Parse the response
            try:
                # Extract JSON from the response
                response_text = response.text.strip()
                if response_text.startswith('```json'):
                    response_text = response_text[7:-3].strip()
                improvement_plan = json.loads(response_text)
                return improvement_plan
            except json.JSONDecodeError as e:
                print(f"Failed to parse LLM response: {str(e)}")
                print(f"Raw response: {response.text}")
                return []
                
        except Exception as e:
            print(f"Error generating improvement plan: {str(e)}")
            return []

def format_improvement_plan(plan: Dict[str, Any]) -> None:
    """Format and write the improvement plan to output.txt"""
    output = []
    output.append("\n" + "="*80)
    output.append("Resume Improvement Plan")
    output.append("="*80)
    
    if "error" in plan:
        output.append(f"\nError: {plan['error']}")
        if "raw_response" in plan:
            output.append("\nRaw response:")
            output.append(plan['raw_response'])
    else:
        for gap_plan in plan.get('improvement_plan', []):
            output.append(f"\nGap: {gap_plan['gap']}")
            output.append("-"*80)
            
            for rec in gap_plan['recommendations']:
                output.append(f"\n{rec['type'].upper()}: {rec['title']}")
                output.append(f"Priority: {rec['priority']}")
                output.append(f"Timeline: {rec['timeline']}")
                output.append("\nDescription:")
                output.append(rec['description'])
                
                if rec['resources']:
                    output.append("\nResources:")
                    for resource in rec['resources']:
                        output.append(f"• {resource}")
                
                output.append("\n" + "-"*40)
    
    output.append("\n" + "="*80)
    
    # Write to output.txt
    with open('output.txt', 'a') as f:
        f.write('\n'.join(output))

if __name__ == "__main__":
    # Example usage
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    improver = ResumeImprover(api_key)
    
    # Example evaluation (you would get this from the previous LLM evaluation)
    example_evaluation = {
        "gaps": [
            "Lack of explicit detail on experience building end-to-end machine learning pipelines",
            "Unspecified experience with ETL pipelines and data streaming",
            "No mention of specific experience in classical machine learning algorithms",
            "Absence of demonstrable experience in Computer Vision and NLP"
        ]
    }
    
    improvement_plan = improver.generate_improvement_plan(example_evaluation)
    format_improvement_plan(improvement_plan) 