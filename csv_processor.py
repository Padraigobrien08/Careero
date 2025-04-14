import pandas as pd
import re
from typing import List, Dict, Optional

class JobDataProcessor:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the CSV data"""
        try:
            # Read CSV with proper encoding and handle potential issues
            self.df = pd.read_csv(self.csv_path, encoding='utf-8', on_bad_lines='skip')
            
            # Print column names for debugging
            print("Available columns:", self.df.columns.tolist())
            
            # Rename columns to match expected format
            column_mapping = {
                'Job Title': 'title',
                'Job Description': 'description',
                'index': 'id'  # Use the index column as ID
            }
            
            # Rename columns that exist in the mapping
            for old_col, new_col in column_mapping.items():
                if old_col in self.df.columns:
                    self.df.rename(columns={old_col: new_col}, inplace=True)
            
            # Ensure required columns exist
            required_columns = ['title', 'description']
            missing_columns = [col for col in required_columns if col not in self.df.columns]
            if missing_columns:
                raise Exception(f"Missing required columns: {missing_columns}")
            
            # Clean text data
            text_columns = self.df.select_dtypes(include=['object']).columns
            for col in text_columns:
                self.df[col] = self.df[col].apply(self.clean_text)
            
            # Add ID if not present
            if 'id' not in self.df.columns:
                self.df['id'] = range(1, len(self.df) + 1)
            
            print(f"Successfully loaded {len(self.df)} jobs")
            print("Final column names:", self.df.columns.tolist())
                
        except Exception as e:
            raise Exception(f"Error loading CSV data: {str(e)}")
    
    def clean_text(self, text: str) -> str:
        """Clean and format text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string if not already
        text = str(text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:()\-]', ' ', text)
        
        return text.strip()
    
    def search_jobs(self, query: Optional[str] = None, limit: int = 10) -> List[Dict]:
        """Search for jobs based on a query"""
        try:
            if query:
                # Create a search mask for title and description
                search_mask = (
                    self.df['title'].str.contains(query, case=False, na=False) |
                    self.df['description'].str.contains(query, case=False, na=False)
                )
                results = self.df[search_mask]
            else:
                results = self.df
            
            # Convert to list of dictionaries
            jobs = results.head(limit).to_dict('records')
            
            # Clean up the results
            for job in jobs:
                for key, value in job.items():
                    if pd.isna(value):
                        job[key] = None
                    elif isinstance(value, str):
                        job[key] = value.strip()
            
            return jobs
            
        except Exception as e:
            raise Exception(f"Error searching jobs: {str(e)}")
    
    def get_job_by_id(self, job_id: str) -> Optional[Dict]:
        """Get a specific job by ID"""
        try:
            # Convert job_id to integer if possible
            try:
                job_id = int(job_id)
            except ValueError:
                pass
            
            # Try to find the job
            job = self.df[self.df['id'] == job_id]
            
            if len(job) == 0:
                return None
            
            job_dict = job.iloc[0].to_dict()
            
            # Clean up the result
            for key, value in job_dict.items():
                if pd.isna(value):
                    job_dict[key] = None
                elif isinstance(value, str):
                    job_dict[key] = value.strip()
            
            return job_dict
            
        except Exception as e:
            raise Exception(f"Error getting job by ID: {str(e)}") 