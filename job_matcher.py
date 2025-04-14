import re
from typing import List, Dict
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

class JobMatcher:
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.vectorizer = None
        self.job_vectors = None
        self.load_data()
    
    def load_data(self):
        """Load and preprocess the job data"""
        try:
            # Read CSV
            self.df = pd.read_csv(self.csv_path, encoding='utf-8')
            
            # Print available columns for debugging
            print(f"Available columns: {list(self.df.columns)}")
            print(f"Successfully loaded {len(self.df)} jobs")
            
            # Handle different column naming conventions
            # Map column names to standardized names
            column_mapping = {
                'Job Description': 'description',
                'Job Title': 'title'
            }
            
            # Rename columns if old format exists
            for old_col, new_col in column_mapping.items():
                if old_col in self.df.columns and new_col not in self.df.columns:
                    self.df[new_col] = self.df[old_col]
            
            # Ensure required columns exist
            if 'description' not in self.df.columns:
                raise Exception("Required column 'description' not found in the job data")
            
            print(f"Final column names: {list(self.df.columns)}")
            
            # Clean and preprocess job descriptions
            self.df['processed_description'] = self.df['description'].apply(self.preprocess_text)
            
            # Initialize TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=1000
            )
            
            # Fit and transform job descriptions
            self.job_vectors = self.vectorizer.fit_transform(self.df['processed_description'])
            
        except Exception as e:
            raise Exception(f"Error loading job data: {str(e)}")
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for matching"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def extract_keywords(self, resume_text: str) -> List[str]:
        """Extract relevant keywords from resume"""
        # Preprocess resume text
        processed_text = self.preprocess_text(resume_text)
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(processed_text)
        stop_words = set(stopwords.words('english'))
        keywords = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        # Get most common keywords
        keyword_counts = Counter(keywords)
        top_keywords = [word for word, count in keyword_counts.most_common(20)]
        
        return top_keywords
    
    def find_matching_jobs(self, resume_text: str, top_n: int = 5) -> List[Dict]:
        """Find the most relevant jobs for the resume"""
        try:
            # Preprocess resume text
            processed_resume = self.preprocess_text(resume_text)
            
            # Transform resume text using the same vectorizer
            resume_vector = self.vectorizer.transform([processed_resume])
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(resume_vector, self.job_vectors).flatten()
            
            # Get top matching jobs
            top_indices = similarity_scores.argsort()[-top_n:][::-1]
            
            # Prepare results
            results = []
            for idx in top_indices:
                job = self.df.iloc[idx]
                # Convert numpy types to Python native types
                similarity_score = float(similarity_scores[idx])  # Convert numpy.float64 to float
                results.append({
                    'id': int(idx),  # Convert numpy.int64 to int
                    'title': str(job['title']),  # Ensure string type
                    'description': str(job['description']),  # Ensure string type
                    'similarity_score': similarity_score
                })
            
            return results
            
        except Exception as e:
            raise Exception(f"Error finding matching jobs: {str(e)}")
    
    def get_job_details(self, job_id: int) -> Dict:
        """Get detailed information about a specific job"""
        try:
            job = self.df.iloc[job_id]
            return {
                'id': int(job_id),  # Convert numpy.int64 to int
                'title': str(job['title']),  # Ensure string type
                'description': str(job['description'])  # Ensure string type
            }
        except Exception as e:
            raise Exception(f"Error getting job details: {str(e)}")
    
    def match_jobs(self, resume_text: str) -> Dict[str, float]:
        """Match resume text against jobs and return similarity scores"""
        if self.df is None or len(self.df) == 0:
            return {}
        
        try:
            # Process the resume text
            processed_resume = self.preprocess_text(resume_text)
            
            # Create a dictionary to store job_id -> similarity_score
            similarity_dict = {}
            
            # If we have a vectorizer, use it for all jobs at once
            if self.vectorizer and self.job_vectors is not None:
                resume_vector = self.vectorizer.transform([processed_resume])
                all_scores = cosine_similarity(resume_vector, self.job_vectors).flatten()
                
                for idx, score in enumerate(all_scores):
                    # Get the job ID as a string
                    job_id = str(self.df.iloc[idx]['id'])
                    similarity_dict[job_id] = float(score)
            else:
                # Fallback to individual comparisons
                for idx, row in self.df.iterrows():
                    job_id = str(row['id'])
                    job_description = row['description']
                    score = self.calculate_similarity(processed_resume, job_description)
                    similarity_dict[job_id] = score
            
            return similarity_dict
            
        except Exception as e:
            print(f"Error in match_jobs: {str(e)}")
            return {}
    
    def calculate_similarity(self, resume_text: str, job_description: str) -> float:
        """Calculate similarity between resume text and job description"""
        try:
            # Preprocess texts
            processed_resume = self.preprocess_text(resume_text)
            processed_job = self.preprocess_text(job_description)
            
            # Create temporary vectorizer for this comparison
            temp_vectorizer = TfidfVectorizer(stop_words='english')
            
            # Transform both texts
            tfidf_matrix = temp_vectorizer.fit_transform([processed_resume, processed_job])
            
            # Calculate cosine similarity
            sim_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            
            return float(sim_score)
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0 