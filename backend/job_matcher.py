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
from nltk.stem import WordNetLemmatizer
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

class JobMatcher:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
        self.job_descriptions = None
        self.tfidf_matrix = None
        self.lemmatizer = WordNetLemmatizer()

    def load_jobs(self, csv_path='job_title_des.csv'):
        """Load job data from CSV and prepare for matching"""
        try:
            df = pd.read_csv(csv_path, encoding='utf-8')
            # Clean up the DataFrame
            if 'Job Description' in df.columns and 'description' in df.columns:
                df['description'] = df['description'].fillna(df['Job Description'])
                df = df.drop('Job Description', axis=1)
            if 'Job Title' in df.columns and 'title' in df.columns:
                df['title'] = df['title'].fillna(df['Job Title'])
                df = df.drop('Job Title', axis=1)
            
            # Ensure we have the required columns
            required_columns = ['id', 'title', 'description']
            for col in required_columns:
                if col not in df.columns:
                    if col == 'id' and 'index' in df.columns:
                        df['id'] = df['index'].astype(str).str.split('.').str[0]
                    else:
                        df[col] = None
            
            # Combine title and description for better matching
            df['combined_text'] = df['title'].fillna('') + ' ' + df['description'].fillna('')
            
            # Remove rows with empty combined text
            df = df[df['combined_text'].str.strip() != '']
            
            # Fit the vectorizer and transform the job descriptions
            self.job_descriptions = df
            self.tfidf_matrix = self.vectorizer.fit_transform(df['combined_text'])
            
            return True
        except Exception as e:
            print(f"Error loading jobs: {str(e)}")
            return False

    def match_jobs(self, resume_text):
        """Match resume text against job descriptions"""
        try:
            if not isinstance(resume_text, str):
                raise ValueError("Resume text must be a string")
                
            if self.job_descriptions is None or self.tfidf_matrix is None:
                if not self.load_jobs():
                    raise Exception("Failed to load job data")
            
            # Transform resume text using the same vectorizer
            resume_vector = self.vectorizer.transform([resume_text])
            
            # Calculate similarity scores
            similarity_scores = cosine_similarity(resume_vector, self.tfidf_matrix).flatten()
            
            # Create a dictionary of job IDs and their similarity scores
            matches = {}
            for idx, score in enumerate(similarity_scores):
                job_id = str(self.job_descriptions.iloc[idx]['id'])
                matches[job_id] = float(score)
            
            return matches
        except Exception as e:
            print(f"Error finding matching jobs: {str(e)}")
            raise

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
    
    def calculate_similarity(self, resume_text: str, job_description: str) -> float:
        """Calculate similarity between resume and a single job description"""
        try:
            if not isinstance(resume_text, str) or not isinstance(job_description, str):
                raise ValueError("Both resume_text and job_description must be strings")
            
            # Transform both texts using the same vectorizer
            if self.vectorizer is None:
                self.vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
                self.vectorizer.fit([resume_text, job_description])
            
            vectors = self.vectorizer.transform([resume_text, job_description])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {str(e)}")
            return 0.0

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """Extract top keywords from text using TF-IDF"""
        try:
            # Create TF-IDF vector
            tfidf_matrix = self.vectorizer.transform([text])
            
            # Get feature names
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get top keywords
            top_keywords = []
            for i in tfidf_matrix[0].indices:
                score = tfidf_matrix[0, i]
                if score > 0:
                    top_keywords.append((feature_names[i], score))
            
            # Sort by score and get top N
            top_keywords.sort(key=lambda x: x[1], reverse=True)
            return [word for word, _ in top_keywords[:top_n]]
        except Exception as e:
            print(f"Error extracting keywords: {str(e)}")
            return [] 