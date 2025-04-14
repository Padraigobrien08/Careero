import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re

def scrape_job_description(url: str) -> str:
    """
    Scrape job description from a given URL.
    
    Args:
        url (str): URL of the job posting
        
    Returns:
        str: Extracted job description
    """
    try:
        # Add headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text = soup.get_text()
        
        # Clean up the text
        text = clean_job_text(text)
        
        return text
    except requests.exceptions.RequestException as e:
        raise Exception(f"Error fetching job description: {str(e)}")
    except Exception as e:
        raise Exception(f"Error processing job description: {str(e)}")

def clean_job_text(text: str) -> str:
    """
    Clean the scraped job description text.
    
    Args:
        text (str): Raw text from job posting
        
    Returns:
        str: Cleaned job description
    """
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Remove multiple newlines
    text = re.sub(r'\n+', '\n', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove common unwanted patterns
    text = re.sub(r'Share this job.*', '', text, flags=re.DOTALL)
    text = re.sub(r'Apply now.*', '', text, flags=re.DOTALL)
    
    return text 