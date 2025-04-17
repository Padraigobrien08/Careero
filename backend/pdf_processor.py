import PyPDF2
import os
import re
import unicodedata

class PDFProcessor:
    def __init__(self):
        pass

    def extract_text(self, pdf_path: str) -> str:
        """Extract text from a PDF file"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() or "" # Add fallback for empty pages
        return text

    def clean_text(self, text: str) -> str:
        """Basic text cleaning (placeholder - implement actual cleaning if needed)."""
        if not isinstance(text, str):
            return ""
        # Example basic cleaning: remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        # Add more cleaning rules here if necessary (e.g., remove headers/footers)
        print(f"PDFProcessor: Ran clean_text (returning cleaned text).") # Log when called
        return text

def extract_text_from_pdf(pdf_path: str) -> str:
    """Legacy function for backward compatibility"""
    processor = PDFProcessor()
    return processor.extract_text(pdf_path)

def clean_text(text: str) -> str:
    """
    Clean the extracted text by removing extra whitespace and normalizing line breaks.
    Preserves special characters and maintains basic formatting.
    
    Args:
        text (str): Raw text from PDF
        
    Returns:
        str: Cleaned text
    """
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKC', text)
    
    # Replace common problematic characters
    text = text.replace('•', '•')  # Keep bullet points
    text = text.replace('–', '–')  # Keep en-dash
    text = text.replace('—', '—')  # Keep em-dash
    
    # Preserve intentional line breaks while removing excessive ones
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Keep double newlines for paragraphs
    text = re.sub(r'\n+', '\n', text)  # Remove multiple newlines
    
    # Clean up spaces
    text = re.sub(r'[ \t]+', ' ', text)  # Replace multiple spaces/tabs with single space
    text = re.sub(r'\n ', '\n', text)  # Remove spaces after newlines
    text = re.sub(r' \n', '\n', text)  # Remove spaces before newlines
    
    # Format section headers (lines in all caps or with colons)
    text = re.sub(r'\n([A-Z][A-Z\s]+[A-Z])\n', r'\n\n\1\n', text)  # All caps headers
    text = re.sub(r'\n([^:]+:)\n', r'\n\n\1\n', text)  # Lines ending with colon
    
    # Format bullet points
    text = re.sub(r'\n\s*[•\-*]\s*', '\n• ', text)  # Standardize bullet points
    
    # Format dates and numbers
    text = re.sub(r'(\d{4})[-\s](\d{4})', r'\1–\2', text)  # Date ranges
    text = re.sub(r'(\d+)[\s,]+(\d+)', r'\1,\2', text)  # Numbers with commas
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text 