# Resume and Job Description Processor

A FastAPI-based backend service that can process PDF resumes and scrape job descriptions from URLs.

## Features

- PDF resume text extraction
- Job description scraping from URLs
- RESTful API endpoints
- CORS support for frontend integration

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the server:
```bash
python main.py
```

The server will start at `http://localhost:8000`

## API Endpoints

### Process Resume
- **Endpoint**: `/process-resume`
- **Method**: POST
- **Content-Type**: multipart/form-data
- **Body**: PDF file
- **Response**: Extracted text from the PDF

### Scrape Job Description
- **Endpoint**: `/scrape-job`
- **Method**: GET
- **Parameters**: `url` (query parameter)
- **Response**: Scraped job description text

## API Documentation

Once the server is running, you can access the interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Error Handling

The API includes proper error handling for:
- Invalid file types
- PDF processing errors
- URL scraping errors
- Network issues

## Security Notes

- In production, configure CORS with specific allowed origins
- Consider adding rate limiting
- Implement proper authentication/authorization
- Validate and sanitize all inputs 