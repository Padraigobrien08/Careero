FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libmagic1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy all code first
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r backend/requirements.txt

# Install local modules as packages
RUN pip install -e .

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV ALLOWED_ORIGINS="*"

# Expose the port
EXPOSE 8080

# Command to run the application - important to listen on 0.0.0.0 with PORT from env var
CMD exec uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080} 