# Cloud Deployment Guide for CareeroOS Backend

This guide explains how to deploy the CareeroOS backend to cloud services like Google Cloud Run and Heroku.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) installed locally
- Account on your chosen cloud platform (Google Cloud, Heroku, etc.)
- Cloud CLI tools installed (optional but recommended)

## Building the Docker Image

Before deploying, build and test your Docker image locally:

```bash
# Build the backend image
cd /path/to/CareeroOS
docker build -t careeroos-backend -f backend/Dockerfile .

# Test locally
docker run -p 8080:8000 careeroos-backend
```

Visit http://localhost:8080 to verify it's working.

## Deploying to Google Cloud Run

### 1. Authenticate with Google Cloud

```bash
gcloud auth login
gcloud config set project [YOUR_PROJECT_ID]
```

### 2. Build and Push the Image

```bash
# Build and tag the image for Google Container Registry
docker build -t gcr.io/[YOUR_PROJECT_ID]/careeroos-backend -f backend/Dockerfile .

# Push the image
docker push gcr.io/[YOUR_PROJECT_ID]/careeroos-backend
```

### 3. Deploy to Cloud Run

```bash
gcloud run deploy careeroos-backend \
  --image gcr.io/[YOUR_PROJECT_ID]/careeroos-backend \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --set-env-vars="ALLOWED_ORIGINS=https://your-frontend-domain.com"
```

## Deploying to Heroku

### 1. Login to Heroku

```bash
heroku login
heroku container:login
```

### 2. Create a Heroku App

```bash
heroku create careeroos-backend
```

### 3. Build and Push

```bash
# Build and push directly to Heroku
heroku container:push web -a careeroos-backend

# Release the container
heroku container:release web -a careeroos-backend
```

### 4. Configure Environment Variables

```bash
heroku config:set ALLOWED_ORIGINS=https://your-frontend-domain.com -a careeroos-backend
```

## Configuration for Production

For both platforms, you'll need to set these environment variables:

- `ALLOWED_ORIGINS`: Set to your frontend domain(s) separated by commas
- `SECRET_KEY`: A secure random string for JWT token signing

## Database and Storage Considerations

For a production deployment, consider:

1. **Persistence**: Both Cloud Run and Heroku containers are ephemeral. Use cloud storage services:
   - Google Cloud Storage for Google Cloud Run
   - AWS S3 or similar for Heroku

2. **Database**: Replace local files with a proper database:
   - Google Cloud SQL with PostgreSQL
   - Heroku PostgreSQL add-on

## Connecting Your Frontend

Update your frontend environment variables to point to your cloud backend URL:

- For Vercel: Set `REACT_APP_API_BASE_URL` to your Cloud Run/Heroku URL

## Monitoring and Scaling

- Google Cloud Run: Use Cloud Monitoring and set up auto-scaling
- Heroku: Monitor with the Heroku dashboard and upgrade dynos as needed

## Troubleshooting

### CORS Issues

If you encounter CORS errors after deployment:
1. Verify `ALLOWED_ORIGINS` is set correctly
2. Check frontend requests use the correct protocol (https)

### Container Crashes

View logs to diagnose issues:
- Google Cloud Run: `gcloud logs read --limit=50 --service=careeroos-backend`
- Heroku: `heroku logs --tail -a careeroos-backend`
