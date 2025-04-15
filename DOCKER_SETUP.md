# Docker Setup for CareeroOS

This guide explains how to run the CareeroOS application using Docker.

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

## Running the Application

### Using Docker Compose (Recommended)

The easiest way to run both the frontend and backend together is with Docker Compose:

```bash
# Build and start both frontend and backend
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

The services will be available at:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8080

### Running Backend Only

If you want to run only the backend service:

```bash
# Build the backend image
docker build -t careeroos-backend -f backend/Dockerfile .

# Run the backend container
docker run -p 8080:8000 -v ./uploads:/app/uploads -v ./resumes.json:/app/resumes.json careeroos-backend
```

## Development with Docker

For development purposes, you can mount local directories to see your changes in real-time:

```bash
# Start with mounted volumes for development
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up
```

## Troubleshooting

### Container fails to start

Check the logs for error messages:

```bash
docker-compose logs backend
docker-compose logs frontend
```

### Port conflicts

If you get an error about ports already in use, make sure:
1. You've stopped any local instances of the backend running on port 8000
2. You can change the port mappings in docker-compose.yml if needed

### API connection issues

Make sure the `REACT_APP_API_BASE_URL` environment variable in the frontend service is correctly set to the backend URL (http://localhost:8080).

### Volume Permissions

If you encounter issues with file permissions in mounted volumes:

```bash
# Fix permissions for uploads directory
chmod -R 777 ./uploads
```

## Deployment

For production deployment, create a `.env.production` file with appropriate environment variables, then build and run the containers:

```bash
# Build for production
docker-compose -f docker-compose.yml -f docker-compose.prod.yml build

# Run in production mode
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```
