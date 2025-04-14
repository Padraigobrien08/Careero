import logging.config
import os
from pathlib import Path

# Create logs directory if it doesn't exist
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": "uvicorn.logging.DefaultFormatter",
            "fmt": "%(levelprefix)s %(asctime)s %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
        "access": {
            "()": "uvicorn.logging.AccessFormatter",
            "fmt": '%(levelprefix)s %(asctime)s %(client_addr)s - "%(request_line)s" %(status_code)s',
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "default": {
            "formatter": "default",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stderr",
        },
        "access": {
            "formatter": "access",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filename": "logs/app.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
        "security": {
            "class": "logging.handlers.RotatingFileHandler",
            "formatter": "default",
            "filename": "logs/security.log",
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
        },
    },
    "loggers": {
        "uvicorn": {"handlers": ["default", "file"], "level": "INFO", "propagate": False},
        "uvicorn.error": {"level": "INFO", "handlers": ["default", "file"], "propagate": False},
        "uvicorn.access": {"handlers": ["access", "file"], "level": "INFO", "propagate": False},
        "security": {"handlers": ["security"], "level": "WARNING", "propagate": False},
    },
    "root": {"level": "INFO", "handlers": ["default", "file"]},
}

def setup_logging():
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler for all logs
    file_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "app.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)

    # File handler for security logs
    security_handler = logging.handlers.RotatingFileHandler(
        os.path.join(log_dir, "security.log"),
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    security_handler.setLevel(logging.WARNING)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    security_formatter = logging.Formatter(
        '%(asctime)s - SECURITY - %(levelname)s - %(message)s'
    )

    # Set formatters
    file_handler.setFormatter(file_formatter)
    security_handler.setFormatter(security_formatter)
    console_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(security_handler)
    logger.addHandler(console_handler)

    return logger

# Security logging functions
def log_security_event(event_type: str, details: str, user: str = None):
    security_logger = logging.getLogger("security")
    log_message = f"Event: {event_type}"
    if user:
        log_message += f" | User: {user}"
    log_message += f" | Details: {details}"
    security_logger.warning(log_message)

def log_authentication_attempt(username: str, success: bool, ip: str):
    log_security_event(
        "AUTH_ATTEMPT",
        f"Success: {success} | IP: {ip}",
        username
    )

def log_file_upload(filename: str, user: str, success: bool):
    log_security_event(
        "FILE_UPLOAD",
        f"File: {filename} | Success: {success}",
        user
    )

def log_api_request(endpoint: str, method: str, user: str = None, status: int = None):
    log_security_event(
        "API_REQUEST",
        f"Endpoint: {endpoint} | Method: {method} | Status: {status}",
        user
    ) 