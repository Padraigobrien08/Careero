from setuptools import setup, find_packages

setup(
    name="careeroos",
    version="0.1",
    packages=find_packages(),
    py_modules=[
        "job_matcher",
        "llm_evaluator", 
        "resume_parser",
        "pdf_processor",
        "gemini_service",
        "csv_processor",
        "job_scraper",
        "resume_improver"
    ],
)
