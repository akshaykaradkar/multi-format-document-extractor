"""
Configuration module for Document Automation System.

Handles environment variables, API keys, and system settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", "gpt-4o")

# Confidence Score Thresholds
CONFIDENCE_AUTO_APPROVE = float(os.getenv("CONFIDENCE_AUTO_APPROVE", "0.9"))
CONFIDENCE_REVIEW_THRESHOLD = float(os.getenv("CONFIDENCE_REVIEW_THRESHOLD", "0.7"))

# Supported file extensions mapped to parser types
FILE_TYPE_MAPPING = {
    ".pdf": "pdf",
    ".xlsx": "excel",
    ".xls": "excel",
    ".docx": "word",
    ".doc": "word",
    ".csv": "csv",
    ".jpg": "ocr",
    ".jpeg": "ocr",
    ".png": "ocr",
}

# Default currency
DEFAULT_CURRENCY = "USD"


def validate_config() -> bool:
    """Validate that required configuration is present."""
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set. OCR features will not work.")
        return False
    return True


def get_file_type(file_path: str | Path) -> str | None:
    """
    Determine the parser type based on file extension.

    Args:
        file_path: Path to the file

    Returns:
        Parser type string or None if unsupported
    """
    ext = Path(file_path).suffix.lower()
    return FILE_TYPE_MAPPING.get(ext)
