import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = os.getenv("INDEX_DIR", str(BASE_DIR / "index"))
DOCS_DIR = os.getenv("DOCS_DIR", str(BASE_DIR / "docs"))

# Model Configuration
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))

# API Keys (ensure they are loaded)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
