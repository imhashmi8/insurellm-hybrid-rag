# Import necessary libraries

import os
from pathlib import Path
from pydantic import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(override=True)

# Path to the directory containing the data files
ROOT = Path(__file__).parent
KB_PATH = ROOT.parent / "knowledge_base"
DB_NAME = str(ROOT / "preprocessed_db")
CHUNK_CACHE = ROOT / "chunk_cache"
COLLECTION = "docs"

# Model configuration

GEN_MODEL = os.getenv("GEN_MODEL","groq/openai/gpt-oss-120b")
EMBED_MODEL = "text-embedding-3-large"
CHUNK_MODEL = os.getenv("CHUNK_MODEL", "openai/gpt-4.1-nano")
JUDGE_MODEL  = os.getenv("JUDGE_MODEL", "openai/gpt-4.1-nano")


# Retriever configuration

AVERAGE_CHUNK_SIZE = 800
RETRIEVE_K = 20
FINAL_K = 10
RERANKER = os.getenv("RERANKER", "llm")
EMBED_BATCH = 100
CHUNK_CACHE.mkdir(exist_ok=True)

# Result class

class Result(BaseModel):
    page_content: str
    metadata: dict