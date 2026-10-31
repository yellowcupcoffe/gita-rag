import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

DATA_DIR = BASE_DIR / "data"          # put your PDF here
DB_DIR = BASE_DIR / "chroma_db"       # vector database is stored here
COLLECTION_NAME = "bhagavad_gita"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # free, runs locally

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

CHUNK_SIZE = 1000      # characters per chunk
CHUNK_OVERLAP = 200    # overlap between neighbouring chunks
TOP_K = 5              # how many chunks to retrieve per question
