import os

from dotenv import load_dotenv


load_dotenv()

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

READER_MODEL = os.getenv(
    "READER_MODEL",
    "distilbert-base-cased-distilled-squad",
)

CHUNKS_PATH = os.getenv(
    "CHUNKS_PATH",
    "data/processed/chunks.json",
)

FAISS_INDEX_PATH = os.getenv(
    "FAISS_INDEX_PATH",
    "data/processed/faiss_index.bin",
)

CHUNKS_PKL_PATH = os.getenv(
    "CHUNKS_PKL_PATH",
    "data/processed/chunks.pkl",
)

DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
