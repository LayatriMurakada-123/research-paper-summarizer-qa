"""Central configuration. Values come from the environment / .env file."""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MODEL = "openai/gpt-oss-120b"


@dataclass
class Settings:
    hf_token: str | None = None
    model_name: str = DEFAULT_MODEL
    # Q&A chunking: small chunks give precise retrieval
    qa_chunk_size: int = 800
    qa_chunk_overlap: int = 150
    top_k: int = 5
    # Summary chunking: big chunks keep sections together for the LLM
    summary_chunk_size: int = 6000
    summary_chunk_overlap: int = 200
    # How many past chat turns are sent back to the model
    history_turns: int = 4


def load_settings() -> Settings:
    return Settings(
        hf_token=os.getenv("HF_TOKEN"),
        model_name=os.getenv("MODEL_NAME", DEFAULT_MODEL),
    )
