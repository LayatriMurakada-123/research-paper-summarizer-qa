"""Knowledge base manager: Knowledge/ folder -> chunks -> FAISS -> vector_db/.

The index is saved to disk, so papers are embedded only once. A small
manifest (file name + size + modified time) detects when PDFs were added,
removed or changed, and the index is rebuilt automatically.
"""
import json
from pathlib import Path

from .config import Settings
from .pdf_loader import load_pdf
from .vector_store import build_vector_store, load_store, save_store, split_documents

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = ROOT / "Knowledge"
VECTOR_DIR = ROOT / "vector_db"
MANIFEST = VECTOR_DIR / "manifest.json"


def list_pdfs(folder: Path = KNOWLEDGE_DIR) -> list[Path]:
    folder.mkdir(exist_ok=True)
    return sorted(folder.glob("*.pdf"), key=lambda p: p.name.lower())


def fingerprint(folder: Path = KNOWLEDGE_DIR) -> dict:
    return {p.name: [p.stat().st_size, int(p.stat().st_mtime)] for p in list_pdfs(folder)}


def is_index_current(knowledge: Path = KNOWLEDGE_DIR, vector: Path = VECTOR_DIR) -> bool:
    manifest = vector / "manifest.json"
    if not (vector / "index.faiss").exists() or not manifest.exists():
        return False
    try:
        return json.loads(manifest.read_text()) == fingerprint(knowledge)
    except Exception:
        return False


def load_papers(folder: Path = KNOWLEDGE_DIR) -> tuple[dict, list[str]]:
    """Return ({name: [page Documents]}, [names of unreadable PDFs])."""
    papers, skipped = {}, []
    for pdf in list_pdfs(folder):
        try:
            docs = load_pdf(str(pdf), pdf.name)
        except Exception:
            docs = []
        if docs:
            papers[pdf.name] = docs
        else:
            skipped.append(pdf.name)
    return papers, skipped


def build_or_load(embeddings, settings: Settings, force: bool = False,
                  knowledge: Path = KNOWLEDGE_DIR, vector: Path = VECTOR_DIR):
    """Return (store, papers, skipped). Rebuilds only when needed."""
    papers, skipped = load_papers(knowledge)
    if not papers:
        return None, {}, skipped
    if not force and is_index_current(knowledge, vector):
        return load_store(embeddings, str(vector)), papers, skipped
    chunks = []
    for docs in papers.values():
        chunks += split_documents(docs, settings.qa_chunk_size, settings.qa_chunk_overlap)
    store = build_vector_store(chunks, embeddings)
    vector.mkdir(exist_ok=True)
    save_store(store, str(vector))
    (vector / "manifest.json").write_text(json.dumps(fingerprint(knowledge)))
    return store, papers, skipped
