"""Steps 2-4 of the pipeline: chunk -> embed -> FAISS vector database."""
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import EMBEDDING_MODEL


def get_embeddings():
    """Load the sentence-transformer once (same model as your Placement Bot)."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def split_documents(docs: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata.pop("raw_head", None)   # helper field, not needed in the index
    return chunks


def build_vector_store(chunks: list[Document], embeddings) -> FAISS:
    if not chunks:
        raise ValueError("No text chunks to index. Is the PDF a scanned image?")
    return FAISS.from_documents(chunks, embeddings)


def save_store(store: FAISS, folder: str = "vector_db") -> None:
    store.save_local(folder)


def load_store(embeddings, folder: str = "vector_db") -> FAISS:
    # Only load an index that YOU created: pickle files can run code.
    return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
