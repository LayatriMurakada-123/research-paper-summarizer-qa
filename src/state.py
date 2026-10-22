"""Shared Streamlit session logic used by every page (kept out of the UI files)."""
import os

import streamlit as st

from .config import load_settings


def bootstrap():
    """Call at the top of every page. Returns Settings; stops the page if no token."""
    try:  # Streamlit Cloud secrets -> environment
        for key in ("HF_TOKEN", "MODEL_NAME", "LLM_BASE_URL"):
            if key in st.secrets and not os.getenv(key):
                os.environ[key] = st.secrets[key]
    except Exception:
        pass
    settings = load_settings()
    for key, value in {"papers": {}, "qa": None, "summaries": {}, "chat": [],
                       "skipped": [], "status": "Knowledge base not built yet."}.items():
        st.session_state.setdefault(key, value)
    return settings


def require_token(settings):
    if not settings.hf_token:
        st.error("HF_TOKEN is missing. Create a `.env` file with `HF_TOKEN=hf_...` "
                 "(see `.env.example`) and restart the app.")
        st.stop()


@st.cache_resource(show_spinner="Loading the embedding model (first run only)...")
def cached_embeddings():
    from .vector_store import get_embeddings
    return get_embeddings()


@st.cache_resource(show_spinner=False)
def cached_llm(token: str, model_name: str):
    from .llm import LLM
    return LLM(token=token, model_name=model_name)


def build_knowledge_base(settings, force: bool = False, model_name: str | None = None,
                         top_k: int | None = None):
    """Build/load the index from Knowledge/ and store everything in session_state."""
    from .kb import build_or_load
    from .qa import PaperQA
    if model_name:
        settings.model_name = model_name
    if top_k:
        settings.top_k = top_k
    try:
        with st.spinner("Reading PDFs and preparing the vector database..."):
            store, papers, skipped = build_or_load(cached_embeddings(), settings, force=force)
    except Exception as error:
        st.session_state.status = f"Could not build the knowledge base: {error}"
        return
    st.session_state.skipped = skipped
    if store is None:
        st.session_state.update(papers={}, qa=None,
                                status="No readable PDFs found in the Knowledge/ folder.")
        return
    llm = cached_llm(settings.hf_token, settings.model_name)
    st.session_state.update(
        papers=papers, qa=PaperQA(store, llm, settings), summaries={}, chat=[],
        status=f"Ready: {len(papers)} paper(s), {sum(len(d) for d in papers.values())} pages.")


def need_knowledge_base():
    if st.session_state.qa is None:
        st.warning("The knowledge base is empty. Go to the **Knowledge Base** page, "
                   "add PDFs and click **Build / refresh**.")
        st.stop()
