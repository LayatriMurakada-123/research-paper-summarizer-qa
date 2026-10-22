# Run with:  streamlit run app.py
import streamlit as st

st.set_page_config(page_title="Research Paper Summarizer & Q&A", page_icon="📄", layout="wide")

from src.kb import KNOWLEDGE_DIR, is_index_current, list_pdfs
from src.state import bootstrap, build_knowledge_base, require_token

settings = bootstrap()
require_token(settings)

# Auto-load the Knowledge/ folder once per session
if st.session_state.qa is None and list_pdfs() and not st.session_state.get("auto_tried"):
    st.session_state.auto_tried = True
    build_knowledge_base(settings)

st.title("📄 Research Paper Summarizer & Q&A")
st.caption("Put PDFs in the Knowledge/ folder (or upload them), summarize them, and chat with citations.")

c1, c2, c3 = st.columns(3)
c1.metric("PDFs in Knowledge/", len(list_pdfs()))
c2.metric("Papers loaded", len(st.session_state.papers))
c3.metric("Saved index up to date", "Yes" if is_index_current() else "No")
st.info(st.session_state.status)

st.markdown(
    """
### How to use
1. **📚 Knowledge Base**: add PDFs and click *Build / refresh*.
2. **📝 Summarize**: TL;DR, structured or detailed notes, for a student or an expert.
3. **💬 Chat**: ask questions; answers cite `[1]`, `[2]` with paper name and page.
4. **⬇️ Export**: download summaries (.md) and the conversation (.txt).
"""
)
