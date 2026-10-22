import streamlit as st

st.set_page_config(page_title="Knowledge Base", page_icon="📚", layout="wide")

from src.kb import KNOWLEDGE_DIR, VECTOR_DIR, is_index_current, list_pdfs
from src.state import bootstrap, build_knowledge_base, require_token

settings = bootstrap()
require_token(settings)
st.title("📚 Knowledge Base")
st.caption(f"Papers are stored in the `{KNOWLEDGE_DIR.name}/` folder; the index is saved in `{VECTOR_DIR.name}/`.")

uploads = st.file_uploader("Add research paper PDFs", type=["pdf"], accept_multiple_files=True)
if uploads and st.button("Save to Knowledge folder"):
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    for f in uploads:
        (KNOWLEDGE_DIR / f.name).write_bytes(f.getbuffer())
    st.success(f"Saved {len(uploads)} file(s). Now click Build / refresh.")

pdfs = list_pdfs()
st.subheader(f"Papers in the knowledge base ({len(pdfs)})")
if not pdfs:
    st.write("No PDFs yet. Upload some above, or copy them into the Knowledge/ folder.")
for pdf in pdfs:
    a, b = st.columns([5, 1])
    a.write(f"📄 {pdf.name}  ·  {pdf.stat().st_size / 1024:.0f} KB")
    if b.button("Delete", key=f"del_{pdf.name}"):
        pdf.unlink()
        st.rerun()

with st.expander("Settings"):
    model_name = st.text_input("LLM model", value=settings.model_name)
    top_k = st.slider("Passages retrieved per question (k)", 2, 10, settings.top_k)

st.write("Index status:", "✅ up to date" if is_index_current() else "⚠️ needs (re)building")
c1, c2 = st.columns(2)
if c1.button("Build / refresh", type="primary"):
    build_knowledge_base(settings, model_name=model_name, top_k=top_k)
if c2.button("Force full rebuild"):
    build_knowledge_base(settings, force=True, model_name=model_name, top_k=top_k)
st.info(st.session_state.status)
for name in st.session_state.skipped:
    st.warning(f"No readable text in {name} (scanned image PDF?). It was skipped.")
