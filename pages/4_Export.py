import streamlit as st

st.set_page_config(page_title="Export", page_icon="⬇️", layout="wide")

from src.exporter import chat_to_text, summaries_to_markdown
from src.state import bootstrap

bootstrap()
st.title("⬇️ Export")
if st.session_state.summaries:
    st.download_button("Download summaries (.md)", summaries_to_markdown(st.session_state.summaries),
                       file_name="paper_summaries.md", mime="text/markdown")
else:
    st.caption("Generate at least one summary to export it.")
if st.session_state.chat:
    st.download_button("Download conversation (.txt)", chat_to_text(st.session_state.chat),
                       file_name="qa_conversation.txt", mime="text/plain")
else:
    st.caption("Ask at least one question to export the conversation.")
