import streamlit as st

st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")

from src.llm import LLMError
from src.state import bootstrap, need_knowledge_base, require_token

settings = bootstrap()
require_token(settings)
need_knowledge_base()
st.title("💬 Ask questions about your papers")

scope = st.selectbox("Search in", ["All papers"] + list(st.session_state.papers))
if st.button("Clear chat"):
    st.session_state.chat = []
    st.session_state.qa.clear_history()
    st.rerun()

for message in st.session_state.chat:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("Sources"):
                for s in message["sources"]:
                    st.markdown(f"**[{s.number}] {s.paper} - page {s.page}** (similarity {s.score})")
                    st.caption(s.text[:400] + ("..." if len(s.text) > 400 else ""))

question = st.chat_input("Ask about the papers, e.g. 'What dataset was used and what were the results?'")
if question:
    st.session_state.chat.append({"role": "user", "content": question})
    try:
        with st.spinner("Searching the papers and writing an answer..."):
            answer = st.session_state.qa.ask(question, None if scope == "All papers" else scope)
        st.session_state.chat.append(
            {"role": "assistant", "content": answer.answer, "sources": answer.sources})
    except (LLMError, ValueError) as error:
        st.session_state.chat.append({"role": "assistant", "content": f"⚠️ {error}", "sources": []})
    st.rerun()
