import streamlit as st

st.set_page_config(page_title="Summarize", page_icon="📝", layout="wide")

from src.llm import LLMError
from src.state import bootstrap, cached_llm, need_knowledge_base, require_token
from src.summarizer import AUDIENCES, FORMATS, summarize_paper

settings = bootstrap()
require_token(settings)
need_knowledge_base()
st.title("📝 Summarize a paper")

c1, c2, c3 = st.columns([2, 1.5, 1.5])
chosen = c1.selectbox("Paper", list(st.session_state.papers))
fmt = c2.selectbox("Summary type", list(FORMATS))
audience = c3.selectbox("Audience", list(AUDIENCES))

if st.button("Generate summary", type="primary"):
    bar = st.progress(0.0, text="Starting...")
    try:
        llm = cached_llm(settings.hf_token, st.session_state.qa.llm.model_name)
        result = summarize_paper(st.session_state.papers[chosen], llm, settings, fmt=fmt,
                                 audience=audience,
                                 progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        result["fmt"] = fmt
        st.session_state.summaries[chosen] = result
    except (LLMError, ValueError) as error:
        st.error(str(error))
    finally:
        bar.empty()

result = st.session_state.summaries.get(chosen)
if result:
    st.subheader(result["title"])
    st.caption(f"{result['fmt']} · built from {result['parts']} text parts")
    if result["abstract"]:
        with st.expander("Abstract (extracted from the paper)"):
            st.write(result["abstract"])
    st.markdown(result["summary"])
