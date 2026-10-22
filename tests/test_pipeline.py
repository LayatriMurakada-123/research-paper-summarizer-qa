"""Offline tests: no internet, no Hugging Face token needed.

Run with:  python -m pytest -q      (or)   python tests/test_pipeline.py
"""
import hashlib
import os
import re
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.config import Settings
from src.exporter import chat_to_text, summaries_to_markdown
from src.pdf_loader import clean_text, extract_abstract, guess_title, load_pdf, strip_references
from src.qa import NOT_FOUND, PaperQA
from src.summarizer import FORMATS, summarize_paper
from src.vector_store import build_vector_store, split_documents


class FakeEmbeddings(Embeddings):
    """Bag-of-words hashing -> normalized vectors. Similar words => similar vectors."""
    dim = 256

    def _vec(self, text):
        v = np.zeros(self.dim, dtype="float32")
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
        n = np.linalg.norm(v)
        return (v / n if n else v).tolist()

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """Records every prompt so tests can inspect them."""
    def __init__(self):
        self.calls = []

    def ask(self, system_prompt, user_prompt, **kw):
        self.calls.append(("ask", user_prompt))
        if "STANDALONE QUESTION" in user_prompt:
            return "What dataset did the transformer model use for training?"
        return "- fake bullet"

    def chat(self, messages, **kw):
        self.calls.append(("chat", messages))
        return "It used the WMT dataset [1]."


def make_pdf(path):
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    c = canvas.Canvas(path, pagesize=A4)
    pages = [
        ["Attention Based Neural Translation", "Abstract", "We propose a transformer model for translation.",
         "It reaches 28.4 BLEU on the WMT dataset.", "1 Introduction", "Machine translation is hard."],
        ["2 Method", "The transformer uses self attention instead of recurrence.", "3 Results",
         "Training used the WMT 2014 English-German dataset."],
        ["References", "[1] Vaswani et al. Attention is all you need."],
    ]
    for lines in pages:
        y = 780
        for line in lines:
            c.drawString(60, y, line)
            y -= 22
        c.showPage()
    c.save()


def _setup():
    tmp = tempfile.mkdtemp()
    pdf = os.path.join(tmp, "paper.pdf")
    make_pdf(pdf)
    docs = load_pdf(pdf, "paper.pdf")
    return docs


def test_clean_text_fixes_hyphens_and_newlines():
    assert clean_text("trans-\nlation is\nhard") == "translation is hard"


def test_load_pdf_and_metadata():
    docs = _setup()
    assert len(docs) == 3
    assert docs[0].metadata["source_name"] == "paper.pdf" and docs[0].metadata["page"] == 0
    assert docs[0].metadata["is_references"] is False
    assert docs[2].metadata["is_references"] is True


def test_title_abstract_and_references():
    docs = _setup()
    assert "Attention" in guess_title(docs)
    assert "28.4 BLEU" in extract_abstract(docs)
    body = strip_references(docs)
    assert not any("Vaswani" in d.page_content for d in body)
    assert len(body) == 2


def test_chunking_and_retrieval_with_page_numbers():
    docs = _setup()
    chunks = split_documents(docs, 120, 20)
    assert len(chunks) > 3
    assert all("raw_head" not in c.metadata for c in chunks)
    store = build_vector_store(chunks, FakeEmbeddings())
    qa = PaperQA(store, FakeLLM(), Settings(top_k=2))
    sources = qa.retrieve("which dataset was used for training")
    assert sources and sources[0].page == 2          # 1-based page of the Results section
    assert 0 < sources[0].score <= 1
    # filtering by paper name works
    assert all(s.paper == "paper.pdf" for s in qa.retrieve("dataset", paper="paper.pdf"))
    assert qa.retrieve("dataset", paper="other.pdf") == []


def test_qa_memory_and_followup_rewrite():
    docs = _setup()
    store = build_vector_store(split_documents(docs, 200, 30), FakeEmbeddings())
    llm = FakeLLM()
    qa = PaperQA(store, llm, Settings(top_k=3))
    first = qa.ask("What BLEU score did it reach?")
    assert first.answer and first.sources
    second = qa.ask("And which dataset was that on?")
    assert len(qa.history) == 4
    # a rewrite call happened only for the follow-up (history existed)
    rewrites = [c for c in llm.calls if c[0] == "ask" and "STANDALONE QUESTION" in c[1]]
    assert len(rewrites) == 1
    # the final chat prompt contains numbered context and the earlier turn
    last_messages = [c for c in llm.calls if c[0] == "chat"][-1][1]
    assert "[1]" in last_messages[-1]["content"]
    assert any(m["role"] == "assistant" for m in last_messages)


def test_empty_question_rejected():
    store = build_vector_store([Document(page_content="hello world", metadata={"page": 0})], FakeEmbeddings())
    try:
        PaperQA(store, FakeLLM(), Settings()).ask("   ")
        assert False, "should raise"
    except ValueError:
        pass


def test_summarizer_map_reduce_all_formats():
    docs = _setup()
    long_docs = docs * 6                       # force several chunks
    for fmt in FORMATS:
        llm = FakeLLM()
        steps = []
        result = summarize_paper(long_docs, llm, Settings(summary_chunk_size=300, summary_chunk_overlap=30),
                                 fmt=fmt, progress=lambda f, m: steps.append(f))
        assert result["summary"] and result["parts"] >= 2
        assert steps[-1] == 1.0
        assert len(llm.calls) >= result["parts"] + 1   # map calls + 1 reduce


def test_knowledge_folder_index_saved_and_reused():
    from pathlib import Path
    from src.kb import build_or_load, is_index_current
    tmp = Path(tempfile.mkdtemp())
    kdir, vdir = tmp / "Knowledge", tmp / "vector_db"
    kdir.mkdir()
    make_pdf(str(kdir / "paper.pdf"))
    assert not is_index_current(kdir, vdir)
    store, papers, skipped = build_or_load(FakeEmbeddings(), Settings(), knowledge=kdir, vector=vdir)
    assert store is not None and "paper.pdf" in papers and skipped == []
    assert (vdir / "index.faiss").exists() and is_index_current(kdir, vdir)
    # second call loads from disk (no rebuild)
    store2, _, _ = build_or_load(FakeEmbeddings(), Settings(), knowledge=kdir, vector=vdir)
    assert store2.index.ntotal == store.index.ntotal
    # adding a new PDF makes the saved index stale
    make_pdf(str(kdir / "second.pdf"))
    assert not is_index_current(kdir, vdir)
    # empty folder -> no store
    empty = tmp / "empty"
    assert build_or_load(FakeEmbeddings(), Settings(), knowledge=empty, vector=vdir)[0] is None


def test_exporters():
    md = summaries_to_markdown({"a.pdf": {"title": "T", "abstract": "ab", "summary": "S"}})
    assert "## T" in md and "S" in md
    txt = chat_to_text([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}])
    assert "YOU:" in txt and "AI:" in txt


def test_llm_fails_fast_on_billing_error_and_retries_others():
    from src.llm import LLM, LLMError

    class Err(Exception):
        def __init__(self, code): self.status_code = code

    class FakeClient:
        def __init__(self, errors):
            self.errors, self.n = list(errors), 0
            self.chat = self; self.completions = self
        def create(self, **kw):
            self.n += 1
            if self.errors:
                raise self.errors.pop(0)
            class R: choices = [type("C", (), {"message": type("M", (), {"content": " ok "})()})()]
            return R

    import src.llm as mod
    mod.time.sleep = lambda s: None
    llm = LLM("tok", "m")
    llm.client = FakeClient([Err(402)])
    try:
        llm.chat([{"role": "user", "content": "hi"}]); assert False
    except LLMError as e:
        assert "credits" in str(e) and llm.client.n == 1      # no retries on 402
    llm.client = FakeClient([Err(503), Err(503)])
    assert llm.chat([{"role": "user", "content": "hi"}]) == "ok" and llm.client.n == 3
    llm.client = FakeClient([Err(503)] * 3)
    try:
        llm.chat([{"role": "user", "content": "hi"}]); assert False
    except LLMError as e:
        assert "unavailable" in str(e)


def test_llm_uses_openai_client_when_base_url_given():
    from src.llm import LLM
    llm = LLM("key", "llama", base_url="http://localhost:11434/v1")
    assert type(llm.client).__module__.startswith("openai")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for t in tests:
        t()
        print("PASS", t.__name__)
    print(f"\nAll {len(tests)} tests passed")
