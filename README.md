# Research Paper Summarizer & Q&A

RAG project built from the GenAI notes: Hugging Face `InferenceClient`, LangChain,
`all-MiniLM-L6-v2` embeddings, FAISS and a multi-page Streamlit app.

## Folder structure
```
research-paper-summarizer-qa/
├── app.py                 # Home page (status + auto-load of Knowledge/)
├── pages/
│   ├── 1_Knowledge_Base.py   # upload / delete PDFs, build or refresh the index
│   ├── 2_Summarize.py        # TL;DR / structured / detailed summaries
│   ├── 3_Chat.py             # Q&A with [1] [2] citations (paper + page)
│   └── 4_Export.py           # download summaries (.md) and chat (.txt)
├── Knowledge/             # YOUR PDFs go here (a sample paper is included)
├── vector_db/             # FAISS index saved here automatically (created on first build)
├── src/                   # pdf_loader, vector_store, kb, qa, summarizer, llm, state, exporter, config
├── cli.py                 # terminal version
├── tests/test_pipeline.py # offline tests
├── requirements.txt
└── .env.example
```

## Run on Windows
```
cd research-paper-summarizer-qa
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env        (then open .env and paste your HF_TOKEN)
streamlit run app.py
```
If `requirements.txt` is "not found", you are in the wrong folder or the zip was only
partly extracted. Right-click the zip -> **Extract All**, then `cd` into the inner
`research-paper-summarizer-qa` folder (the one that contains `app.py`).

## How it works
1. PDFs in `Knowledge/` are read page by page (references are tagged and skipped in summaries).
2. Text is split into chunks, embedded with MiniLM and stored in FAISS (`vector_db/`).
3. A manifest detects added/removed/changed PDFs, so the index is rebuilt only when needed.
4. Summaries use map-reduce so long papers fit the model's context window.
5. Q&A retrieves the top-k chunks, answers only from them, cites sources, and says
   "I don't know based on the uploaded papers." when the answer is missing.
   Follow-ups are rewritten into standalone questions before searching.

## Tests
`python tests/test_pipeline.py`  (offline, no token needed)

## Limitations
Scanned PDFs need OCR; tables, equations and two-column layouts can extract poorly.
The FAISS index uses pickle: only load a `vector_db/` that you created yourself.

## Troubleshooting: "402 Payment Required / no remaining credits"

Hugging Face Inference Providers bill against your account credits. When they run out, switch the backend in `.env`
(see `.env.example`): **Groq** (free key) or **Ollama** (local, free) via `LLM_BASE_URL`, then `pip install -r requirements.txt`
and restart Streamlit. Billing/auth errors (401/402/403/404) now fail immediately with a clear message instead of retrying.
