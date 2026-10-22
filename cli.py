"""Command-line version (like your AI Placement Bot app_file.py).

Usage:
    python cli.py                      # uses every PDF in the Knowledge/ folder
    python cli.py paper1.pdf [more.pdf ...]
Commands inside the chat:  /summary   /tldr   /detailed   /quit
"""
import sys

from src.config import load_settings
from src.kb import build_or_load
from src.llm import LLM, LLMError
from src.pdf_loader import load_pdf
from src.qa import PaperQA
from src.summarizer import summarize_paper
from src.vector_store import build_vector_store, get_embeddings, split_documents


def main(paths: list[str]) -> None:
    settings = load_settings()
    llm = LLM(settings.hf_token, settings.model_name)

    if paths:   # explicit files: build an in-memory index
        papers, chunks = {}, []
        for path in paths:
            docs = load_pdf(path)
            papers[docs[0].metadata["source_name"]] = docs
            chunks += split_documents(docs, settings.qa_chunk_size, settings.qa_chunk_overlap)
        store = build_vector_store(chunks, get_embeddings())
    else:       # default: Knowledge/ folder with the saved vector_db/ index
        store, papers, _ = build_or_load(get_embeddings(), settings)
        if store is None:
            sys.exit("No PDFs found. Put PDFs in the Knowledge/ folder.")
    print(f"Loaded {len(papers)} paper(s). Ask away.")
    qa = PaperQA(store, llm, settings)

    formats = {"/summary": "Structured summary", "/tldr": "Quick TL;DR", "/detailed": "Detailed notes"}
    print("Ask a question, or type /summary, /tldr, /detailed, /quit")
    while True:
        text = input("\nYou: ").strip()
        if not text:
            continue
        if text == "/quit":
            break
        try:
            if text in formats:
                for name, docs in papers.items():
                    result = summarize_paper(docs, llm, settings, fmt=formats[text],
                                             progress=lambda f, m: print(f"  {m}...", end="\r"))
                    print(f"\n=== {result['title']} ===\n{result['summary']}")
            else:
                answer = qa.ask(text)
                print(f"\nAI: {answer.answer}")
                for s in answer.sources:
                    print(f"   [{s.number}] {s.paper}, page {s.page} (similarity {s.score})")
        except (LLMError, ValueError) as error:
            print(f"Error: {error}")


if __name__ == "__main__":
    main(sys.argv[1:])
