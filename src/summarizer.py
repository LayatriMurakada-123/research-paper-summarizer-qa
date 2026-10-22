"""Research-paper summarizer using the map-reduce pattern.

A paper is usually longer than one prompt can comfortably hold (context window!),
so we:
  MAP    -> summarize every big chunk of the paper separately
  REDUCE -> merge those partial summaries into one final summary
"""
from langchain_core.documents import Document

from .config import Settings
from .llm import LLM
from .pdf_loader import extract_abstract, guess_title, strip_references
from .vector_store import split_documents

SYSTEM_PROMPT = (
    "You are an expert research assistant who summarizes academic papers accurately. "
    "Use ONLY the text you are given. Never invent results, numbers, datasets or "
    "citations. If something is not stated in the text, write 'Not stated in the paper'."
)

MAP_PROMPT = """Below is one part of a research paper.
Write 4-6 short bullet points covering the key facts in this part:
problem or motivation, method, data, results (keep exact numbers), limitations.
Skip anything this part does not contain.

PAPER PART:
{text}

BULLET POINTS:"""

AUDIENCES = {
    "Student (simple language)": "Explain in simple, student-friendly language. Define technical terms briefly.",
    "Expert (technical)": "Use precise technical language. Assume the reader knows the field.",
}

FORMATS = {
    "Quick TL;DR": (
        "Write a TL;DR of 3-4 sentences covering the problem, the approach and the main result.",
        350,
    ),
    "Structured summary": (
        "Write a structured summary in Markdown with EXACTLY these headings:\n"
        "### Problem\n### Method\n### Key Results\n### Limitations\n### Conclusion\n"
        "Keep each section to 2-4 sentences (bullets are fine for results).",
        800,
    ),
    "Detailed notes": (
        "Write detailed study notes in Markdown with these headings:\n"
        "### Background & Motivation\n### Main Contributions\n### Method\n"
        "### Experiments & Datasets\n### Key Results\n### Limitations\n"
        "### Key Terms (a short glossary of 4-8 terms)\n### Takeaways for a Student",
        1400,
    ),
}

REDUCE_PROMPT = """Below are notes taken from consecutive parts of ONE research paper.
{instruction}
{audience}

NOTES:
{notes}

FINAL SUMMARY:"""


def _batch_text(parts: list[str], max_chars: int) -> list[str]:
    """Group short texts together so the reduce step fits in one prompt."""
    batches, current, size = [], [], 0
    for part in parts:
        if current and size + len(part) > max_chars:
            batches.append("\n\n".join(current))
            current, size = [], 0
        current.append(part)
        size += len(part)
    if current:
        batches.append("\n\n".join(current))
    return batches


def summarize_paper(
    docs: list[Document],
    llm: LLM,
    settings: Settings,
    fmt: str = "Structured summary",
    audience: str = "Student (simple language)",
    progress=None,
) -> dict:
    """Return {'title', 'abstract', 'summary', 'parts'} for one paper.

    `progress` is an optional callback: progress(fraction, message).
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unknown format: {fmt}")
    notify = progress or (lambda fraction, message: None)

    body = strip_references(docs)
    title = guess_title(docs)
    abstract = extract_abstract(docs)

    chunks = split_documents(body, settings.summary_chunk_size, settings.summary_chunk_overlap)
    if not chunks:
        raise ValueError("This PDF has no readable text (it may be a scanned image).")

    # ---- MAP: one mini-summary per chunk --------------------------------
    partials = []
    for i, chunk in enumerate(chunks, start=1):
        notify(0.05 + 0.65 * i / len(chunks), f"Reading part {i} of {len(chunks)}")
        partials.append(llm.ask(SYSTEM_PROMPT, MAP_PROMPT.format(text=chunk.page_content),
                                max_tokens=350, temperature=0.1))

    # ---- hierarchical REDUCE if the notes are still too long --------------
    max_chars = 9000
    while len("\n\n".join(partials)) > max_chars and len(partials) > 1:
        notify(0.75, "Condensing notes")
        batches = _batch_text(partials, max_chars)
        if len(batches) == len(partials):  # cannot shrink further
            break
        partials = [
            llm.ask(SYSTEM_PROMPT,
                    "Merge these notes into 6-8 bullet points, keeping exact numbers:\n\n" + b,
                    max_tokens=450, temperature=0.1)
            for b in batches
        ]

    # ---- REDUCE: final summary --------------------------------------------
    notify(0.85, "Writing the final summary")
    instruction, max_tokens = FORMATS[fmt]
    summary = llm.ask(
        SYSTEM_PROMPT,
        REDUCE_PROMPT.format(instruction=instruction,
                             audience=AUDIENCES.get(audience, ""),
                             notes="\n\n".join(partials)),
        max_tokens=max_tokens, temperature=0.2,
    )
    notify(1.0, "Done")
    return {"title": title, "abstract": abstract, "summary": summary, "parts": len(chunks)}
