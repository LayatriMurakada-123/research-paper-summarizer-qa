"""Step 1 of the pipeline: PDF -> clean text pages (PyPDFLoader)."""
import re
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

# A line that is just the heading "References" / "Bibliography"
_REF_HEADING = re.compile(r"^\s*(\d+\.?\s*)?(references|bibliography)\s*$",
                          re.IGNORECASE | re.MULTILINE)


def clean_text(text: str) -> str:
    """Fix common PDF extraction problems."""
    text = text.replace("\x00", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join hyphenated words
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)       # single newline -> space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: str, display_name: str | None = None) -> list[Document]:
    """Load one PDF and return one cleaned Document per non-empty page.

    Line structure is needed to spot headings, so we look for the "References"
    heading in the RAW page text first, and only then clean the text.
    Everything from that heading onward is flagged `is_references=True`
    (summaries skip it; Q&A can still search it).
    """
    name = display_name or Path(path).name
    pages = PyPDFLoader(str(path)).load()
    docs: list[Document] = []
    in_refs = False

    for page in pages:
        raw = page.page_content
        number = page.metadata.get("page", 0)   # zero-based; +1 when displaying
        base = {"source_name": name, "page": number}
        if number == 0:
            base["raw_head"] = raw[:600]         # used to guess the title

        pieces = [(raw, in_refs)]
        if not in_refs:
            match = _REF_HEADING.search(raw)
            if match and number > 0:
                pieces = [(raw[:match.start()], False), (raw[match.start():], True)]
                in_refs = True

        for text, is_refs in pieces:
            text = clean_text(text)
            if text:
                docs.append(Document(page_content=text,
                                     metadata={**base, "is_references": is_refs}))
    return docs


def strip_references(docs: list[Document]) -> list[Document]:
    """Drop the bibliography so summaries focus on the paper itself."""
    return [d for d in docs if not d.metadata.get("is_references")]


def guess_title(docs: list[Document], fallback: str = "Untitled paper") -> str:
    """Heuristic: the first meaningful line(s) of page 1 are usually the title."""
    if not docs:
        return fallback
    head = docs[0].metadata.get("raw_head") or docs[0].page_content[:300]
    lines = [ln.strip() for ln in head.splitlines() if len(ln.strip()) > 12]
    if not lines:
        return docs[0].page_content[:100].strip() or fallback
    title = lines[0]
    # titles often wrap onto a second line
    if len(title) < 50 and len(lines) > 1 and "@" not in lines[1] and not re.search(r"\d", lines[1]):
        title += " " + lines[1]
    return title[:150]


def extract_abstract(docs: list[Document], max_chars: int = 1800) -> str:
    """Return the abstract if the paper has one, else an empty string."""
    if not docs:
        return ""
    text = " ".join(d.page_content for d in docs[:2])
    match = re.search(r"\babstract\b[\s:.\-—]*(.+?)(?=\b(?:1\.?\s+)?introduction\b|\bkeywords?\b)",
                      text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip()[:max_chars] if match else ""
