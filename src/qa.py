"""Question answering over the papers (RAG: Search first -> Answer later).

Flow:  question -> (rewrite follow-up) -> retrieve top-k chunks from FAISS
       -> prompt with context + chat history -> LLM -> answer + sources
"""
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS

from .config import Settings
from .llm import LLM

NOT_FOUND = "I don't know based on the uploaded papers."

QA_SYSTEM_PROMPT = (
    "You are a research assistant that answers questions about academic papers. "
    "Answer ONLY from the numbered context passages. Do not use outside knowledge. "
    "Cite the passages you used like [1] or [2]. "
    f'If the context does not contain the answer, reply exactly: "{NOT_FOUND}" '
    "Be clear and concise, and keep numbers exactly as written in the context."
)

REWRITE_PROMPT = """Given the chat history and a follow-up question, rewrite the follow-up
as a single standalone question that can be understood without the history.
Return ONLY the rewritten question.

CHAT HISTORY:
{history}

FOLLOW-UP QUESTION: {question}

STANDALONE QUESTION:"""


@dataclass
class Source:
    number: int
    paper: str
    page: int          # human-readable (1-based)
    score: float       # cosine similarity, higher = more related
    text: str


@dataclass
class Answer:
    question: str
    answer: str
    sources: list[Source] = field(default_factory=list)


def _cosine_from_faiss(squared_l2: float) -> float:
    """Embeddings are normalized, so cosine = 1 - (squared L2 distance) / 2."""
    return round(1 - squared_l2 / 2, 3)


class PaperQA:
    def __init__(self, store: FAISS, llm: LLM, settings: Settings):
        self.store = store
        self.llm = llm
        self.settings = settings
        self.history: list[dict] = []  # [{'role': 'user'|'assistant', 'content': str}]

    # -- retrieval ------------------------------------------------------
    def retrieve(self, question: str, paper: str | None = None) -> list[Source]:
        kwargs = {"k": self.settings.top_k}
        if paper:
            kwargs["filter"] = {"source_name": paper}
            kwargs["fetch_k"] = max(50, self.settings.top_k * 10)
        results = self.store.similarity_search_with_score(question, **kwargs)
        sources = []
        for n, (doc, distance) in enumerate(results, start=1):
            sources.append(Source(
                number=n,
                paper=doc.metadata.get("source_name", "Unknown"),
                page=int(doc.metadata.get("page", 0)) + 1,
                score=_cosine_from_faiss(float(distance)),
                text=doc.page_content,
            ))
        return sources

    # -- helpers --------------------------------------------------------
    def _recent_history(self) -> list[dict]:
        return self.history[-2 * self.settings.history_turns:]

    def _standalone_question(self, question: str) -> str:
        """Turn 'what about its limitations?' into a self-contained search query."""
        recent = self._recent_history()
        if not recent:
            return question
        history_text = "\n".join(f"{m['role'].upper()}: {m['content'][:400]}" for m in recent)
        try:
            rewritten = self.llm.ask(
                "You rewrite questions. Output only the rewritten question.",
                REWRITE_PROMPT.format(history=history_text, question=question),
                max_tokens=80, temperature=0.0,
            )
            return rewritten.strip().strip('"') or question
        except Exception:
            return question  # fall back to the raw question

    # -- main entry point -----------------------------------------------
    def ask(self, question: str, paper: str | None = None) -> Answer:
        question = (question or "").strip()
        if not question:
            raise ValueError("Please type a question.")

        search_query = self._standalone_question(question)
        sources = self.retrieve(search_query, paper)

        if not sources:
            answer = Answer(question, NOT_FOUND, [])
        else:
            context = "\n\n".join(
                f"[{s.number}] ({s.paper}, page {s.page})\n{s.text}" for s in sources
            )
            messages = [{"role": "system", "content": QA_SYSTEM_PROMPT}]
            messages += self._recent_history()
            messages.append({
                "role": "user",
                "content": f"CONTEXT PASSAGES:\n{context}\n\nQUESTION: {question}\n\nANSWER:",
            })
            text = self.llm.chat(messages, max_tokens=500, temperature=0.2)
            answer = Answer(question, text, sources)

        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer.answer})
        return answer

    def clear_history(self) -> None:
        self.history = []
