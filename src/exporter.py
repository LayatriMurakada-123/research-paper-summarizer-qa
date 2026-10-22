"""Export summaries and chat history (like save_conversation() in your chatbot)."""
from datetime import datetime


def summaries_to_markdown(summaries: dict[str, dict]) -> str:
    lines = [f"# Paper Summaries\n\n_Generated on {datetime.now():%Y-%m-%d %H:%M}_\n"]
    for filename, result in summaries.items():
        lines.append(f"\n---\n\n## {result['title']}\n\n_File: {filename}_\n")
        if result.get("abstract"):
            lines.append(f"**Abstract (from the paper):** {result['abstract']}\n")
        lines.append(result["summary"] + "\n")
    return "\n".join(lines)


def chat_to_text(chat: list[dict]) -> str:
    """chat items: {'role', 'content', optional 'sources': [Source]}"""
    out = ["=" * 50, "Research Paper Q&A - Conversation History", "=" * 50]
    for message in chat:
        label = "YOU" if message["role"] == "user" else "AI"
        out.append(f"{label}:\n{message['content']}")
        for s in message.get("sources", []):
            out.append(f"   [{s.number}] {s.paper}, page {s.page}")
    out += ["=" * 50, "End of conversation"]
    return "\n".join(out)
