"""Lab 06 Part C - agent with calculator, memory search and RAG search."""
import os
import sys

from langchain_core.tools import tool

from agent_memory import build_agent, chat_loop, init_db, search_memory
from agent_tools import calculator

RAG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Lab-05-RAG"))
sys.path.append(RAG_DIR)
import rag_pipeline as rag

rag.DB_DIR = os.path.join(RAG_DIR, "chroma_db")   # use the database built in Lab 5


@tool
def search_documents(question: str) -> str:
    """Answer questions about the university documents (assessment brief, deadlines, rules)."""
    answer, sources = rag.ask(question)
    cites = ", ".join(sorted({f"{s['source']} p.{s['page']}" for s in sources}))
    return f"{answer}\nSources: {cites}" if cites else answer


SYSTEM_PROMPT = (
    "You are a university assistant with three tools. Use search_documents for questions about "
    "the assessment brief, deadlines, marks or rules. Use search_memory for things the user told "
    "you earlier. Use calculator for maths. Never guess; if a tool finds nothing, say so. "
    "Keep answers short."
)

if __name__ == "__main__":
    init_db()
    chat_loop(build_agent([calculator, search_memory, search_documents], SYSTEM_PROMPT))
