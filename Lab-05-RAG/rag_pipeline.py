"""rag_pipeline.py - the core RAG logic: add PDFs to the vector store, retrieve chunks, ask the LLM."""
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader   # reads a PDF page by page
from langchain_text_splitters import RecursiveCharacterTextSplitter  # cuts text into chunks
from langchain_ollama import OllamaEmbeddings, ChatOllama        # talk to models running in Ollama
from langchain_chroma import Chroma                              # the ChromaDB vector database

# ---- Settings (change them here and nowhere else) ----
DB_DIR = "chroma_db"                  # folder where the vector database is saved
COLLECTION = "university_docs"        # name of the collection inside ChromaDB
EMBED_MODEL = "qwen3-embedding:0.6b"  # turns text into numbers
LLM_MODEL = "llama3.2:3b"             # writes the final answer
CHUNK_SIZE = 800                      # max characters per chunk
CHUNK_OVERLAP = 150                   # characters shared between neighbouring chunks
TOP_K = 4                             # how many chunks to retrieve per question

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0)   # temperature 0 = least random, best for factual answers
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def get_store():
    """Open (or create) the vector database."""
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,   # Chroma uses this to embed every chunk and every question
        persist_directory=DB_DIR,        # saved to disk, so we only ingest once
    )


def add_pdf(pdf_path, display_name=None):
    """Read one PDF, split it into chunks, embed them and store them. Returns (pages, chunks)."""
    name = display_name or Path(pdf_path).name
    pages = PyPDFLoader(str(pdf_path)).load()       # one Document per page

    # Keep only the two metadata fields we need. The loader adds many more, and
    # ChromaDB can crash on empty (None) metadata values.
    for page in pages:
        page.metadata = {"source": name, "page": int(page.metadata.get("page", 0))}

    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]   # drop empty chunks (e.g. scanned pages)

    store = get_store()
    old = store.get(where={"source": name})           # if this file was added before...
    if old["ids"]:
        store.delete(ids=old["ids"])                  # ...remove the old copy so we get no duplicates
    store.add_documents(chunks)
    return len(pages), len(chunks)


def retrieve(question, k=TOP_K):
    """Find the k chunks most similar to the question. Lower score = closer match."""
    return get_store().similarity_search_with_score(question, k=k)


def build_prompt(question, results):
    """Put the retrieved chunks and the question into one prompt for the LLM."""
    context_parts = []
    for i, (doc, _score) in enumerate(results, start=1):
        label = f"[{i}] ({doc.metadata['source']}, page {doc.metadata['page'] + 1})"
        context_parts.append(f"{label}\n{doc.page_content}")
    context = "\n\n".join(context_parts)

    return (
        "You are a helpful university assistant. Answer the question using ONLY the context below.\n"
        "If the answer is not in the context, reply exactly: "
        "\"I could not find that in the documents.\"\n"
        "Mention the source number like [1] when you use a passage.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def ask(question):
    """Full RAG loop. Returns (answer_text, list_of_sources)."""
    results = retrieve(question)
    if not results:
        return "The knowledge base is empty. Add some PDFs first.", []

    answer = llm.invoke(build_prompt(question, results)).content
    sources = [
        {"source": d.metadata["source"], "page": d.metadata["page"] + 1,
         "score": round(float(s), 3), "text": d.page_content}
        for d, s in results
    ]
    return answer, sources


if __name__ == "__main__":
    # Quick terminal test:  python rag_pipeline.py "your question here"
    q = " ".join(sys.argv[1:]) or "What is this document about?"
    reply, srcs = ask(q)
    print("\nANSWER:\n", reply)
    print("\nSOURCES:")
    for s in srcs:
        print(f"  - {s['source']} p.{s['page']} (distance {s['score']})")
