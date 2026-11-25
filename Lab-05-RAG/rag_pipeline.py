
import sys
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader  
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_ollama import OllamaEmbeddings, ChatOllama       
from langchain_chroma import Chroma                              

DB_DIR = "chroma_db"                  
COLLECTION = "university_docs"        
EMBED_MODEL = "qwen3-embedding:0.6b"  
LLM_MODEL = "llama3.2:3b"             
CHUNK_SIZE = 800                      
CHUNK_OVERLAP = 150                   
TOP_K = 4                             

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
llm = ChatOllama(model=LLM_MODEL, temperature=0)   
splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def get_store():
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,   
        persist_directory=DB_DIR,        
    )


def add_pdf(pdf_path, display_name=None):
    name = display_name or Path(pdf_path).name
    pages = PyPDFLoader(str(pdf_path)).load()       

    for page in pages:
        page.metadata = {"source": name, "page": int(page.metadata.get("page", 0))}

    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]   

    store = get_store()
    old = store.get(where={"source": name})           
    if old["ids"]:
        store.delete(ids=old["ids"])                 
    store.add_documents(chunks)
    return len(pages), len(chunks)


def retrieve(question, k=TOP_K):
    return get_store().similarity_search_with_score(question, k=k)


def build_prompt(question, results):
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
    q = " ".join(sys.argv[1:]) or "What is this document about?"
    reply, srcs = ask(q)
    print("\nANSWER:\n", reply)
    print("\nSOURCES:")
    for s in srcs:
        print(f"  - {s['source']} p.{s['page']} (distance {s['score']})")
