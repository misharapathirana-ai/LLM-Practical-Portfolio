"""app.py - Streamlit chat window with PDF upload. Run with: streamlit run app.py"""
from pathlib import Path

import streamlit as st

import rag_pipeline as rag

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="University Knowledge Assistant")
st.title("University Knowledge Assistant")
st.caption(f"Local RAG: {rag.LLM_MODEL} + {rag.EMBED_MODEL} + ChromaDB")

# ---- Sidebar: upload extra PDFs ----
with st.sidebar:
    st.header("Add documents")
    uploaded = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if st.button("Add to knowledge base") and uploaded:
        for file in uploaded:
            save_path = DATA_DIR / file.name
            save_path.write_bytes(file.getbuffer())          # keep a copy in data/
            with st.spinner(f"Reading {file.name}..."):
                pages, chunks = rag.add_pdf(save_path)
            st.success(f"{file.name}: {pages} pages, {chunks} chunks added")

# ---- Chat history (Streamlit reruns the script on every action, so we store it in session_state) ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        for s in msg.get("sources", []):
            with st.expander(f"Source: {s['source']}, page {s['page']} (distance {s['score']})"):
                st.write(s["text"])

question = st.chat_input("Ask a question about your documents")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and thinking..."):
            answer, sources = rag.ask(question)
        st.markdown(answer)
        for s in sources:
            with st.expander(f"Source: {s['source']}, page {s['page']} (distance {s['score']})"):
                st.write(s["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
