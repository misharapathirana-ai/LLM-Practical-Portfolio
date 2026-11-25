"""ingest.py - build the vector database from every PDF in the data/ folder."""
import shutil
from pathlib import Path

import rag_pipeline as rag

DATA_DIR = Path("data")

if Path(rag.DB_DIR).exists():
    shutil.rmtree(rag.DB_DIR)          # start fresh so re-running never creates duplicates

pdfs = sorted(DATA_DIR.glob("*.pdf"))
if not pdfs:
    raise SystemExit("No PDFs found in the data/ folder. Add some and run again.")

total_chunks = 0
for pdf in pdfs:
    pages, chunks = rag.add_pdf(pdf)
    total_chunks += chunks
    print(f"{pdf.name}: {pages} pages -> {chunks} chunks")

print(f"\nDone. {len(pdfs)} PDF(s), {total_chunks} chunks stored in '{rag.DB_DIR}'.")
