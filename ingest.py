"""Step 1: read the PDF(s) in /data, split into chunks, embed, store in ChromaDB.

Run once:   python ingest.py
Re-run it any time you change the PDF (it rebuilds the database from scratch).
"""
import re
import sys

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

import config


def clean_text(text: str) -> str:
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)   # join broken lines
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Sliding-window splitter that tries to end chunks at a sentence boundary."""
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"), window.rfind("। "))
            if cut > size * 0.5:
                end = start + cut + 1
        chunk = text[start:end].strip()
        if len(chunk) >= 50:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_pdf_chunks(pdf_path):
    reader = PdfReader(str(pdf_path))
    docs, metas, ids = [], [], []
    total_chars = 0
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        total_chars += len(text)
        for i, chunk in enumerate(split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)):
            docs.append(chunk)
            metas.append({"source": pdf_path.name, "page": page_no})
            ids.append(f"{pdf_path.stem}-p{page_no}-c{i}")
    return docs, metas, ids, len(reader.pages), total_chars


def main():
    pdfs = sorted(config.DATA_DIR.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDF found. Put your PDF inside: {config.DATA_DIR}")

    all_docs, all_metas, all_ids = [], [], []
    for pdf in pdfs:
        print(f"Reading {pdf.name} ...")
        docs, metas, ids, pages, chars = load_pdf_chunks(pdf)
        print(f"  {pages} pages, {chars} characters, {len(docs)} chunks")
        if chars < 200:
            print("  WARNING: almost no text found. This PDF may be scanned images; "
                  "it needs OCR first (see README).")
        all_docs += docs
        all_metas += metas
        all_ids += ids

    if not all_docs:
        sys.exit("No text could be extracted from the PDF(s).")

    print(f"Loading embedding model {config.EMBED_MODEL} (first run downloads ~90 MB) ...")
    model = SentenceTransformer(config.EMBED_MODEL)

    client = chromadb.PersistentClient(path=str(config.DB_DIR))
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    batch = 64
    for i in range(0, len(all_docs), batch):
        d = all_docs[i:i + batch]
        embeddings = model.encode(d, normalize_embeddings=True).tolist()
        collection.add(
            ids=all_ids[i:i + batch],
            documents=d,
            metadatas=all_metas[i:i + batch],
            embeddings=embeddings,
        )
        print(f"  stored {min(i + batch, len(all_docs))}/{len(all_docs)}")

    print(f"\nDone. Vector DB saved in: {config.DB_DIR}")
    print("Now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
