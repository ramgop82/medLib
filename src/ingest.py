"""
Ingestion script: Reads data files, chunks them, and stores embeddings in ChromaDB.
Supports PDFs and text files.
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF, returning a list of page-wise chunks."""
    import fitz
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            pages.append({
                "text": text.strip(),
                "source": os.path.basename(pdf_path),
                "page": page_num + 1,
            })
    doc.close()
    return pages


def extract_text_from_txt(txt_path: str) -> list[dict]:
    """Extract text from a .txt file, splitting by separator or pages."""
    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by separator (used in boericke_complete.txt)
    sections = content.split("=" * 80)
    pages = []
    for i, section in enumerate(sections):
        text = section.strip()
        if text:
            pages.append({
                "text": text,
                "source": os.path.basename(txt_path),
                "page": i + 1,
            })
    return pages


def chunk_pages(pages: list[dict], chunk_size: int = 1000, overlap: int = 200) -> list[dict]:
    """Split page text into smaller overlapping chunks."""
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunks.append({
                "text": chunk_text,
                "source": page["source"],
                "page": page["page"],
                "chunk_start": start,
            })
            start += chunk_size - overlap
    return chunks


def build_vectorstore():
    """Build vector store from all data files (PDFs and text files)."""
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)

    all_chunks = []

    # Process text files
    txt_files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
    for txt_path in txt_files:
        if "README" in txt_path:
            continue
        print(f"Processing: {os.path.basename(txt_path)}...")
        pages = extract_text_from_txt(txt_path)
        chunks = chunk_pages(pages)
        all_chunks.extend(chunks)
        print(f"  → {len(chunks)} chunks")

    # Process PDFs (if any exist)
    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))
    for pdf_path in pdf_files:
        print(f"Processing: {os.path.basename(pdf_path)}...")
        try:
            pages = extract_text_from_pdf(pdf_path)
            if pages:
                chunks = chunk_pages(pages)
                all_chunks.extend(chunks)
                print(f"  → {len(chunks)} chunks")
            else:
                print(f"  → Skipped (no extractable text)")
        except Exception as e:
            print(f"  → Error: {e}")

    if not all_chunks:
        print("No data to ingest!")
        return

    # Build ChromaDB collection
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Delete existing and recreate
    try:
        client.delete_collection("homeopathy")
    except Exception:
        pass

    collection = client.create_collection(
        name="homeopathy",
        embedding_function=embedding_fn,
    )

    # Add chunks in batches
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i:i + batch_size]
        collection.add(
            ids=[f"chunk_{i + j}" for j in range(len(batch))],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
        )

    print(f"\n✅ Vector store built: {collection.count()} chunks")
    print(f"Stored at: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()
