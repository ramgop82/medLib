"""
Ingestion script: Reads PDFs from data/ folder, chunks them, and stores embeddings in ChromaDB.
Supports both text-based and scanned (OCR) PDFs.
Creates separate collections for Homeopathy and Ayurveda.
"""

import os
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions


DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")

# Map PDF filenames to their category
PDF_CATEGORIES = {
    "2015.92217.Organon-Of-Medicine_text.pdf": "homeopathy",
    "Charaka_Samhita_Text_with_English_Tanslation_-_P.V._Sharma.pdf": "ayurveda",
}


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from PDF. Falls back to OCR for scanned pages."""
    doc = fitz.open(pdf_path)
    pages = []

    # Check if first 5 pages have text (to detect scanned PDFs)
    has_text = any(doc[i].get_text().strip() for i in range(min(5, len(doc))))

    if has_text:
        # Text-based PDF
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            if text.strip():
                pages.append({
                    "text": text.strip(),
                    "source": os.path.basename(pdf_path),
                    "page": page_num + 1,
                })
    else:
        # Scanned PDF - use OCR
        print(f"  → Scanned PDF detected. Running OCR (this may take a while)...")
        try:
            from pdf2image import convert_from_path
            import pytesseract

            # Process in batches to manage memory
            total_pages = len(doc)
            doc.close()  # Close fitz, use pdf2image instead

            batch_size = 20
            for start in range(0, total_pages, batch_size):
                end = min(start + batch_size, total_pages)
                print(f"  → OCR pages {start+1}-{end}/{total_pages}...")
                images = convert_from_path(
                    pdf_path,
                    first_page=start + 1,
                    last_page=end,
                    dpi=200,
                )
                for i, img in enumerate(images):
                    text = pytesseract.image_to_string(img)
                    if text.strip():
                        pages.append({
                            "text": text.strip(),
                            "source": os.path.basename(pdf_path),
                            "page": start + i + 1,
                        })
            return pages
        except Exception as e:
            print(f"  → OCR failed: {e}")
            print(f"  → Skipping this PDF.")
            return []

    doc.close()
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
    """Main function: extract all PDFs, chunk, embed into separate collections."""
    import glob

    pdf_files = glob.glob(os.path.join(DATA_DIR, "*.pdf"))

    if not pdf_files:
        raise FileNotFoundError("No PDF files found in data/ folder.")

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Group PDFs by category
    categorized = {"homeopathy": [], "ayurveda": []}
    for pdf_path in pdf_files:
        filename = os.path.basename(pdf_path)
        category = PDF_CATEGORIES.get(filename, "homeopathy")
        categorized[category].append(pdf_path)

    # Process each category
    for category, paths in categorized.items():
        if not paths:
            continue

        print(f"\n{'='*50}")
        print(f"Processing: {category.upper()}")
        print(f"{'='*50}")

        all_chunks = []
        for pdf_path in paths:
            print(f"\nExtracting: {os.path.basename(pdf_path)}...")
            pages = extract_text_from_pdf(pdf_path)
            print(f"  → {len(pages)} pages extracted.")
            chunks = chunk_pages(pages)
            print(f"  → {len(chunks)} chunks created.")
            all_chunks.extend(chunks)

        if not all_chunks:
            print(f"  → No chunks for {category}. Skipping.")
            continue

        # Delete existing collection
        try:
            client.delete_collection(category)
        except Exception:
            pass

        collection = client.create_collection(
            name=category,
            embedding_function=embedding_fn,
        )

        # Add chunks in batches
        batch_size = 100
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i + batch_size]
            collection.add(
                ids=[f"{category}_chunk_{i + j}" for j in range(len(batch))],
                documents=[c["text"] for c in batch],
                metadatas=[{"source": c["source"], "page": c["page"]} for c in batch],
            )

        print(f"\n✅ {category} collection: {collection.count()} chunks stored.")

    print(f"\nDone! Vector store at: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()
