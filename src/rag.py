"""
RAG module: Retrieves relevant chunks from ChromaDB based on user query and selected system.
"""

import os
import chromadb
from chromadb.utils import embedding_functions


VECTORSTORE_DIR = os.path.join(os.path.dirname(__file__), "..", "vectorstore")


def get_collection(system: str = "homeopathy"):
    """Get the ChromaDB collection for the specified medical system."""
    client = chromadb.PersistentClient(path=VECTORSTORE_DIR)
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = client.get_collection(
        name=system,
        embedding_function=embedding_fn,
    )
    return collection


def retrieve(query: str, n_results: int = 5, system: str = "homeopathy") -> list[dict]:
    """Retrieve top-n relevant chunks for a given symptom query."""
    collection = get_collection(system)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i],
        })

    return retrieved
