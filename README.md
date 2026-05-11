# 🌿 Homeopathy RAG Assistant

A homeopathy remedy finder using CrewAI (3 agents), RAG, Ollama, and Streamlit.

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Install and run Ollama
```bash
ollama serve
ollama pull llama3
```

### 3. Ingest the PDF into vector store
```bash
python -m src.ingest
```

### 4. Run the app
```bash
streamlit run app.py
```

## Architecture

- **Agent 1 (Symptom Analyzer):** Normalizes user symptoms into homeopathic terminology
- **Agent 2 (Remedy Finder):** Searches the book via RAG for matching remedies
- **Agent 3 (Proof Extractor):** Extracts exact quotes with page numbers as proof

## Stack
- Python
- CrewAI (3 agents)
- Ollama (local LLM - llama3)
- ChromaDB (vector store)
- RAG (retrieval augmented generation)
- Streamlit (UI)
