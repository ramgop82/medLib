"""
Remedy finder: Uses RAG + single LLM call to find remedies with proof.
No CrewAI needed — simple and fast.
"""

import os
from dotenv import load_dotenv
import litellm
from src.rag import retrieve

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")


def _call_llm(prompt: str) -> str:
    """Call the LLM based on .env config."""
    if LLM_PROVIDER == "anthropic":
        response = litellm.completion(
            model=f"anthropic/{ANTHROPIC_MODEL}",
            api_key=ANTHROPIC_API_KEY,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
    elif LLM_PROVIDER == "gemini":
        response = litellm.completion(
            model=f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}",
            api_key=os.getenv("GEMINI_API_KEY", ""),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
    elif LLM_PROVIDER == "groq":
        response = litellm.completion(
            model=f"groq/{os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}",
            api_key=os.getenv("GROQ_API_KEY", ""),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
    else:
        response = litellm.completion(
            model=f"ollama/{OLLAMA_MODEL}",
            api_base=OLLAMA_BASE_URL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1500,
        )
    return response.choices[0].message.content


def find_remedy(symptoms: str, language: str = "English", system: str = "homeopathy") -> str:
    """Find remedy using RAG + single LLM call."""

    # 1. Retrieve relevant passages from books
    passages = retrieve(symptoms, n_results=8, system=system)

    if not passages:
        return "No relevant information found in the books."

    # 2. Build context from retrieved passages
    context = ""
    for i, p in enumerate(passages, 1):
        context += f"\n--- Passage {i} (Source: {p['source']}, Page {p['page']}) ---\n"
        context += p["text"]
        context += "\n"

    # 3. Build prompt
    if language == "Telugu (తెలుగు)":
        lang_instruction = (
            "Provide your ENTIRE response in Telugu (తెలుగు లిపి). "
            "Medicine names can stay in English, but all explanations must be in Telugu."
        )
    else:
        lang_instruction = "Respond in English."

    prompt = (
        "You are a homeopathic reference assistant. Based ONLY on the patient's symptoms "
        "and the book passages below, provide:\n\n"
        "1. **Recommended Remedy** — the best matching homeopathic medicine\n"
        "2. **Why** — brief explanation of why it matches the symptoms\n"
        "3. **Dosage** — potency and dosage as mentioned in the book passages\n"
        "4. **Proof from Books** — exact quotes from the passages with source and page numbers\n\n"
        f"Patient Symptoms: {symptoms}\n\n"
        f"Book Passages:\n{context}\n\n"
        f"{lang_instruction}\n\n"
        "STRICT RULES:\n"
        "- If the patient symptoms are unclear, incomplete, or contain nonsense words, "
        "respond ONLY with: 'I could not understand your symptoms clearly. Please describe "
        "your symptoms in plain language (e.g., headache, fever, stomach pain).'\n"
        "- ONLY recommend a remedy if you can clearly match the symptoms to a remedy "
        "described in the passages above.\n"
        "- Do NOT guess, assume, or invent connections that are not explicitly in the passages.\n"
        "- If no passage clearly matches the symptoms, say: 'No matching remedy found in the "
        "books for these symptoms. Please try describing your symptoms differently.'\n"
        "- Always cite the exact source and page number.\n"
        "- For dosage, use what the book states — if not mentioned, say 'Consult a practitioner for dosage.'"
    )

    # 4. Call LLM
    result = _call_llm(prompt)

    # 5. Post-validation: check if response contains actual citations
    # If LLM hallucinated (no page/source reference), reject it
    has_citation = any(word in result.lower() for word in ["page", "source:", "passage", "boericke", "organon"])
    has_refusal = any(phrase in result.lower() for phrase in ["could not understand", "no matching remedy", "please describe", "please provide"])

    if not has_citation and not has_refusal:
        return "I could not find a clear remedy match for your symptoms in the books. Please describe your symptoms more clearly (e.g., 'throbbing headache worse in sunlight with nausea')."

    return result
