"""
Follow-up question logic: determines if symptoms are detailed enough
and generates clarifying questions if not.
"""

import os
from dotenv import load_dotenv
import litellm

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")

SYMPTOM_WORDS = [
    "pain", "ache", "head", "stomach", "fever", "cold", "cough", "nausea",
    "vomit", "diarrhea", "burn", "itch", "rash", "sore", "throat", "back",
    "chest", "dizzy", "tired", "weak", "swelling", "bleeding", "cramp",
    "sleep", "anxiety", "stress", "hot", "chills", "sneeze", "nose",
    "eye", "ear", "skin", "joint", "muscle", "breath", "heart", "leg",
    "arm", "neck", "shoulder", "knee", "foot", "hand", "abdomen", "belly",
    "worse", "better", "morning", "night", "eating", "throbbing", "sharp",
    "dull", "chronic", "acute", "inflammation", "infection", "allergy",
    "spin", "vertigo", "weight", "appetite", "thirst", "sweat",
]


def get_followup_questions(symptoms: str) -> tuple[bool, str]:
    """
    Check if symptoms are detailed enough.
    Returns (is_sufficient, followup_message).
    """
    words = symptoms.lower().split()

    # Reject gibberish
    has_valid_symptom = any(
        any(sw in word for sw in SYMPTOM_WORDS)
        for word in words
    )

    if not has_valid_symptom:
        return True, "INVALID"

    # If enough detail, proceed
    if len(words) >= 20:
        return True, ""

    # Ask follow-up questions
    followup = _generate_followup(symptoms)
    return False, followup


def _call_followup_llm(prompt: str) -> str:
    """Call LLM for follow-up questions (supports all providers)."""
    if LLM_PROVIDER == "anthropic":
        response = litellm.completion(
            model=f"anthropic/{ANTHROPIC_MODEL}",
            api_key=ANTHROPIC_API_KEY,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
    elif LLM_PROVIDER == "gemini":
        response = litellm.completion(
            model=f"gemini/{os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}",
            api_key=os.getenv("GEMINI_API_KEY", ""),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
    elif LLM_PROVIDER == "groq":
        response = litellm.completion(
            model=f"groq/{os.getenv('GROQ_MODEL', 'llama-3.3-70b-versatile')}",
            api_key=os.getenv("GROQ_API_KEY", ""),
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
    else:
        response = litellm.completion(
            model=f"ollama/{OLLAMA_MODEL}",
            api_base=OLLAMA_BASE_URL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
        )
    return response.choices[0].message.content


def _generate_followup(symptoms: str) -> str:
    """Generate follow-up questions using LLM."""
    try:
        prompt = (
            f"A patient described these symptoms: \"{symptoms}\"\n\n"
            f"As a homeopathic practitioner, ask 2-3 short follow-up questions "
            f"to better understand their condition. Ask about:\n"
            f"- What makes it better or worse (modalities)\n"
            f"- Time of day it's worse\n"
            f"- Any other accompanying symptoms\n\n"
            f"Keep it brief and friendly. Start with 'To find the best remedy, "
            f"I need a bit more detail:'"
        )
        return _call_followup_llm(prompt)
    except Exception:
        return (
            "To find the best remedy, I need a bit more detail:\n\n"
            "1. What makes it better or worse? (heat, cold, movement, rest?)\n"
            "2. What time of day is it worst?\n"
            "3. Any other symptoms along with it?"
        )
