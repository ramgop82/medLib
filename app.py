"""
Streamlit UI for Homeopathy RAG Application with conversational follow-up.
"""

import os
import streamlit as st
from src.remedy_finder import find_remedy
from src.followup import get_followup_questions
from src.cache import get_cached_result, save_to_cache

# Auto-build vector store if it doesn't exist
vectorstore_path = os.path.join(os.path.dirname(__file__), "vectorstore")
if not os.path.exists(vectorstore_path) or not os.listdir(vectorstore_path):
    with st.spinner("First-time setup: building knowledge base..."):
        from src.ingest import build_vectorstore
        build_vectorstore()


st.set_page_config(page_title="Homeopathy Assistant", page_icon="🌿", layout="wide")

st.title("🌿 Homeopathy Assistant")
st.subheader("Find remedies from Boericke's Materia Medica & Organon of Medicine")

st.warning("⚠️ **Disclaimer:** This tool is for educational and reference purposes only. It is NOT a substitute for professional medical advice. Always consult a qualified homeopathic practitioner before taking any medicine.")
st.markdown("Describe your symptoms and the AI will ask follow-up questions before recommending a remedy.")

st.divider()

# Language selection
language = st.selectbox("Select output language:", ["English", "Telugu (తెలుగు)"])

# Medical system selection
system = st.selectbox("Select medical system:", ["Homeopathy", "Ayurveda"])
system_key = system.lower()

st.divider()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "all_symptoms" not in st.session_state:
    st.session_state.all_symptoms = ""
if "ready_for_remedy" not in st.session_state:
    st.session_state.ready_for_remedy = False

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Describe your symptoms...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Validate input first — reject gibberish immediately
    symptom_words = [
        "pain", "ache", "head", "stomach", "fever", "cold", "cough", "nausea",
        "vomit", "diarrhea", "burn", "itch", "rash", "sore", "throat", "back",
        "chest", "dizzy", "tired", "weak", "swelling", "bleeding", "cramp",
        "sleep", "anxiety", "stress", "hot", "chills", "sneeze", "nose",
        "eye", "ear", "skin", "joint", "muscle", "breath", "heart", "leg",
        "arm", "neck", "shoulder", "knee", "foot", "hand", "abdomen", "belly",
        "worse", "better", "morning", "night", "eating", "throbbing", "sharp",
        "dull", "chronic", "acute", "inflammation", "infection", "allergy",
        "yes", "no", "left", "right", "both", "sometimes", "always", "never",
    ]
    input_lower = user_input.lower()
    is_valid = any(word in input_lower for word in symptom_words)

    if not is_valid and not st.session_state.ready_for_remedy:
        with st.chat_message("assistant"):
            msg = "⚠️ Please provide valid medical symptoms (e.g., headache, fever, cough, stomach pain) so I can help you find a remedy."
            st.markdown(msg)
            st.session_state.messages.append({"role": "assistant", "content": msg})
    else:
        # Accumulate symptoms
        st.session_state.all_symptoms += " " + user_input

        # Check if we have enough info
        with st.chat_message("assistant"):
            if st.session_state.ready_for_remedy:
                # User answered follow-up — go straight to finding remedy
                symptoms = st.session_state.all_symptoms.strip()
                with st.spinner("Searching books for remedy..."):
                    cached = get_cached_result(symptoms, language)
                    if cached:
                        result = cached
                    else:
                        result = find_remedy(symptoms, language=language, system=system_key)
                        save_to_cache(symptoms, language, result)

                    st.markdown(result)
                    st.session_state.messages.append({"role": "assistant", "content": result})

                st.session_state.ready_for_remedy = False
                st.session_state.all_symptoms = ""

            else:
                # Check if symptoms are detailed enough
                with st.spinner("Analyzing..."):
                    sufficient, followup = get_followup_questions(st.session_state.all_symptoms.strip())

                if sufficient:
                    # Enough detail, find remedy
                    with st.spinner("Searching books for remedy..."):
                        symptoms = st.session_state.all_symptoms.strip()

                        # Check for invalid input
                        if followup == "INVALID":
                            result = "Please provide valid medical symptoms so I can help you find a remedy."
                        else:
                            cached = get_cached_result(symptoms, language)
                            if cached:
                                result = cached
                            else:
                                result = find_remedy(symptoms, language=language, system=system_key)
                                save_to_cache(symptoms, language, result)

                        st.markdown(result)
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        st.session_state.all_symptoms = ""
                else:
                    # Ask follow-up questions
                    st.markdown(followup)
                    st.session_state.messages.append({"role": "assistant", "content": followup})
                    st.session_state.ready_for_remedy = True

# Reset button
if st.session_state.messages:
    if st.button("🔄 Start New Consultation"):
        st.session_state.messages = []
        st.session_state.all_symptoms = ""
        st.session_state.ready_for_remedy = False
        st.rerun()

st.divider()
st.caption("Source: Boericke's Materia Medica & Organon of Medicine by Samuel Hahnemann | For educational purposes only. Not medical advice.")
