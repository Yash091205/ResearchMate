"""
ResearchMate – AI Research Assistant
Streamlit interface for text summarization and context-based Q&A.
"""

import streamlit as st
from model import (
    load_model,
    build_hf_pipeline,
    get_summarization_chain,
    get_qa_chain,
    summarize,
    answer_question,
    get_device,
)

st.set_page_config(page_title="ResearchMate", page_icon="🔬", layout="centered")


@st.cache_resource(show_spinner="Loading model... This may take a minute on first run.")
def init_model():
    """Load model and build LangChain chains (cached across reruns)."""
    model, tokenizer, device = load_model()
    llm = build_hf_pipeline(model, tokenizer, device)
    sum_chain = get_summarization_chain(llm)
    qa_chain = get_qa_chain(llm)
    return tokenizer, sum_chain, qa_chain, device


# ── Session State Initialization ──────────────────────────────────

if "summary_result" not in st.session_state:
    st.session_state.summary_result = None
if "qa_result" not in st.session_state:
    st.session_state.qa_result = None

# ── Header ────────────────────────────────────────────────────────

st.title("🔬 ResearchMate")
st.caption("AI Research Assistant — Summarize text and ask questions about it.")

device = get_device()
st.sidebar.markdown("**Model:** `google/flan-t5-base`")
st.sidebar.markdown(f"**Device:** `{device.upper()}`")
st.sidebar.markdown("---")
st.sidebar.markdown("Built with LangChain + HuggingFace + Streamlit")

# ── Load Model ────────────────────────────────────────────────────

try:
    tokenizer, sum_chain, qa_chain, device = init_model()
except Exception as e:
    st.error(
        "⚠️ Failed to load the model. Please check your internet connection "
        "for the initial model download, then restart the app."
    )
    st.exception(e)
    st.stop()

# ── Text Input ────────────────────────────────────────────────────

st.subheader("📄 Input Text")
user_text = st.text_area(
    "Paste or type your research text below:",
    height=200,
    placeholder="Enter the text you want to summarize or ask questions about...",
)

# ── Summarization ─────────────────────────────────────────────────

st.subheader("📝 Summarization")
if st.button("Generate Summary", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
        st.session_state.summary_result = None
    else:
        with st.spinner("Generating summary..."):
            st.session_state.summary_result = summarize(user_text, sum_chain, tokenizer)

# Display persisted summary
if st.session_state.summary_result is not None:
    result = st.session_state.summary_result
    if result["error"]:
        st.error(result["error"])
    else:
        if result["truncated"]:
            st.info("ℹ️ The input was truncated to fit the model's 512-token limit.")
        st.success("Summary generated!")
        st.write(result["summary"])

# ── Question Answering ────────────────────────────────────────────

st.subheader("❓ Ask a Question")
user_question = st.text_input(
    "Enter your question about the text above:",
    placeholder="e.g., What is the main finding?",
)

if st.button("Get Answer", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text first.")
        st.session_state.qa_result = None
    elif not user_question.strip():
        st.warning("Please enter a question.")
        st.session_state.qa_result = None
    else:
        with st.spinner("Generating answer..."):
            st.session_state.qa_result = answer_question(
                user_text, user_question, qa_chain, tokenizer
            )

# Display persisted answer
if st.session_state.qa_result is not None:
    result = st.session_state.qa_result
    if result["error"]:
        st.error(result["error"])
    else:
        if result["truncated"]:
            st.info("ℹ️ The context was truncated to fit the model's 512-token limit.")
        st.success("Answer generated!")
        st.write(result["answer"])
