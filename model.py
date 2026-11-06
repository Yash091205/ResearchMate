"""
ResearchMate - Model loading, summarization, and question answering.
Uses google/flan-t5-base with LangChain LCEL integration.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


MODEL_NAME = "google/flan-t5-base"
MAX_INPUT_TOKENS = 512  # flan-t5-base context window

# Approximate token overhead for each prompt template (measured via tokenizer)
_SUMMARIZE_OVERHEAD = 18  # "Summarize the following text in a concise paragraph:\n\n...\n\nSummary:"
_QA_OVERHEAD = 50         # Full QA template excluding context and question


def get_device() -> str:
    """Detect best available device."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_model():
    """Load the flan-t5-base model and tokenizer.

    Returns:
        tuple: (model, tokenizer, device)

    Raises:
        RuntimeError: If model loading fails.
    """
    try:
        device = get_device()
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME, clean_up_tokenization_spaces=True
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
        return model, tokenizer, device
    except Exception as e:
        raise RuntimeError(
            f"Failed to load model '{MODEL_NAME}'. "
            f"Ensure you have an internet connection for the first download. "
            f"Details: {e}"
        ) from e


def build_hf_pipeline(model, tokenizer, device, max_new_tokens=256):
    """Create a HuggingFace text2text-generation pipeline wrapped for LangChain."""
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if device == "cuda" else -1,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


def count_tokens(text: str, tokenizer) -> int:
    """Return the number of tokens in the given text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def truncate_text(text: str, tokenizer, max_tokens: int = MAX_INPUT_TOKENS) -> str:
    """Truncate text to fit within the specified token limit.

    Returns:
        tuple: (truncated_text, was_truncated)
    """
    tokens = tokenizer.encode(text, truncation=True, max_length=max_tokens)
    truncated = tokenizer.decode(tokens, skip_special_tokens=True)
    was_truncated = count_tokens(text, tokenizer) > max_tokens
    return truncated, was_truncated


# ── LangChain LCEL Chains ─────────────────────────────────────────

SUMMARIZE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in a concise paragraph:\n\n{text}\n\nSummary:",
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer the question using only the information in the context below. "
        "Write a clear, grammatically correct, complete English sentence. "
        "Begin with a capital letter and use correct capitalization for "
        "technical terms, proper nouns, and abbreviations. "
        "Use appropriate punctuation. "
        "Do not give incomplete phrases or unnecessarily short answers. "
        "Do not add information that is not present in the context. "
        "If the answer is not available in the context, respond exactly with: "
        "'The provided text does not contain enough information to answer this question.'\n\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
)


def get_summarization_chain(llm: HuggingFacePipeline):
    """Build a LangChain LCEL summarization chain."""
    return SUMMARIZE_PROMPT | llm | StrOutputParser()


def get_qa_chain(llm: HuggingFacePipeline):
    """Build a LangChain LCEL QA chain."""
    return QA_PROMPT | llm | StrOutputParser()


def summarize(text: str, chain, tokenizer) -> dict:
    """Generate a summary of the input text.

    Returns:
        dict with keys: 'summary' (str), 'truncated' (bool), 'error' (str or None)
    """
    if not text or not text.strip():
        return {"summary": "", "truncated": False, "error": "Please provide some text to summarize."}

    try:
        max_text_tokens = MAX_INPUT_TOKENS - _SUMMARIZE_OVERHEAD
        truncated_text, was_truncated = truncate_text(text.strip(), tokenizer, max_text_tokens)

        if not truncated_text.strip():
            return {"summary": "", "truncated": False, "error": "The input text is too short to summarize."}

        result = chain.invoke({"text": truncated_text})
        return {"summary": result.strip(), "truncated": was_truncated, "error": None}

    except Exception as e:
        return {"summary": "", "truncated": False, "error": f"Summarization failed: {e}"}


def answer_question(context: str, question: str, chain, tokenizer) -> dict:
    """Answer a question based on the provided context.

    Returns:
        dict with keys: 'answer' (str), 'truncated' (bool), 'error' (str or None)
    """
    if not context or not context.strip():
        return {"answer": "", "truncated": False, "error": "Please provide some text as context first."}
    if not question or not question.strip():
        return {"answer": "", "truncated": False, "error": "Please enter a question."}

    try:
        question_tokens = count_tokens(question.strip(), tokenizer)
        max_context_tokens = MAX_INPUT_TOKENS - _QA_OVERHEAD - question_tokens

        if max_context_tokens < 20:
            return {
                "answer": "",
                "truncated": False,
                "error": "The question is too long. Please shorten it and try again.",
            }

        truncated_context, was_truncated = truncate_text(
            context.strip(), tokenizer, max_context_tokens
        )
        result = chain.invoke({
            "context": truncated_context,
            "question": question.strip(),
        })
        return {
            "answer": format_answer(result),
            "truncated": was_truncated,
            "error": None,
        }

    except Exception as e:
        return {"answer": "", "truncated": False, "error": f"Question answering failed: {e}"}


def format_answer(answer: str) -> str:
    """Apply basic English capitalization and punctuation."""
    answer = answer.strip()

    if not answer:
        return answer

    fallback = (
        "The provided text does not contain enough information "
        "to answer this question."
    )

    if answer.lower().rstrip(".!") == fallback.lower().rstrip(".!"):
        return fallback

    # Capitalize the first character without changing the remaining text.
    answer = answer[0].upper() + answer[1:]

    # Add punctuation if missing.
    if answer[-1] not in ".!?":
        answer += "."

    return answer