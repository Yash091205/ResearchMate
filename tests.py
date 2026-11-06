"""
Tests for ResearchMate model.py
Covers: summarization, QA, unsupported questions, empty inputs, token handling.
"""

import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from model import (
    load_model,
    build_hf_pipeline,
    get_summarization_chain,
    get_qa_chain,
    summarize,
    answer_question,
    count_tokens,
    truncate_text,
)

PASS = 0
FAIL = 0


def report(name, passed, detail=""):
    global PASS, FAIL
    status = "PASS" if passed else "FAIL"
    if passed:
        PASS += 1
    else:
        FAIL += 1
    suffix = f" — {detail}" if detail else ""
    print(f"  [{status}] {name}{suffix}")


def run_tests():
    global PASS, FAIL

    print("Loading model...")
    model, tokenizer, device = load_model()
    llm = build_hf_pipeline(model, tokenizer, device)
    sum_chain = get_summarization_chain(llm)
    qa_chain = get_qa_chain(llm)
    print(f"Model loaded on: {device}\n")

    sample_text = (
        "Artificial intelligence (AI) is intelligence demonstrated by machines, "
        "as opposed to the natural intelligence displayed by animals and humans. "
        "AI research has been defined as the field of study of intelligent agents, "
        "which refers to any system that perceives its environment and takes actions "
        "that maximize its chance of achieving its goals. The term artificial intelligence "
        "had previously been used to describe machines that mimic and display human "
        "cognitive skills that are associated with the human mind, such as learning "
        "and problem-solving."
    )

    # ── Summarization Tests ───────────────────────────────────────

    print("=== Summarization Tests ===")

    # Test 1: Basic summarization
    result = summarize(sample_text, sum_chain, tokenizer)
    report(
        "Basic summarization",
        result["error"] is None and len(result["summary"]) > 0,
        f"summary='{result['summary'][:80]}...'",
    )

    # Test 2: Empty input
    result = summarize("", sum_chain, tokenizer)
    report(
        "Empty input returns error",
        result["error"] is not None and result["summary"] == "",
        f"error='{result['error']}'",
    )

    # Test 3: Whitespace-only input
    result = summarize("   \n\t  ", sum_chain, tokenizer)
    report(
        "Whitespace-only input returns error",
        result["error"] is not None,
        f"error='{result['error']}'",
    )

    # Test 4: Long input truncation
    long_text = sample_text * 20  # Repeat to exceed 512 tokens
    result = summarize(long_text, sum_chain, tokenizer)
    report(
        "Long input is truncated and summarized",
        result["error"] is None and result["truncated"] is True and len(result["summary"]) > 0,
        f"truncated={result['truncated']}",
    )

    # ── QA Tests ──────────────────────────────────────────────────

    print("\n=== Question Answering Tests ===")

    # Test 5: Basic QA
    result = answer_question(sample_text, "What is artificial intelligence?", qa_chain, tokenizer)
    report(
        "Basic QA",
        result["error"] is None and len(result["answer"]) > 0,
        f"answer='{result['answer'][:80]}'",
    )

    # Test 6: Unsupported question (answer not in context)
    result = answer_question(sample_text, "What is the capital of France?", qa_chain, tokenizer)
    report(
        "Unsupported question handling",
        result["error"] is None and len(result["answer"]) > 0,
        f"answer='{result['answer'][:80]}'",
    )

    # Test 7: Empty context
    result = answer_question("", "What is AI?", qa_chain, tokenizer)
    report(
        "Empty context returns error",
        result["error"] is not None and result["answer"] == "",
        f"error='{result['error']}'",
    )

    # Test 8: Empty question
    result = answer_question(sample_text, "", qa_chain, tokenizer)
    report(
        "Empty question returns error",
        result["error"] is not None and result["answer"] == "",
        f"error='{result['error']}'",
    )

    # Test 9: Both empty
    result = answer_question("", "", qa_chain, tokenizer)
    report(
        "Both empty returns error",
        result["error"] is not None,
        f"error='{result['error']}'",
    )

    # ── Token Utility Tests ───────────────────────────────────────

    print("\n=== Token Utility Tests ===")

    # Test 10: count_tokens
    token_count = count_tokens("Hello world", tokenizer)
    report(
        "count_tokens returns positive int",
        isinstance(token_count, int) and token_count > 0,
        f"tokens={token_count}",
    )

    # Test 11: truncate_text within limit
    short = "This is a short sentence."
    truncated, was_truncated = truncate_text(short, tokenizer, max_tokens=100)
    report(
        "Short text not truncated",
        was_truncated is False and truncated == short,
        f"was_truncated={was_truncated}",
    )

    # Test 12: truncate_text exceeding limit
    long_text = sample_text * 20
    truncated, was_truncated = truncate_text(long_text, tokenizer, max_tokens=50)
    report(
        "Long text is truncated",
        was_truncated is True and count_tokens(truncated, tokenizer) <= 50,
        f"was_truncated={was_truncated}, tokens={count_tokens(truncated, tokenizer)}",
    )

    # ── Summary ───────────────────────────────────────────────────

    print(f"\n{'='*50}")
    print(f"Results: {PASS} passed, {FAIL} failed out of {PASS + FAIL} tests")
    if FAIL == 0:
        print("All tests passed! ✓")
    else:
        print(f"{FAIL} test(s) FAILED.")
    return FAIL


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)

