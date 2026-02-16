from __future__ import annotations

import math
import re
from typing import Any

from openai import OpenAI


# ---------------------------------------------------------------------------
# Grade Level (Flesch-Kincaid) — pure Python, zero extra dependencies
# ---------------------------------------------------------------------------

_VOWELS = set("aeiouyAEIOUY")


def _count_syllables(word: str) -> int:
    """Heuristic English syllable count."""
    word = word.lower().strip()
    if not word:
        return 0
    # Remove trailing silent-e
    if len(word) > 2 and word.endswith("e") and word[-2] not in "aeiouy":
        word = word[:-1]
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in _VOWELS
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    return max(count, 1)


_SENTENCE_BOUNDARY = re.compile(r"[.!?]+")
_WORD_TOKEN = re.compile(r"[A-Za-z''\u2019]+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_BOUNDARY.split(text)
    return [s.strip() for s in parts if s.strip()]


def _tokenize_words(text: str) -> list[str]:
    return _WORD_TOKEN.findall(text)


def compute_grade_level(*, text: str) -> dict[str, Any]:
    """Flesch-Kincaid grade level and reading ease from raw text."""
    text = str(text).strip()
    if not text:
        return {
            "mechanism": "grade_level",
            "available": False,
            "reason": "empty_text",
        }

    words = _tokenize_words(text)
    sentences = _split_sentences(text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)

    if word_count < 3:
        return {
            "mechanism": "grade_level",
            "available": False,
            "reason": "too_few_words",
        }

    syllable_count = sum(_count_syllables(w) for w in words)
    avg_words_per_sentence = word_count / sentence_count
    avg_syllables_per_word = syllable_count / word_count

    fk_grade = (
        0.39 * avg_words_per_sentence
        + 11.8 * avg_syllables_per_word
        - 15.59
    )
    fk_reading_ease = (
        206.835
        - 1.015 * avg_words_per_sentence
        - 84.6 * avg_syllables_per_word
    )

    return {
        "mechanism": "grade_level",
        "available": True,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "syllable_count": syllable_count,
        "avg_words_per_sentence": round(avg_words_per_sentence, 2),
        "avg_syllables_per_word": round(avg_syllables_per_word, 2),
        "flesch_kincaid_grade": round(fk_grade, 2),
        "flesch_reading_ease": round(fk_reading_ease, 2),
    }


def format_grade_level_for_prompt(feedback: dict[str, Any]) -> str:
    if not isinstance(feedback, dict):
        return ""
    if not feedback.get("available"):
        reason = str(feedback.get("reason", "unavailable")).strip()
        return f"Grade-level readability unavailable ({reason})."

    grade = feedback.get("flesch_kincaid_grade")
    ease = feedback.get("flesch_reading_ease")
    wps = feedback.get("avg_words_per_sentence")

    grade_str = f"{float(grade):.1f}" if isinstance(grade, (int, float)) else "n/a"
    ease_str = f"{float(ease):.1f}" if isinstance(ease, (int, float)) else "n/a"
    wps_str = f"{float(wps):.1f}" if isinstance(wps, (int, float)) else "n/a"

    return (
        f"Readability: Flesch-Kincaid grade={grade_str}, "
        f"reading_ease={ease_str}, avg_words_per_sentence={wps_str}. "
        "Lower grade = easier reading. Reading ease >60 is plain English."
    )


# ---------------------------------------------------------------------------
# Embedding Cosine Similarity — uses OpenRouter embedding models via OpenAI client
# ---------------------------------------------------------------------------

DEFAULT_EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_embedding_similarity(
    *,
    client: OpenAI,
    source_text: str,
    translation_text: str,
    model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Cosine similarity between source and translation via an OpenRouter
    embedding model. Higher similarity suggests better meaning preservation."""
    source_text = str(source_text).strip()
    translation_text = str(translation_text).strip()

    if not source_text or not translation_text:
        return {
            "mechanism": "embedding_similarity",
            "model": model,
            "available": False,
            "reason": "empty_text",
        }

    response = client.embeddings.create(
        model=model,
        input=[source_text, translation_text],
    )
    source_vec = response.data[0].embedding
    translation_vec = response.data[1].embedding
    similarity = _cosine_similarity(source_vec, translation_vec)

    return {
        "mechanism": "embedding_similarity",
        "model": model,
        "available": True,
        "cosine_similarity": round(similarity, 4),
        "source_length_chars": len(source_text),
        "translation_length_chars": len(translation_text),
    }


def format_embedding_similarity_for_prompt(feedback: dict[str, Any]) -> str:
    if not isinstance(feedback, dict):
        return ""
    if not feedback.get("available"):
        reason = str(feedback.get("reason", "unavailable")).strip()
        return f"Embedding faithfulness proxy unavailable ({reason})."

    model_name = str(feedback.get("model", "")).strip() or "unknown"
    sim = feedback.get("cosine_similarity")
    sim_str = f"{float(sim):.4f}" if isinstance(sim, (int, float)) else "n/a"

    return (
        f"Faithfulness proxy ({model_name}): cosine_similarity={sim_str}. "
        "Higher = more meaning preserved between source and translation. "
        "Typical good translations score >0.75."
    )
