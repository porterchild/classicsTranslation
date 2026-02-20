from __future__ import annotations

import json
import math
import os
import re
import time
from http.client import IncompleteRead
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI


_LLAMACPP_INFO_CACHE: dict[str, dict[str, Any]] = {}
LOCAL_MODEL_ALIAS = "local_model"


def _short_reason(value: Any, limit: int = 280) -> str:
    text = str(value).replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_completion_logprobs(response: Any) -> list[float]:
    choices = getattr(response, "choices", None)
    if not isinstance(choices, list) or not choices:
        return []
    choice = choices[0]

    logprobs_obj = getattr(choice, "logprobs", None)
    if logprobs_obj is None:
        return []

    values: list[float] = []
    token_logprobs = getattr(logprobs_obj, "token_logprobs", None)
    if token_logprobs is None and isinstance(logprobs_obj, dict):
        token_logprobs = logprobs_obj.get("token_logprobs")
    if isinstance(token_logprobs, list):
        for value in token_logprobs:
            if isinstance(value, (int, float)) and math.isfinite(value):
                values.append(float(value))
        if values:
            return values

    content_items = getattr(logprobs_obj, "content", None)
    if content_items is None and isinstance(logprobs_obj, dict):
        content_items = logprobs_obj.get("content")
    if not isinstance(content_items, list):
        return []

    for item in content_items:
        logprob = None
        if isinstance(item, dict):
            logprob = item.get("logprob")
        else:
            logprob = getattr(item, "logprob", None)
        if isinstance(logprob, (int, float)) and math.isfinite(logprob):
            values.append(float(logprob))
    return values


def _build_score_payload(
    *,
    mechanism: str,
    model: str,
    logprobs: list[float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not logprobs:
        return {
            "mechanism": mechanism,
            "model": model,
            "available": False,
            "reason": "missing_logprobs_or_no_tokens",
        }

    token_count = len(logprobs)
    avg_logprob = sum(logprobs) / token_count
    perplexity = math.exp(-avg_logprob)
    payload = {
        "mechanism": mechanism,
        "model": model,
        "available": True,
        "token_count": token_count,
        "avg_logprob": avg_logprob,
        "perplexity": perplexity,
    }
    if extra:
        payload.update(extra)
    return payload


def _is_local_llamacpp_model(model: str) -> bool:
    model_name = str(model).strip().lower()
    return (
        model_name == LOCAL_MODEL_ALIAS
        or model_name.endswith(".gguf")
        or model_name.startswith("llamacpp/")
    )


def _http_get_json(base_url: str, path: str, timeout: int) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    req = Request(url, method="GET")
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(
    base_url: str,
    path: str,
    payload: dict[str, Any],
    timeout: int,
    retries: int = 3,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    last_error: Exception | None = None
    for _ in range(max(1, retries)):
        try:
            req = Request(url, data=body, headers=headers, method="POST")
            with urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except IncompleteRead as exc:
            last_error = exc
            continue
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            continue
    assert last_error is not None
    raise last_error


def _llamacpp_model_info(base_url: str, timeout: int) -> dict[str, Any]:
    cached = _LLAMACPP_INFO_CACHE.get(base_url)
    if isinstance(cached, dict):
        return cached

    models_data = _http_get_json(base_url, "/v1/models", timeout=timeout)
    data_entries = models_data.get("data", [])
    if not isinstance(data_entries, list) or not data_entries:
        raise RuntimeError("llamacpp_models_unavailable")
    first_entry = data_entries[0] if isinstance(data_entries[0], dict) else {}
    default_model_id = str(first_entry.get("id", "")).strip()
    if not default_model_id:
        raise RuntimeError("llamacpp_default_model_unavailable")
    meta = first_entry.get("meta", {})
    n_vocab = meta.get("n_vocab") if isinstance(meta, dict) else None
    if not isinstance(n_vocab, int) or n_vocab < 1:
        raise RuntimeError("llamacpp_n_vocab_unavailable")

    props_data = _http_get_json(base_url, "/props", timeout=timeout)
    bos_token = str(props_data.get("bos_token", "")).strip()
    if not bos_token:
        raise RuntimeError("llamacpp_bos_token_unavailable")

    tokenized_bos = _http_post_json(
        base_url,
        "/tokenize",
        {"content": bos_token},
        timeout=timeout,
        retries=2,
    )
    bos_tokens = tokenized_bos.get("tokens", [])
    if not isinstance(bos_tokens, list) or not bos_tokens or not isinstance(bos_tokens[0], int):
        raise RuntimeError("llamacpp_bos_tokenization_failed")
    bos_id = int(bos_tokens[0])

    info = {
        "default_model_id": default_model_id,
        "n_vocab": n_vocab,
        "bos_id": bos_id,
        "bos_token": bos_token,
    }
    _LLAMACPP_INFO_CACHE[base_url] = info
    return info


def _extract_llamacpp_top_logprobs(response: dict[str, Any]) -> list[dict[str, Any]]:
    probs = response.get("completion_probabilities", [])
    if not isinstance(probs, list) or not probs:
        return []
    first = probs[0] if isinstance(probs[0], dict) else {}
    top = first.get("top_logprobs", []) if isinstance(first, dict) else []
    if not isinstance(top, list):
        return []
    values: list[dict[str, Any]] = []
    for item in top:
        if isinstance(item, dict):
            values.append(item)
    return values


def _llamacpp_step_logprob(
    *,
    base_url: str,
    model_name: str,
    prefix_tokens: list[int],
    target_token_id: int,
    n_vocab: int,
    timeout: int,
) -> tuple[float, int]:
    initial_top_n_raw = os.getenv("LLAMACPP_PERPLEXITY_TOP_N", "256").strip()
    expansion_factor_raw = os.getenv("LLAMACPP_PERPLEXITY_EXPANSION_FACTOR", "4").strip()
    try:
        initial_top_n = max(8, int(initial_top_n_raw))
    except ValueError:
        initial_top_n = 256
    try:
        expansion_factor = max(2, int(expansion_factor_raw))
    except ValueError:
        expansion_factor = 4

    n_probs = min(n_vocab, initial_top_n)
    expansions = 0
    while True:
        payload: dict[str, Any] = {
            "prompt": prefix_tokens,
            "n_predict": 0,
            "temperature": 0,
            "n_probs": n_probs,
            "post_sampling_probs": False,
        }
        if model_name:
            payload["model"] = model_name

        response = _http_post_json(
            base_url,
            "/completion",
            payload,
            timeout=max(timeout, 120),
            retries=4,
        )
        top_items = _extract_llamacpp_top_logprobs(response)
        found = next(
            (
                item
                for item in top_items
                if isinstance(item.get("id"), int) and int(item["id"]) == target_token_id
            ),
            None,
        )
        if isinstance(found, dict):
            logprob = found.get("logprob")
            if isinstance(logprob, (int, float)) and math.isfinite(logprob):
                return float(logprob), expansions
            raise RuntimeError("llamacpp_target_logprob_invalid")

        if n_probs >= n_vocab:
            raise RuntimeError("llamacpp_target_token_not_found")

        n_probs = min(n_vocab, n_probs * expansion_factor)
        expansions += 1


def _score_with_llamacpp_exact_perplexity(
    *,
    model: str,
    text: str,
    timeout: int,
) -> dict[str, Any]:
    base_url = str(os.getenv("LLAMACPP_BASE_URL", "http://localhost:8081")).strip()
    try:
        model_info = _llamacpp_model_info(base_url, timeout=max(timeout, 30))
        requested_model = str(model).strip()
        requested_lower = requested_model.lower()
        if requested_lower == LOCAL_MODEL_ALIAS:
            model_name = str(model_info.get("default_model_id", "")).strip()
        elif requested_lower.startswith("llamacpp/"):
            model_name = requested_model.split("/", 1)[1].strip()
        else:
            model_name = requested_model

        if not model_name:
            return {
                "mechanism": "llamacpp_exact_token_logprobs",
                "model": model,
                "available": False,
                "reason": "llamacpp_model_name_unavailable",
            }

        n_vocab = int(model_info["n_vocab"])
        bos_id = int(model_info["bos_id"])

        tokenized = _http_post_json(
            base_url,
            "/tokenize",
            {"content": text},
            timeout=max(timeout, 60),
            retries=2,
        )
        target_tokens = tokenized.get("tokens", [])
        if not isinstance(target_tokens, list) or not target_tokens:
            return {
                "mechanism": "llamacpp_exact_token_logprobs",
                "model": model,
                "available": False,
                "reason": "llamacpp_tokenize_empty",
            }
        if not all(isinstance(token_id, int) for token_id in target_tokens):
            return {
                "mechanism": "llamacpp_exact_token_logprobs",
                "model": model,
                "available": False,
                "reason": "llamacpp_tokenize_invalid",
            }

        logprobs: list[float] = []
        expansion_steps = 0
        for idx, token_id in enumerate(target_tokens):
            prefix_tokens = [bos_id] if idx == 0 else target_tokens[:idx]
            if not prefix_tokens:
                return {
                    "mechanism": "llamacpp_exact_token_logprobs",
                    "model": model,
                    "available": False,
                    "reason": "llamacpp_empty_prefix",
                }
            step_logprob, step_expansions = _llamacpp_step_logprob(
                base_url=base_url,
                model_name=model_name,
                prefix_tokens=prefix_tokens,
                target_token_id=int(token_id),
                n_vocab=n_vocab,
                timeout=timeout,
            )
            logprobs.append(step_logprob)
            expansion_steps += step_expansions

        score = _build_score_payload(
            mechanism="llamacpp_exact_token_logprobs",
            model=model,
            logprobs=logprobs,
            extra={
                "base_url": base_url,
                "resolved_model": model_name,
                "n_vocab": n_vocab,
                "expansion_steps": expansion_steps,
                "token_scoring": "exact",
            },
        )
        return score
    except (URLError, HTTPError, RuntimeError, TimeoutError) as exc:
        return {
            "mechanism": "llamacpp_exact_token_logprobs",
            "model": model,
            "available": False,
            "reason": f"request_failed: {_short_reason(exc)}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "mechanism": "llamacpp_exact_token_logprobs",
            "model": model,
            "available": False,
            "reason": f"unexpected_failure: {_short_reason(exc)}",
        }


def _score_with_prompt_echo_logprobs(
    *,
    client: OpenAI,
    model: str,
    text: str,
    timeout: int,
) -> dict[str, Any]:
    try:
        response = client.completions.create(
            model=model,
            prompt=text,
            max_tokens=0,
            echo=True,
            temperature=0,
            logprobs=5,
            timeout=timeout,
            extra_body={"provider": {"require_parameters": True}},
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "mechanism": "prompt_echo_logprobs",
            "model": model,
            "available": False,
            "reason": f"request_failed: {_short_reason(exc)}",
        }

    score = _build_score_payload(
        mechanism="prompt_echo_logprobs",
        model=model,
        logprobs=_extract_completion_logprobs(response),
    )
    choices = getattr(response, "choices", None)
    echoed_text = ""
    if isinstance(choices, list) and choices:
        echoed_text = str(getattr(choices[0], "text", "") or "")
    if echoed_text != text:
        return {
            "mechanism": "prompt_echo_logprobs",
            "model": model,
            "available": False,
            "reason": "prompt_echo_not_honored",
        }
    if not score.get("available"):
        score["reason"] = "prompt_echo_unavailable_or_ignored"
    return score


def compute_smoothness_feedback_from_perplexity(
    *,
    client: OpenAI,
    model: str,
    text: str,
    timeout: int = 60,
) -> dict[str, Any]:
    text = str(text).strip()
    if not text:
        return {
            "mechanism": "small_lm_perplexity",
            "model": model,
            "available": False,
            "reason": "empty_text",
        }

    if _is_local_llamacpp_model(model):
        return _score_with_llamacpp_exact_perplexity(
            model=model,
            text=text,
            timeout=timeout,
        )

    prompt_echo_score = _score_with_prompt_echo_logprobs(
        client=client,
        model=model,
        text=text,
        timeout=timeout,
    )
    if prompt_echo_score.get("available"):
        return prompt_echo_score

    return {
        "mechanism": "prompt_echo_logprobs",
        "model": model,
        "available": False,
        "reason": str(prompt_echo_score.get("reason", "prompt_echo_unavailable")).strip(),
    }


def format_smoothness_feedback_for_prompt(feedback: dict[str, Any]) -> str:
    if not isinstance(feedback, dict):
        return ""
    if not feedback.get("available"):
        reason = str(feedback.get("reason", "unavailable")).strip()
        return f"Perplexity feedback unavailable ({reason})."

    model = str(feedback.get("model", "")).strip() or "unknown_model"
    ppl = feedback.get("perplexity")
    avg_lp = feedback.get("avg_logprob")
    tok = feedback.get("token_count")
    ppl_str = f"{float(ppl):.3f}" if isinstance(ppl, (int, float)) else "n/a"
    lp_str = f"{float(avg_lp):.4f}" if isinstance(avg_lp, (int, float)) else "n/a"
    tok_str = str(tok) if isinstance(tok, int) else "n/a"

    if str(feedback.get("mechanism", "")).strip() == "llamacpp_exact_token_logprobs":
        expansions = feedback.get("expansion_steps")
        exp_str = str(expansions) if isinstance(expansions, int) else "n/a"
        return (
            f"Perplexity feedback from {model} (llama.cpp exact token scoring): "
            f"perplexity={ppl_str}, avg_logprob={lp_str}, token_count={tok_str}, "
            f"expansion_steps={exp_str}."
        )

    return (
        f"Perplexity feedback from {model} (prompt-echo logprobs): "
        f"perplexity={ppl_str}, avg_logprob={lp_str}, token_count={tok_str}. "
        "Lower perplexity usually indicates smoother local phrasing."
    )


# ---------------------------------------------------------------------------
# Grade Level (Flesch-Kincaid) — pure Python, zero extra dependencies
# ---------------------------------------------------------------------------

_VOWELS = set("aeiouyAEIOUY")


def _count_syllables(word: str) -> int:
    """Heuristic English syllable count."""
    word = word.lower().strip()
    if not word:
        return 0
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
        return {"mechanism": "grade_level", "available": False, "reason": "empty_text"}

    words = _tokenize_words(text)
    sentences = _split_sentences(text)
    word_count = len(words)
    sentence_count = max(len(sentences), 1)

    if word_count < 3:
        return {"mechanism": "grade_level", "available": False, "reason": "too_few_words"}

    syllable_count = sum(_count_syllables(w) for w in words)
    avg_words_per_sentence = word_count / sentence_count
    avg_syllables_per_word = syllable_count / word_count

    fk_grade = 0.39 * avg_words_per_sentence + 11.8 * avg_syllables_per_word - 15.59
    fk_reading_ease = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word

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

    response = client.embeddings.create(model=model, input=[source_text, translation_text])
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


# ---------------------------------------------------------------------------
# Shared: JSON parsing + LLM call helper
# ---------------------------------------------------------------------------

def _parse_json_object(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        value = json.loads(match.group(0))
        if isinstance(value, dict):
            return value
    raise ValueError(f"Model did not return valid JSON object:\n{text}")


def _call_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
    retries: int = 3,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                timeout=120,
                extra_body={"reasoning": {"enabled": True}},
            )
            content = resp.choices[0].message.content or ""
            return _parse_json_object(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------------
# Back-Translation Divergence — translate English→Greek, compare to original
# ---------------------------------------------------------------------------

def compute_back_translation(
    *,
    client: OpenAI,
    model: str,
    original_greek: str,
    translation: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """Translate English back to Greek, then measure embedding similarity
    between the back-translated Greek and the original Greek (same-language
    comparison, much higher signal than cross-lingual)."""
    original_greek = str(original_greek).strip()
    translation = str(translation).strip()

    if not original_greek or not translation:
        return {
            "mechanism": "back_translation",
            "model": model,
            "available": False,
            "reason": "empty_text",
        }

    back_result = _call_json(
        client,
        model,
        system_prompt=(
            "You are a translator. Translate the given English text into "
            "Ancient Greek. Preserve the meaning as closely as possible. "
            "Output JSON only."
        ),
        user_prompt=(
            f"Translate this English into Ancient Greek:\n\n{translation}\n\n"
            'Return strict JSON: {{"greek": "..."}}'
        ),
        temperature=0.2,
    )
    back_greek = str(back_result.get("greek", "")).strip()

    if not back_greek:
        return {
            "mechanism": "back_translation",
            "model": model,
            "available": False,
            "reason": "empty_back_translation",
        }

    similarity = compute_embedding_similarity(
        client=client,
        source_text=original_greek,
        translation_text=back_greek,
        model=embedding_model,
    )
    cosine = similarity.get("cosine_similarity", 0.0)

    return {
        "mechanism": "back_translation",
        "model": model,
        "embedding_model": embedding_model,
        "available": True,
        "back_greek": back_greek,
        "cosine_similarity": cosine,
    }


def format_back_translation_for_prompt(feedback: dict[str, Any]) -> str:
    if not isinstance(feedback, dict):
        return ""
    if not feedback.get("available"):
        reason = str(feedback.get("reason", "unavailable")).strip()
        return f"Back-translation faithfulness check unavailable ({reason})."

    sim = feedback.get("cosine_similarity")
    sim_str = f"{float(sim):.4f}" if isinstance(sim, (int, float)) else "n/a"

    return (
        f"Back-translation faithfulness: cosine_similarity={sim_str} "
        "(original Greek vs back-translated Greek). "
        "Higher = meaning better preserved through round-trip. "
        "Same-language comparison, more reliable than cross-lingual."
    )


# ---------------------------------------------------------------------------
# Entity/Relation Extraction — extract checklist from Greek, verify in English
# ---------------------------------------------------------------------------

def extract_entities_and_relations(
    *,
    client: OpenAI,
    model: str,
    greek: str,
) -> dict[str, Any]:
    """Extract named entities, causal relations, and key contrasts from Greek
    source. Run once per paragraph, then check candidates against the list."""
    greek = str(greek).strip()
    if not greek:
        return {
            "mechanism": "entity_relation_extraction",
            "available": False,
            "reason": "empty_text",
        }

    result = _call_json(
        client,
        model,
        system_prompt=(
            "You are a classical Greek philologist. Extract the key semantic "
            "elements from the given Ancient Greek passage. Be exhaustive but "
            "concise. Output JSON only."
        ),
        user_prompt=f"""Ancient Greek passage:
{greek}

Extract:
1) Named entities (people, places, works) — use their conventional English names.
2) Key relations (who does what to whom, causal links, analogies).
3) Key contrasts or oppositions the passage sets up.
4) Modal stance (wish, assertion, command, condition, etc.).

Return strict JSON:
{{
  "entities": ["entity1", "entity2", ...],
  "relations": ["subject verb object/description", ...],
  "contrasts": ["X vs Y", ...],
  "modal_stance": "description of the mood/modality"
}}""",
        temperature=0.1,
    )

    entities = result.get("entities", [])
    relations = result.get("relations", [])
    contrasts = result.get("contrasts", [])
    modal_stance = str(result.get("modal_stance", "")).strip()

    if not isinstance(entities, list):
        entities = []
    if not isinstance(relations, list):
        relations = []
    if not isinstance(contrasts, list):
        contrasts = []

    return {
        "mechanism": "entity_relation_extraction",
        "available": True,
        "entities": [str(e).strip() for e in entities if str(e).strip()],
        "relations": [str(r).strip() for r in relations if str(r).strip()],
        "contrasts": [str(c).strip() for c in contrasts if str(c).strip()],
        "modal_stance": modal_stance,
    }


def check_entities_in_translation(
    *,
    extraction: dict[str, Any],
    translation: str,
) -> dict[str, Any]:
    """Check which extracted entities/relations appear in a translation.
    Uses case-insensitive substring matching on content words (>2 chars).
    Known limitation: any-word matching means common content words satisfy
    items even when polarity/meaning is reversed."""
    translation = str(translation).strip()
    if not extraction.get("available"):
        return {"mechanism": "entity_relation_check", "available": False, "reason": "no_extraction"}
    if not translation:
        return {
            "mechanism": "entity_relation_check",
            "available": False,
            "reason": "empty_translation",
        }

    translation_lower = translation.lower()

    def _check_items(items: list[str]) -> dict[str, bool]:
        results: dict[str, bool] = {}
        for item in items:
            item_lower = item.lower()
            words = [w for w in re.findall(r"[a-z]+", item_lower) if len(w) > 2]
            if not words:
                results[item] = item_lower in translation_lower
            else:
                results[item] = any(w in translation_lower for w in words)
        return results

    entity_checks = _check_items(extraction.get("entities", []))
    relation_checks = _check_items(extraction.get("relations", []))
    contrast_checks = _check_items(extraction.get("contrasts", []))

    all_checks = {**entity_checks, **relation_checks, **contrast_checks}
    total = len(all_checks)
    found = sum(1 for v in all_checks.values() if v)
    missing = [k for k, v in all_checks.items() if not v]

    return {
        "mechanism": "entity_relation_check",
        "available": True,
        "total_items": total,
        "found": found,
        "coverage": round(found / total, 3) if total > 0 else 1.0,
        "missing": missing,
        "entity_results": entity_checks,
        "relation_results": relation_checks,
        "contrast_results": contrast_checks,
        "modal_stance": extraction.get("modal_stance", ""),
    }


def format_entity_check_for_prompt(feedback: dict[str, Any]) -> str:
    if not isinstance(feedback, dict):
        return ""
    if not feedback.get("available"):
        reason = str(feedback.get("reason", "unavailable")).strip()
        return f"Entity/relation check unavailable ({reason})."

    total = feedback.get("total_items", 0)
    found = feedback.get("found", 0)
    coverage = feedback.get("coverage", 0)
    missing = feedback.get("missing", [])
    modal = feedback.get("modal_stance", "")

    cov_str = f"{float(coverage) * 100:.0f}%" if isinstance(coverage, (int, float)) else "n/a"
    parts = [f"Entity/relation coverage: {found}/{total} ({cov_str})."]
    if missing:
        parts.append(f"Missing: {'; '.join(missing[:5])}.")
    if modal:
        parts.append(f"Expected modal stance: {modal}.")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Experiment runner — python translation_feedback_mechanisms.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    def _load_dotenv(path: Path) -> None:
        if not path.exists():
            return
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

    _load_dotenv(Path(".env"))
    api_key = (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        sys.exit("Missing OPENROUTER_API_KEY")

    client = OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1")
    MODEL = "x-ai/grok-4.1-fast"

    GREEK_P3 = (
        "εἴη μὲν οὖν ἡμῖν ἐκκαθαιρόμενον λόγῳ τὸ μυθῶδες ὑπακοῦσαι καὶ λαβεῖν "
        "ἱστορίας ὄψιν, ὅπου δ᾽ ἂν αὐθαδῶς τοῦ πιθανοῦ περιφρονῇ καὶ μὴ δέχηται "
        "τὴν πρὸς τὸ εἰκὸς μῖξιν, εὐγνωμόνων ἀκροατῶν δεησόμεθα καὶ πρᾴως τὴν "
        "ἀρχαιολογίαν προσδεχομένων."
    )

    VARIANTS: dict[str, str] = {
        "faithful_perrin": (
            "May I therefore succeed in purifying Fable, making her submit to reason and take on "
            "the semblance of History. But where she obstinately disdains to make herself credible, "
            "and refuses to admit any element of probability, I shall pray for kindly readers, and "
            "such as receive with indulgence the tales of antiquity."
        ),
        "faithful_modern": (
            "So may the mythical element yield to reasoned purification and take on a historical "
            "appearance; but where it stubbornly disdains plausibility and refuses mixture with the "
            "probable, we will ask for sympathetic readers who receive ancient accounts graciously."
        ),
        "reverses_stubbornness": (
            "So may the mythical element yield to reasoned purification and take on a historical "
            "appearance; but where it readily accepts being made plausible and welcomes mixture with "
            "the probable, we will ask for sympathetic readers who receive ancient accounts graciously."
        ),
        "drops_second_half": (
            "So may the mythical element yield to reasoned purification and take on a historical "
            "appearance."
        ),
        "entity_swap": (
            "So may the mythical element yield to reasoned purification and take on a historical "
            "appearance; but where it stubbornly disdains plausibility and refuses mixture with the "
            "probable, we will ask for sympathetic writers who record ancient accounts carefully."
        ),
        "misread_archaeology": (
            "So may the mythical element yield to reasoned purification and take on a historical "
            "appearance; but where it stubbornly disdains plausibility and refuses mixture with the "
            "probable, we will ask for sympathetic readers who examine the archaeology open-mindedly."
        ),
    }

    print("=" * 60)
    print("ENTITY/RELATION EXTRACTION")
    print("=" * 60)
    extraction = extract_entities_and_relations(client=client, model=MODEL, greek=GREEK_P3)
    print(f"entities    : {extraction.get('entities')}")
    print(f"relations   : {extraction.get('relations')}")
    print(f"contrasts   : {extraction.get('contrasts')}")
    print(f"modal_stance: {extraction.get('modal_stance')}")
    print()

    print("=" * 60)
    print("ENTITY CHECK per variant")
    print("=" * 60)
    for name, text in VARIANTS.items():
        check = check_entities_in_translation(extraction=extraction, translation=text)
        cov = check.get("coverage", 0)
        found = check.get("found", 0)
        total = check.get("total_items", 0)
        print(f"\nVARIANT: {name}  coverage={cov:.2f} ({found}/{total})")
        print(f"  Text: {text[:120]}...")
        all_results = {
            **check.get("entity_results", {}),
            **check.get("relation_results", {}),
            **check.get("contrast_results", {}),
        }
        for item, matched in all_results.items():
            mark = "✓" if matched else "✗"
            print(f"    {mark} {item}")
    print()
