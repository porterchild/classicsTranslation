from __future__ import annotations

import json
import math
import os
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
