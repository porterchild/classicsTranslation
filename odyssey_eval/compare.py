"""Comparison agent: given (values_profile, known_passage, pipeline_output) → score + rationale."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

COMPARE_MODEL = "x-ai/grok-4.1-fast"

_SYSTEM = """\
You are an expert literary critic and classicist specializing in translation studies.
You evaluate machine translations of Homer's Odyssey against a specific human translator's
style and values.

Your job is NOT to judge whether the translation is "good" in an abstract sense.
Your job is to judge how well it matches the target translator's specific approach —
their register, archaism level, rhythm, epithet handling, and taste decisions.

Be honest and precise. A translation that is clear and modern is NOT a good match
for Chapman's archaic verse. A translation that uses "thee" and "thou" is NOT a good
match for Butler's plain prose. Score accordingly.
"""

_USER_TEMPLATE = """\
## Target Translator Profile
{values_profile}

## Known Translation (ground truth for this passage)
{known_passage}

## Pipeline Output (translation to evaluate)
{pipeline_output}

---

Score the pipeline output on how well it matches the TARGET TRANSLATOR'S STYLE AND VALUES.

Scale:
  0-3: Fundamental mismatch — wrong register, wrong approach, could not be confused with this translator
  4-6: Right general direction but missing key style markers
  7-8: Clearly in the same spirit, minor deviations
  9-10: Could plausibly be mistaken for the actual translation

Return strict JSON:
{{
  "score": <integer 0-10>,
  "rationale": "<2-4 sentences explaining the score>",
  "key_gaps": ["<specific thing missing or wrong>", ...],
  "key_matches": ["<specific thing done right>", ...]
}}
"""


def _parse_json(text: str) -> dict[str, Any]:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        v = json.loads(raw)
        if isinstance(v, dict):
            return v
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        v = json.loads(m.group(0))
        if isinstance(v, dict):
            return v
    raise ValueError(f"No JSON object found in response:\n{text[:500]}")


def compare(
    *,
    client: OpenAI,
    values_profile: str,
    known_passage: str,
    pipeline_output: str,
    model: str = COMPARE_MODEL,
    retries: int = 3,
) -> dict[str, Any]:
    """Call the comparison agent and return score + rationale."""
    user_prompt = _USER_TEMPLATE.format(
        values_profile=values_profile.strip(),
        known_passage=known_passage.strip(),
        pipeline_output=pipeline_output.strip(),
    )
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,
                timeout=120,
                extra_body={"reasoning": {"enabled": True}},
            )
            content = resp.choices[0].message.content or ""
            result = _parse_json(content)
            # Ensure score is an int in range
            score = result.get("score")
            if not isinstance(score, (int, float)):
                raise ValueError(f"Missing or invalid score: {score}")
            result["score"] = max(0, min(10, int(score)))
            return result
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    assert last_err is not None
    raise last_err
