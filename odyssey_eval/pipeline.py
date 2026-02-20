"""Run the sequential translation pipeline on a single Odyssey passage.

Wraps the existing sequential pipeline prompts directly so we can use them
for Odyssey passages without the Plutarch-specific reference translations
or the full main.py machinery.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

from openai import OpenAI

DEFAULT_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_ITERATIONS = 2

GOALS_GUIDANCE = (
    "- faithfulness: how strictly similar to the source language is it?\n"
    "- readability: how well does it flow, does it minimize convoluted sentences?\n"
    "- style match: does it match the target translator's register, tone, and values?"
)


# ---------------------------------------------------------------------------
# Minimal JSON call helper
# ---------------------------------------------------------------------------

def _call_json(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float = 0.4,
    retries: int = 3,
) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                timeout=120,
                extra_body={"reasoning": {"enabled": True}},
            )
            content = resp.choices[0].message.content or ""
            return _parse_json(content)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    assert last_err is not None
    raise last_err


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
    raise ValueError(f"No JSON object in response:\n{text[:400]}")


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _is_verse_translator(values_profile: str) -> bool:
    """Return True if this profile requires verse / rhyming output (e.g. Chapman)."""
    markers = ["HEROIC COUPLETS", "RHYME IS REQUIRED", "rhyming iambic pentameter"]
    return any(m.lower() in values_profile.lower() for m in markers)


def _translate_prompt(
    greek: str,
    values_profile: str,
    iteration: int,
    previous_translation: str | None,
    previous_judgment: dict[str, Any] | None,
) -> tuple[str, str]:
    system = (
        "You are a translator of Ancient Greek. Your sole goal is to match "
        "the style and values of a specific target translator. Output JSON only."
    )
    prev_block = ""
    if previous_translation:
        prev_block = f"\nPrevious draft:\n{previous_translation}\n"
    judge_block = ""
    if previous_judgment:
        judge_block = (
            f"\nJudge feedback:\n"
            f"{previous_judgment.get('issues', '')}\n"
            f"Revision plan: {previous_judgment.get('revision_plan', '')}\n"
        )

    # Extra instructions for verse translators (e.g. Chapman's heroic couplets)
    verse_block = ""
    if _is_verse_translator(values_profile):
        verse_block = """
VERSE FORM CHECKLIST (complete before finalizing):
- Every pair of consecutive lines MUST end in a rhyme (aa, bb, cc...).
- Count syllables: each line should be ~10 syllables (iambic pentameter).
- Before writing, jot the intended end-words for each couplet.
- After writing, read out the last word of each line in sequence and verify they rhyme pairwise.
- If any pair does not rhyme, rewrite those lines before submitting.
- Enjambment is encouraged — sense should run across couplet boundaries freely.
"""

    user = f"""Greek passage to translate:
{greek}

Target translator's style and values (match this):
{values_profile}

Iteration: {iteration}
{prev_block}{judge_block}{verse_block}
Task:
1) Write brief planning notes: what style/register choices will match the target translator?
{"   For verse: plan your rhyme scheme — write the intended END WORD of each couplet pair." if _is_verse_translator(values_profile) else ""}
2) Produce one translation of this passage that matches the target translator's style.
3) Prioritize style match over generic quality. If the target is archaic — be archaic.
   If the target is plain prose — be plain. Do not default to generic modern English.
4) Preserve the core meaning of the Greek throughout.

Return strict JSON:
{{
  "observations": "brief planning notes{' + rhyme plan: list end-words for each couplet' if _is_verse_translator(values_profile) else ''}",
  "translation": "...",
  "self_scores": {{
    "faithfulness": 1-10,
    "style_match": 1-10
  }}
}}""".strip()
    return system, user


def _judge_prompt(
    greek: str,
    values_profile: str,
    translation: str,
    iteration: int,
) -> tuple[str, str]:
    system = (
        "You are a strict critic evaluating a translation against a specific target "
        "translator's style. Be concrete. Output JSON only."
    )

    verse_check = ""
    if _is_verse_translator(values_profile):
        verse_check = """
VERSE FORM CHECK (do this before other comments):
- List every consecutive line pair and their end words, e.g. "L1: 'way' / L2: 'stay' → RHYMES"
- Mark each pair RHYMES or FAILS.
- Count how many pairs fail.
- A single unrhymed pair is a significant deduction; many unrhymed pairs = low style_match.
- Also note lines that are clearly not ~10 syllables (iambic pentameter).
"""

    user = f"""Greek passage:
{greek}

Target translator's style and values:
{values_profile}

Translation to judge (iteration {iteration}):
{translation}
{verse_check}
Judge how well this matches the target translator's style. Be specific about:
- Register (right level of formality / archaism?)
- Epithet handling (does it match the target's approach?)
- Sentence structure and rhythm
- Vocabulary choices
- What is most wrong and how to fix it

Return strict JSON:
{{
  "overall_judgment": "candid assessment",
  "strengths": "what matches the target style",
  "issues": "what does not match",
  "revision_plan": "concrete edits for next iteration — for verse: list specific line pairs to fix and suggest rhyming end-words",
  "scores": {{
    "faithfulness": 1-10,
    "style_match": 1-10
  }}
}}""".strip()
    return system, user


def _select_prompt(
    greek: str,
    values_profile: str,
    iteration_logs: list[dict[str, Any]],
) -> tuple[str, str]:
    system = (
        "You are the final selector. Choose and refine the best draft "
        "for style match with the target translator. Output JSON only."
    )
    compact = json.dumps(
        [
            {
                "iteration": log["iteration"],
                "translation": log["translation"],
                "style_match": log.get("judgment", {}).get("scores", {}).get("style_match"),
                "issues": log.get("judgment", {}).get("issues", ""),
                "revision_plan": log.get("judgment", {}).get("revision_plan", ""),
            }
            for log in iteration_logs
        ],
        ensure_ascii=False,
        indent=2,
    )
    user = f"""Greek passage:
{greek}

Target translator's style and values:
{values_profile}

Candidate iterations:
{compact}

Select and lightly refine the draft that best matches the target translator.
You may rewrite but keep the same passage and meaning.

Return strict JSON:
{{
  "selected_iteration": <int>,
  "final_translation": "...",
  "justification": "why this best matches the target style"
}}""".strip()
    return system, user


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_passage(
    *,
    client: OpenAI,
    greek: str,
    values_profile: str,
    model: str = DEFAULT_MODEL,
    iterations: int = DEFAULT_ITERATIONS,
    verbose: bool = False,
) -> dict[str, Any]:
    """Translate a single Greek passage, iterating toward the target style.

    Returns a dict with 'final_translation' and iteration logs.
    """
    current_translation = ""
    current_judgment: dict[str, Any] | None = None
    iteration_logs: list[dict[str, Any]] = []

    for it in range(1, iterations + 1):
        if verbose:
            print(f"  [iter {it}] translating...", flush=True)
        sys_t, usr_t = _translate_prompt(
            greek=greek,
            values_profile=values_profile,
            iteration=it,
            previous_translation=current_translation or None,
            previous_judgment=current_judgment,
        )
        t_result = _call_json(client, DEFAULT_MODEL if model == DEFAULT_MODEL else model,
                              sys_t, usr_t, temperature=0.45)
        current_translation = str(t_result.get("translation", "")).strip()

        if verbose:
            print(f"  [iter {it}] judging...", flush=True)
        sys_j, usr_j = _judge_prompt(
            greek=greek,
            values_profile=values_profile,
            translation=current_translation,
            iteration=it,
        )
        current_judgment = _call_json(client, model, sys_j, usr_j, temperature=0.3)

        iteration_logs.append(
            {
                "iteration": it,
                "translation": current_translation,
                "translate_result": t_result,
                "judgment": current_judgment,
            }
        )

    if verbose:
        print("  [select] choosing best iteration...", flush=True)
    sys_s, usr_s = _select_prompt(
        greek=greek,
        values_profile=values_profile,
        iteration_logs=iteration_logs,
    )
    select_result = _call_json(client, model, sys_s, usr_s, temperature=0.25)
    final_translation = str(select_result.get("final_translation", "")).strip()
    if not final_translation:
        final_translation = current_translation

    return {
        "greek": greek,
        "final_translation": final_translation,
        "selected_iteration": select_result.get("selected_iteration"),
        "justification": select_result.get("justification", ""),
        "iteration_logs": iteration_logs,
    }
