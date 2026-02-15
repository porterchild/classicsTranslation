from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Callable

from openai import OpenAI

from .cognitive_logging import (
    log_dualloop_iteration,
    log_final_selection,
    log_iteration_focus,
    log_reference_inputs,
    make_vprint,
)
from .common import reference_context_block, reference_translations_for_index


def dual_loop_translate_prompt(
    greek: str,
    paragraph_index: int,
    reference_translations: dict[str, str],
    user_preference: str,
    goals_guidance: str,
    iteration: int,
    previous_translation: str | None,
    previous_focus: str | None,
) -> tuple[str, str]:
    system = (
        "You are a dual-loop translation agent for Ancient Greek -> modern English. "
        "Run a meaning loop first, then a wording loop. Output JSON only."
    )
    refs = reference_context_block(reference_translations)
    prev_translation_block = ""
    if previous_translation:
        prev_translation_block = (
            f"\nPrevious iteration translation:\n{previous_translation}\n"
        )
    prev_focus_block = ""
    if previous_focus:
        prev_focus_block = (
            "\nCarry-forward focus from previous iteration:\n"
            f"- {previous_focus}\n"
        )

    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

{refs}

User preference prompt:
{user_preference}

Iteration: {iteration}
{prev_translation_block}{prev_focus_block}
Task (dual-loop process):
Meaning loop:
1) Build a scene model: speaker, addressee, intent, stance, tone.
2) Decompose into atomic claims and relations (contrast, condition, modality, rhetoric).
3) Write a plain restatement for a young reader.
4) Build a constraint ledger:
   - non-negotiables (must preserve),
   - negotiables (may rephrase freely).

Wording loop:
5) Draft three variants: source-close, plain-natural, and balanced.
6) Merge into one final paragraph for publication quality.
7) Prioritize user preference first, then balance:
{goals_guidance}
8) Keep tone clear, readable, and modern while preserving rhetorical intent.
9) Readability for younger audiences means clarity and plain syntax, not childish diction.
10) Keep a consistent literary-prose register; avoid colloquial phrasing.
11) Assume written prose as the output medium.
12) If carry-forward focus is provided, use it directly in this pass.

Return strict JSON with exactly these keys:
{{
  "scene_model": "...",
  "claim_map": ["...", "..."],
  "plain_restatement": "...",
  "constraint_ledger": {{
    "non_negotiables": ["...", "..."],
    "negotiables": ["...", "..."]
  }},
  "drafts": {{
    "source_close": "...",
    "plain_natural": "...",
    "balanced": "..."
  }},
  "translation": "...",
  "next_iteration_focus": "1-3 concrete improvements for the next pass"
}}
""".strip()
    return system, user


def dual_loop_selection_prompt(
    greek: str,
    paragraph_index: int,
    reference_translations: dict[str, str],
    user_preference: str,
    goals_guidance: str,
    iteration_logs: list[dict[str, Any]],
) -> tuple[str, str]:
    system = (
        "You are the final selector for a dual-loop translation pipeline. "
        "Output JSON only."
    )
    refs = reference_context_block(reference_translations)
    compact_logs: list[dict[str, Any]] = []
    for row in iteration_logs:
        tstep = row.get("translation_step", {})
        compact_logs.append(
            {
                "iteration": row.get("iteration"),
                "translation": row.get("translation", ""),
                "plain_restatement": tstep.get("plain_restatement", ""),
                "next_iteration_focus": tstep.get("next_iteration_focus", ""),
            }
        )
    payload = json.dumps(compact_logs, ensure_ascii=False, indent=2)
    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

{refs}

User preference prompt:
{user_preference}

Candidate iterations:
{payload}

Task:
1) Select the best candidate.
2) Rewrite once for final quality.
3) Prioritize user preference first, then:
{goals_guidance}
4) Preserve meaning relations and source stance while using natural modern prose.
5) Keep one paragraph; no added meaning.
6) Keep wording natural, readable, and stylistically coherent for the target audience.
7) Maintain literary prose register with clear plain syntax.

Return strict JSON with exactly these keys:
{{
  "selected_iteration": 1,
  "final_translation": "...",
  "selection_notes": "why this is best and what was refined"
}}
""".strip()
    return system, user


def run_dualloop_cognitive_pipeline(
    client: OpenAI,
    model: str,
    greek_paragraphs: list[str],
    iterations: int,
    verbose: bool,
    color_mode: str,
    user_preference: str,
    *,
    call_json_fn: Callable[..., dict[str, Any]],
    normalize_user_preference_fn: Callable[[str], str],
    should_use_color_fn: Callable[[str], bool],
    colorize_fn: Callable[[str, str | None, bool], str],
    stage_colors: dict[str, str],
    goals_guidance: str,
    dryden_paragraphs: list[str],
    perrin_paragraphs: list[str],
) -> dict[str, Any]:
    paragraphs: list[dict[str, Any]] = []
    color_enabled = should_use_color_fn(color_mode)
    normalized_preference = normalize_user_preference_fn(user_preference)
    vprint = make_vprint(
        verbose=verbose,
        color_enabled=color_enabled,
        colorize_fn=colorize_fn,
        stage_colors=stage_colors,
    )

    for idx, greek in enumerate(greek_paragraphs, start=1):
        reference_translations = reference_translations_for_index(
            dryden_paragraphs=dryden_paragraphs,
            perrin_paragraphs=perrin_paragraphs,
            paragraph_index=idx,
        )
        log_reference_inputs(
            vprint,
            paragraph_index=idx,
            pipeline_label="dual-loop cognitive pipeline",
            user_preference=normalized_preference,
            reference_translations=reference_translations,
        )

        current_translation = ""
        current_focus = ""
        iteration_logs: list[dict[str, Any]] = []

        for it in range(1, iterations + 1):
            vprint(f"[paragraph {idx}] [iter {it}] dual-loop translate...", "iteration")
            system, user = dual_loop_translate_prompt(
                greek=greek,
                paragraph_index=idx,
                reference_translations=reference_translations,
                user_preference=normalized_preference,
                goals_guidance=goals_guidance,
                iteration=it,
                previous_translation=current_translation or None,
                previous_focus=current_focus or None,
            )
            translation_result = call_json_fn(client, model, system, user, temperature=0.45)
            current_translation = str(translation_result.get("translation", "")).strip()
            current_focus = str(translation_result.get("next_iteration_focus", "")).strip()

            log_dualloop_iteration(
                vprint,
                paragraph_index=idx,
                iteration=it,
                translation_result=translation_result,
                translation=current_translation,
            )
            log_iteration_focus(
                vprint,
                paragraph_index=idx,
                iteration=it,
                focus_text=current_focus,
            )

            iteration_logs.append(
                {
                    "iteration": it,
                    "translation_step": translation_result,
                    "translation": current_translation,
                }
            )

        system, user = dual_loop_selection_prompt(
            greek=greek,
            paragraph_index=idx,
            reference_translations=reference_translations,
            user_preference=normalized_preference,
            goals_guidance=goals_guidance,
            iteration_logs=iteration_logs,
        )
        selection_result = call_json_fn(client, model, system, user, temperature=0.25)

        selected_iteration = int(selection_result.get("selected_iteration", iterations) or iterations)
        if not (1 <= selected_iteration <= len(iteration_logs)):
            selected_iteration = iterations
        final_translation = str(selection_result.get("final_translation", "")).strip() or current_translation
        selection_notes = str(selection_result.get("selection_notes", "")).strip()

        log_final_selection(
            vprint,
            paragraph_index=idx,
            translation_label="dual-loop",
            final_translation=final_translation,
            selected_iteration=selected_iteration,
            selection_notes=selection_notes,
        )

        paragraphs.append(
            {
                "paragraph_index": idx,
                "greek": greek,
                "reference_translations": reference_translations,
                "cognitive_iterations": iteration_logs,
                "final_agent_versions": {"cognitive_dualloop": final_translation},
                "final_synthesis": {
                    "final_translation": final_translation,
                    "selection_notes": selection_notes,
                    "selected_iteration": selected_iteration,
                },
            }
        )

    full_translation = "\n\n".join(
        p["final_synthesis"].get("final_translation", "").strip() for p in paragraphs
    ).strip()

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "cognitive_dualloop",
        "model": model,
        "iterations": iterations,
        "agent_count": 1,
        "user_preference": normalized_preference,
        "agents": [
            {
                "key": "cognitive_dualloop",
                "name": "Dual-Loop Cognitive Translator",
                "priority": "balanced",
            }
        ],
        "paragraph_count": len(greek_paragraphs),
        "paragraphs": paragraphs,
        "final_translation": full_translation,
    }

