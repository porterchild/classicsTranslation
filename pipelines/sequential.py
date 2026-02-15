from __future__ import annotations

from datetime import datetime, timezone
import json
import sys
from typing import Any, Callable

from openai import OpenAI

from .common import reference_context_block, reference_translations_for_index


def distilled_judgment_guidance(previous_judgment: dict[str, Any]) -> str:
    scores = previous_judgment.get("scores", {})
    faith = scores.get("faithfulness") if isinstance(scores, dict) else None
    read = scores.get("readability") if isinstance(scores, dict) else None
    modern = scores.get("modernity") if isinstance(scores, dict) else None
    corpus = " ".join(
        [
            str(previous_judgment.get("overall_judgment", "")),
            str(previous_judgment.get("issues", "")),
            str(previous_judgment.get("revision_plan", "")),
        ]
    ).lower()

    lines: list[str] = []
    if isinstance(faith, (int, float)) and faith < 9:
        lines.append("- Keep all core meaning, relations, and contrasts from the Greek.")
    if isinstance(read, (int, float)) and read < 9:
        lines.append("- Use shorter, smoother plain clauses for easier reading.")
    if isinstance(modern, (int, float)) and modern < 9:
        lines.append("- Avoid archaic or bookish wording; keep modern plain prose.")

    if "source-shaped" in corpus or "calque" in corpus or "literal" in corpus:
        lines.append("- Rewrite source-shaped wording into natural English structure.")
    if "cutesy" in corpus or "slang" in corpus or "bookish" in corpus:
        lines.append("- Keep tone clear and dignified; avoid cutesy/slangy/bookish phrasing.")
    if "personif" in corpus:
        lines.append("- Avoid forced personification when plain process wording is clearer.")
    if "pronoun" in corpus:
        lines.append("- Avoid unclear pronoun chains; keep actor/action references explicit.")
    if "abstract" in corpus or "likely" in corpus or "probable" in corpus or "believable" in corpus:
        lines.append("- Prefer direct plain-language outcomes over abstract relation chains.")

    if not lines:
        lines = [
            "- Keep wording natural, plain, and audience-appropriate.",
            "- Preserve meaning and key contrasts while avoiding source-shaped phrasing.",
        ]
    return "\n".join(dict.fromkeys(lines))


def sequential_translate_prompt(
    greek: str,
    paragraph_index: int,
    reference_translations: dict[str, str],
    user_preference: str,
    goals_guidance: str,
    iteration: int,
    previous_translation: str | None,
    previous_judgment: dict[str, Any] | None,
) -> tuple[str, str]:
    system = (
        "You are a single translation agent iterating on one Ancient Greek paragraph. "
        "Output JSON only."
    )
    refs = reference_context_block(reference_translations)
    previous_translation_block = ""
    if previous_translation:
        previous_translation_block = (
            f"\nPrevious iteration translation:\n{previous_translation}\n"
        )
    previous_judgment_block = ""
    if previous_judgment:
        distilled = distilled_judgment_guidance(previous_judgment)
        previous_judgment_block = (
            "\nCarry-forward guidance from previous judgment:\n"
            f"{distilled}\n"
        )
    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

{refs}

User preference prompt:
{user_preference}

Iteration: {iteration}
{previous_translation_block}{previous_judgment_block}
Task:
1) Write observations first as a quick exploration pass. Include:
   a) a concise plain-language restatement in natural conversation for the target audience,
   b) a brief source-close sketch,
   c) a few candidate phrasings, including at least one that breaks source syntax.
2) Then produce one improved modern English translation for this paragraph (single paragraph).
3) Use prior judgment context (if provided) to fix weaknesses.
4) Prioritize the user preference prompt, then balance these goals:
{goals_guidance}
5) Preserve core meaning, key contrasts, and imagery. Do not add new meaning.
6) Translate by meaning, not Greek structure. Rewrite syntax freely for natural English.
7) Prefer clear, concrete, audience-appropriate wording. Keep a dignified tone (not cutesy, slangy, or bookish).
8) If faithfulness and readability conflict, keep core meaning/relations but favor natural readability for the stated audience.
9) Treat reference translations as semantic checks only, not style targets.
10) If figurative carryover sounds stiff, restate the same intent directly in natural prose.
11) Final wording should stay closer to your plain-language restatement than to your source-close sketch unless meaning would be lost.
12) Final self-check: if the result still sounds source-shaped, rewrite once more in plain natural English.
13) Do not copy phrasing from reference translations; paraphrase in fresh plain-English wording.

Return strict JSON with exactly these keys:
{{
  "observations": "brief planning notes for this iteration",
  "translation": "...",
  "self_scores": {{
    "faithfulness": 1-10,
    "readability": 1-10,
    "modernity": 1-10
  }}
}}
""".strip()
    return system, user


def sequential_judge_prompt(
    greek: str,
    paragraph_index: int,
    translation: str,
    reference_translations: dict[str, str],
    user_preference: str,
    goals_guidance: str,
    iteration: int,
) -> tuple[str, str]:
    system = (
        "You are a strict self-critic for a translation iteration. "
        "Judge against instructions and provide actionable revision feedback. Output JSON only."
    )
    refs = reference_context_block(reference_translations)
    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

{refs}

User preference prompt:
{user_preference}

Iteration: {iteration}
Translation to judge:
{translation}

Judge this translation on:
{goals_guidance}
Also judge with these priorities:
1) Prioritize the user preference prompt when scoring and planning revisions.
2) Prefer natural modern English over Greek-shaped phrasing, while preserving meaning.
3) Penalize literal calques, abstraction stacks, forced personification, unclear pronouns, and vocabulary above the target audience level.
4) Penalize cutesy/slangy/bookish phrasing when a plain dignified alternative preserves meaning.
5) Do not penalize merged wording when it preserves the same meaning and improves flow.
6) Do not reward simplification that drops any core relation or contrast from the source.
7) Reward direct intent-level phrasing when figurative carryover sounds stiff.
8) Treat reference translations as semantic checks only, not style targets.
9) Penalize mechanism-heavy translationese phrasing with abstract-noun chains when a direct outcome phrasing would preserve meaning.
10) Give concrete rewrite proposals, not generic feedback.
11) Penalize constructions that explain abstract relations indirectly when a direct plain-language outcome would be clearer.
12) Penalize lexical/style borrowing from reference translations when fresh plain wording would preserve meaning.

Return strict JSON with exactly these keys:
{{
  "overall_judgment": "candid quality assessment",
  "strengths": "concrete strengths",
  "issues": "concrete issues",
  "revision_plan": "concrete edits for next iteration",
  "scores": {{
    "faithfulness": 1-10,
    "readability": 1-10,
    "modernity": 1-10
  }}
}}
""".strip()
    return system, user


def sequential_selection_prompt(
    greek: str,
    paragraph_index: int,
    reference_translations: dict[str, str],
    user_preference: str,
    iteration_logs: list[dict[str, Any]],
) -> tuple[str, str]:
    system = (
        "You are the final selector for a sequential translation loop. "
        "Choose and lightly refine the best iteration result for the stated audience and preference. "
        "Output JSON only."
    )
    refs = reference_context_block(reference_translations)
    compact_iters: list[dict[str, Any]] = []
    for item in iteration_logs:
        translation_step = item.get("translation_step", {})
        judgment_step = item.get("judgment_step", {})
        compact_iters.append(
            {
                "iteration": item.get("iteration"),
                "translation": item.get("translation", ""),
                "self_scores": translation_step.get("self_scores", {}),
                "judge_scores": judgment_step.get("scores", {}),
                "judge_summary": judgment_step.get("overall_judgment", ""),
                "judge_issues": judgment_step.get("issues", ""),
                "judge_plan": judgment_step.get("revision_plan", ""),
            }
        )
    payload = json.dumps(compact_iters, ensure_ascii=False, indent=2)
    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

{refs}

User preference prompt:
{user_preference}

Candidate iterations:
{payload}

Task:
1) Select the strongest candidate for the user preference while preserving core source meaning.
2) You may rewrite the selected candidate, but keep one paragraph and do not add new meaning.
3) Prioritize user preference first, then balance faithfulness, readability, and modernity.
4) If faithfulness and readability conflict, preserve core meaning/relations and favor natural readability for the stated audience.
5) Prefer clear concrete wording over source-shaped abstraction, and avoid cutesy/slangy/bookish phrasing.
6) Prefer direct intent-level phrasing when figurative carryover sounds stiff.
7) Treat reference translations as semantic checks only; do not mirror their style.
8) Prefer outputs whose final style stays closer to plain-restatement phrasing than source-close sketch phrasing.
9) Treat candidate scores as hints only and judge the candidate text directly.

Return strict JSON with exactly these keys:
{{
  "selected_iteration": 1,
  "final_translation": "...",
  "justification": "why this is best for the stated preference",
  "balance_scores": {{
    "faithfulness": 1-10,
    "readability": 1-10,
    "modernity": 1-10
  }}
}}
""".strip()
    return system, user


def sequential_polish_prompt(
    greek: str,
    paragraph_index: int,
    reference_translations: dict[str, str],
    user_preference: str,
    selected_translation: str,
) -> tuple[str, str]:
    system = (
        "You are a final plain-English polisher for a translation pipeline. "
        "Rewrite once for natural readability while preserving meaning. Output JSON only."
    )
    user = f"""
Paragraph {paragraph_index} selected draft:

User preference prompt:
{user_preference}

Selected translation to polish:
{selected_translation}

Task:
1) Rewrite this translation once for final publication quality.
2) Keep one paragraph and preserve the same core meaning, relations, and contrasts from the selected draft.
3) Do not add new meaning or remove core meaning.
4) Prioritize natural readability for the target audience and keep tone clear and dignified.
5) Keep phrasing closer to plain natural English than source-shaped diction.
6) Assume written prose unless the user preference says spoken delivery; if an audience noun is needed, prefer readers over listeners.
7) Prefer direct plain verbs over metaphor-mechanics wording when meaning is unchanged.
8) Prefer reader-facing outcome phrasing over mechanism chains with abstract nouns.
9) If any phrase sounds translated, rewrite it into natural modern English while preserving meaning.
10) If a clause describes an abstract relation indirectly, rewrite it as a direct plain-language outcome.
11) You may substantially rephrase clause structure when needed for natural readability, as long as meaning is preserved.
12) If two nearby clauses express nearly the same idea, collapse them into one clear plain clause.
13) Keep clear domain terms from the selected draft when they are already understandable; avoid cute substitute nouns.
14) Prefer one simple plausibility statement over paired near-synonym checks when both say nearly the same thing.
15) Preserve the selected draft's wish/modality stance; do not shift it into a stronger personal assertion.

Return strict JSON with exactly these keys:
{{
  "polished_translation": "...",
  "polish_notes": "brief explanation of key changes",
  "balance_scores": {{
    "faithfulness": 1-10,
    "readability": 1-10,
    "modernity": 1-10
  }}
}}
""".strip()
    return system, user


def run_sequential_pipeline(
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
    agent_colors: dict[str, str],
    goals_guidance: str,
    dryden_paragraphs: list[str],
    perrin_paragraphs: list[str],
) -> dict[str, Any]:
    paragraphs: list[dict[str, Any]] = []
    color_enabled = should_use_color_fn(color_mode)
    normalized_preference = normalize_user_preference_fn(user_preference)

    def vprint(
        message: str,
        agent_key: str | None = None,
        stage: str | None = None,
    ) -> None:
        if not verbose:
            return
        color: str | None = None
        if agent_key:
            color = agent_colors.get(agent_key)
        elif stage:
            color = stage_colors.get(stage)
        print(colorize_fn(message, color, color_enabled), file=sys.stderr)

    def score_line(scores: Any) -> str:
        if not isinstance(scores, dict):
            return "n/a"
        f = scores.get("faithfulness", "n/a")
        r = scores.get("readability", "n/a")
        m = scores.get("modernity", "n/a")
        return f"faithfulness={f}, readability={r}, modernity={m}"

    for idx, greek in enumerate(greek_paragraphs, start=1):
        vprint(f"[paragraph {idx}] sequential iteration pipeline...", stage="iteration")
        vprint(
            f"[paragraph {idx}] user preference prompt: {normalized_preference}",
            stage="reference",
        )

        reference_translations = reference_translations_for_index(
            dryden_paragraphs=dryden_paragraphs,
            perrin_paragraphs=perrin_paragraphs,
            paragraph_index=idx,
        )

        vprint(
            f"[paragraph {idx}] reference input [dryden_clough]: "
            f"{reference_translations['dryden_clough']}",
            stage="reference",
        )
        vprint(
            f"[paragraph {idx}] reference input [perrin]: "
            f"{reference_translations['perrin']}",
            stage="reference",
        )

        current_translation = ""
        current_judgment: dict[str, Any] | None = None
        iteration_logs: list[dict[str, Any]] = []

        for it in range(1, iterations + 1):
            vprint(f"[paragraph {idx}] [iter {it}] translate...", stage="iteration")
            system, user = sequential_translate_prompt(
                greek=greek,
                paragraph_index=idx,
                reference_translations=reference_translations,
                user_preference=normalized_preference,
                goals_guidance=goals_guidance,
                iteration=it,
                previous_translation=current_translation or None,
                previous_judgment=current_judgment,
            )
            translation_result = call_json_fn(client, model, system, user, temperature=0.45)
            current_translation = str(translation_result.get("translation", "")).strip()
            observations = str(translation_result.get("observations", "")).strip()

            vprint(
                f"[paragraph {idx}] [iter {it}] [sequential] observations: {observations}",
                agent_key="modern",
            )
            vprint(f"[paragraph {idx}] [iter {it}] [sequential] translation:", agent_key="modern")
            vprint(current_translation, agent_key="modern")
            vprint(
                f"[paragraph {idx}] [iter {it}] [sequential] translation self-scores: "
                f"{score_line(translation_result.get('self_scores'))}",
                agent_key="modern",
            )

            vprint(f"[paragraph {idx}] [iter {it}] self-judge...", stage="iteration")
            system, user = sequential_judge_prompt(
                greek=greek,
                paragraph_index=idx,
                translation=current_translation,
                reference_translations=reference_translations,
                user_preference=normalized_preference,
                goals_guidance=goals_guidance,
                iteration=it,
            )
            current_judgment = call_json_fn(client, model, system, user, temperature=0.3)
            vprint(
                f"[paragraph {idx}] [iter {it}] [sequential] judgment: "
                f"{current_judgment.get('overall_judgment', '')}",
                agent_key="faithful",
            )
            vprint(
                f"[paragraph {idx}] [iter {it}] [sequential] strengths: "
                f"{current_judgment.get('strengths', '')}",
                agent_key="faithful",
            )
            vprint(
                f"[paragraph {idx}] [iter {it}] [sequential] issues: "
                f"{current_judgment.get('issues', '')}",
                agent_key="faithful",
            )
            vprint(
                f"[paragraph {idx}] [iter {it}] [sequential] revision plan: "
                f"{current_judgment.get('revision_plan', '')}",
                agent_key="faithful",
            )
            vprint(
                f"[paragraph {idx}] [iter {it}] [sequential] judgment scores: "
                f"{score_line(current_judgment.get('scores'))}",
                agent_key="faithful",
            )

            iteration_logs.append(
                {
                    "iteration": it,
                    "translation_step": translation_result,
                    "judgment_step": current_judgment,
                    "translation": current_translation,
                }
            )

        final_judgment = current_judgment or {}
        final_translation = current_translation
        selected_iteration = iterations
        system, user = sequential_selection_prompt(
            greek=greek,
            paragraph_index=idx,
            reference_translations=reference_translations,
            user_preference=normalized_preference,
            iteration_logs=iteration_logs,
        )
        selection_result = call_json_fn(client, model, system, user, temperature=0.25)
        selected_value = selection_result.get("selected_iteration", iterations)
        try:
            selected_iteration = int(selected_value)
        except (TypeError, ValueError):
            selected_iteration = iterations
        if not (1 <= selected_iteration <= len(iteration_logs)):
            selected_iteration = iterations
        selected_text = str(selection_result.get("final_translation", "")).strip()
        if selected_text:
            final_translation = selected_text
        selected_scores = selection_result.get("balance_scores")
        if isinstance(selected_scores, dict):
            final_judgment = {
                "overall_judgment": str(selection_result.get("justification", "")).strip(),
                "scores": selected_scores,
            }
        system, user = sequential_polish_prompt(
            greek=greek,
            paragraph_index=idx,
            reference_translations=reference_translations,
            user_preference=normalized_preference,
            selected_translation=final_translation,
        )
        polish_result = call_json_fn(client, model, system, user, temperature=0.55)
        polished_text = str(polish_result.get("polished_translation", "")).strip()
        if polished_text:
            final_translation = polished_text
        polish_scores = polish_result.get("balance_scores")
        if isinstance(polish_scores, dict):
            final_judgment = {
                "overall_judgment": str(polish_result.get("polish_notes", "")).strip(),
                "scores": polish_scores,
            }
        vprint(f"[paragraph {idx}] final sequential translation:", stage="final")
        vprint(final_translation, stage="final")
        vprint(
            f"[paragraph {idx}] selected iteration: {selected_iteration}",
            stage="final",
        )
        vprint(
            f"[paragraph {idx}] final polish notes: "
            f"{str(polish_result.get('polish_notes', '')).strip()}",
            stage="final",
        )
        vprint(
            f"[paragraph {idx}] final sequential judgment: "
            f"{final_judgment.get('overall_judgment', '')}",
            stage="final",
        )
        vprint(
            f"[paragraph {idx}] final sequential scores: "
            f"{score_line(final_judgment.get('scores'))}",
            stage="final",
        )

        paragraphs.append(
            {
                "paragraph_index": idx,
                "greek": greek,
                "reference_translations": reference_translations,
                "sequential_iterations": iteration_logs,
                "final_agent_versions": {"sequential": final_translation},
                "final_synthesis": {
                    "final_translation": final_translation,
                    "justification": str(final_judgment.get("overall_judgment", "")).strip(),
                    "balance_scores": final_judgment.get("scores", {}),
                    "selected_iteration": selected_iteration,
                    "polish": {
                        "notes": str(polish_result.get("polish_notes", "")).strip(),
                        "scores": polish_result.get("balance_scores", {}),
                    },
                },
            }
        )

    full_translation = "\n\n".join(
        p["final_synthesis"].get("final_translation", "").strip() for p in paragraphs
    ).strip()

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "sequential",
        "model": model,
        "iterations": iterations,
        "agent_count": 1,
        "user_preference": normalized_preference,
        "agents": [
            {
                "key": "sequential",
                "name": "Sequential Translator",
                "priority": "balanced",
            }
        ],
        "paragraph_count": len(greek_paragraphs),
        "paragraphs": paragraphs,
        "final_translation": full_translation,
    }

