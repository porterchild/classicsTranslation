from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import sys
from typing import Any, Callable

from openai import OpenAI

from .common import reference_context_block, reference_translations_for_index


@dataclass(frozen=True)
class Agent:
    key: str
    name: str
    priority: str


AGENTS = [
    Agent("faithful", "Faithfulness-First", "faithfulness"),
    Agent("readable", "Readability-First", "readability"),
    Agent("modern", "Modernity-First", "modernity"),
]


def run_agent_tasks_parallel(
    agents: list[Agent],
    task_fn: Callable[[Agent], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if not agents:
        return {}
    results: dict[str, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=len(agents)) as pool:
        future_to_agent = {pool.submit(task_fn, agent): agent for agent in agents}
        for future in as_completed(future_to_agent):
            agent = future_to_agent[future]
            results[agent.key] = future.result()
    return results


def initial_translation_prompt(
    agent: Agent,
    greek: str,
    paragraph_index: int,
    reference_translations: dict[str, str],
    user_preference: str,
    goals_guidance: str,
) -> tuple[str, str]:
    system = (
        "You are one member of a translation quorum translating Ancient Greek into English. "
        "You always output JSON only."
    )
    refs = reference_context_block(reference_translations)
    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

{refs}

User preference prompt:
{user_preference}

Task:
1) Before translating, write your observations as initial exploration/ideation.
   Focus on how you plan to handle difficult phrases, tone, and tradeoffs.
2) In those observations, compare multiple plausible wording/structure options before choosing one.
3) In observations, start with a concise true plain-language restatement as if explaining the meaning in everyday conversation.
   This restatement must not mirror source syntax or source-shaped abstractions.
4) Then add a source-close sketch and interpolate between the two to choose final phrasing.
5) Then translate this paragraph into modern English, using those observations.
6) Keep all 3 goals in view:
{goals_guidance}
7) Consider the user preference prompt while balancing the 3 goals above.
8) Your personal priority is: {agent.priority}.
9) Translate by meaning, not by Greek word order; recast syntax when needed so the English reads naturally.
10) Before finalizing, explore multiple plausible phrasings and choose the clearest natural wording that still preserves meaning.
11) Keep to one paragraph.
12) Keep the tone clear and dignified: simple modern English without cutesy wording, slang, or cartoonish substitutions.
13) Before moving past the restatement stage, self-check: if the restatement still sounds like translationese, rewrite it in plainer conversational English.

Return strict JSON with exactly these keys:
{{
  "observations": "pre-translation ideation and plan",
  "translation": "...",
  "self_scores": {{
    "faithfulness": 1-10,
    "readability": 1-10,
    "modernity": 1-10
  }}
}}
""".strip()
    return system, user


def debate_prompt(
    agent: Agent,
    greek: str,
    paragraph_index: int,
    translations: dict[str, str],
    iteration: int,
    reference_translations: dict[str, str],
    user_preference: str,
    goals_guidance: str,
) -> tuple[str, str]:
    system = (
        "You are a translator-debater in a quorum. "
        "Critique rigorously but constructively. Output JSON only."
    )
    payload = json.dumps(translations, ensure_ascii=False, indent=2)
    refs = reference_context_block(reference_translations)
    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

{refs}

User preference prompt:
{user_preference}

Current translations by agent:
{payload}

Debate iteration: {iteration}
Your personal priority: {agent.priority}

Assess every translation (including your own) using these goal definitions:
{goals_guidance}
Also assess how well each translation follows the user preference prompt.

Return strict JSON with exactly these keys:
{{
  "round_summary": "summary of strongest arguments",
  "critiques": [
    {{
      "agent": "faithful|readable|modern",
      "strengths": "...",
      "concerns": "...",
      "scores": {{
        "faithfulness": 1-10,
        "readability": 1-10,
        "modernity": 1-10
      }}
    }}
  ],
  "self_revision_plan": "concrete edits you will make next"
}}
""".strip()
    return system, user


def revision_prompt(
    agent: Agent,
    greek: str,
    paragraph_index: int,
    current_translations: dict[str, str],
    debate_round: dict[str, Any],
    own_previous: str,
    iteration: int,
    reference_translations: dict[str, str],
    user_preference: str,
    goals_guidance: str,
) -> tuple[str, str]:
    system = (
        "You are revising your translation after debate. "
        "Preserve meaning while improving according to your priority. Output JSON only."
    )
    translations_json = json.dumps(current_translations, ensure_ascii=False, indent=2)
    debates_json = json.dumps(debate_round, ensure_ascii=False, indent=2)
    refs = reference_context_block(reference_translations)
    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

{refs}

User preference prompt:
{user_preference}

Your previous translation:
{own_previous}

Current translations:
{translations_json}

Debate outputs this round:
{debates_json}

Debate iteration: {iteration}
Your personal priority: {agent.priority}

Revise using these goal definitions:
{goals_guidance}
Also satisfy the user preference prompt while balancing those goals.
Translate by meaning, not by Greek word order; recast syntax when needed so the English reads naturally.
Before finalizing, explore multiple plausible phrasings and choose the clearest natural wording that still preserves meaning.
Keep the tone clear and dignified: simple modern English without cutesy wording, slang, or cartoonish substitutions.

Return strict JSON with exactly these keys:
{{
  "translation": "...",
  "change_summary": "what changed and why",
  "self_scores": {{
    "faithfulness": 1-10,
    "readability": 1-10,
    "modernity": 1-10
  }}
}}
""".strip()
    return system, user


def final_synthesis_prompt(
    greek: str,
    paragraph_index: int,
    final_translations: dict[str, str],
    agent_summaries: dict[str, Any],
    debate_summaries: list[dict[str, Any]],
    reference_translations: dict[str, str],
    user_preference: str,
    goals_guidance: str,
) -> tuple[str, str]:
    system = (
        "You are the final synthesis agent. "
        "Prioritize the user preference prompt, then balance faithfulness, readability, and modernity. Output JSON only."
    )
    payload = {
        "paragraph_index": paragraph_index,
        "final_translations": final_translations,
        "agent_summaries": agent_summaries,
        "debate_summaries": debate_summaries,
        "reference_translations": reference_translations,
        "user_preference": user_preference,
    }
    refs = reference_context_block(reference_translations)
    user = f"""
Greek paragraph:
{greek}

{refs}

User preference prompt (highest priority):
{user_preference}

Quorum context (JSON):
{json.dumps(payload, ensure_ascii=False, indent=2)}

Task:
- Produce one final translation for this paragraph.
- Use these goal definitions:
{goals_guidance}
- Prioritize the user preference prompt above when tradeoffs conflict.
- Then balance the three goals while preserving core meaning, argument flow, and key imagery.
- Resolve disagreements using the debate summaries.
- Translate by meaning, not by Greek word order; recast syntax when needed so the English reads naturally.
- Before finalizing, explore multiple plausible phrasings and choose the clearest natural wording that still preserves meaning.
- Keep the tone clear and dignified: simple modern English without cutesy wording, slang, or cartoonish substitutions.

Return strict JSON with exactly these keys:
{{
  "final_translation": "...",
  "justification": "explanation of balance choices",
  "balance_scores": {{
    "faithfulness": 1-10,
    "readability": 1-10,
    "modernity": 1-10
  }}
}}
""".strip()
    return system, user


def run_debate_pipeline(
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
        vprint(f"[paragraph {idx}] initial translations...", stage="iteration")
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

        current: dict[str, str] = {}
        agent_logs: dict[str, Any] = {}

        def initial_task(agent: Agent) -> dict[str, Any]:
            system, user = initial_translation_prompt(
                agent,
                greek,
                idx,
                reference_translations,
                normalized_preference,
                goals_guidance,
            )
            return call_json_fn(client, model, system, user, temperature=0.45)

        initial_results = run_agent_tasks_parallel(AGENTS, initial_task)
        for agent in AGENTS:
            result = initial_results[agent.key]
            current[agent.key] = str(result.get("translation", "")).strip()
            observations = str(result.get("observations", "")).strip()
            agent_logs[agent.key] = {
                "priority": agent.priority,
                "initial": result,
                "debates": [],
                "revisions": [],
            }
            vprint(
                f"[paragraph {idx}] [{agent.key}] initial observations: {observations}",
                agent_key=agent.key,
            )
            vprint(f"[paragraph {idx}] [{agent.key}] initial translation:", agent_key=agent.key)
            vprint(current[agent.key], agent_key=agent.key)
            vprint(
                f"[paragraph {idx}] [{agent.key}] initial scores: "
                f"{score_line(result.get('self_scores'))}",
                agent_key=agent.key,
            )

        debate_round_summaries: list[dict[str, Any]] = []

        for it in range(1, iterations + 1):
            vprint(f"[paragraph {idx}] debate iteration {it}...", stage="iteration")

            def debate_task(agent: Agent) -> dict[str, Any]:
                system, user = debate_prompt(
                    agent,
                    greek,
                    idx,
                    current,
                    it,
                    reference_translations,
                    normalized_preference,
                    goals_guidance,
                )
                return call_json_fn(client, model, system, user, temperature=0.35)

            round_debates = run_agent_tasks_parallel(AGENTS, debate_task)
            for agent in AGENTS:
                debate = round_debates[agent.key]
                agent_logs[agent.key]["debates"].append(debate)
                vprint(
                    f"[paragraph {idx}] [iter {it}] [{agent.key}] debate summary: "
                    f"{debate.get('round_summary', '')}",
                    agent_key=agent.key,
                )
                vprint(
                    f"[paragraph {idx}] [iter {it}] [{agent.key}] revision plan: "
                    f"{debate.get('self_revision_plan', '')}",
                    agent_key=agent.key,
                )
                critiques = debate.get("critiques", [])
                if isinstance(critiques, list):
                    for critique in critiques:
                        if not isinstance(critique, dict):
                            continue
                        target = critique.get("agent", "unknown")
                        strengths = critique.get("strengths", "")
                        concerns = critique.get("concerns", "")
                        scores = score_line(critique.get("scores"))
                        vprint(
                            f"[paragraph {idx}] [iter {it}] [{agent.key}] critique of [{target}] "
                            f"scores: {scores}",
                            agent_key=agent.key,
                        )
                        vprint(
                            f"[paragraph {idx}] [iter {it}] [{agent.key}] critique strengths: {strengths}",
                            agent_key=agent.key,
                        )
                        vprint(
                            f"[paragraph {idx}] [iter {it}] [{agent.key}] critique concerns: {concerns}",
                            agent_key=agent.key,
                        )

            revised: dict[str, str] = {}

            def revision_task(agent: Agent) -> dict[str, Any]:
                system, user = revision_prompt(
                    agent=agent,
                    greek=greek,
                    paragraph_index=idx,
                    current_translations=current,
                    debate_round=round_debates,
                    own_previous=current[agent.key],
                    iteration=it,
                    reference_translations=reference_translations,
                    user_preference=normalized_preference,
                    goals_guidance=goals_guidance,
                )
                return call_json_fn(client, model, system, user, temperature=0.45)

            revision_results = run_agent_tasks_parallel(AGENTS, revision_task)
            for agent in AGENTS:
                revision = revision_results[agent.key]
                revised[agent.key] = str(revision.get("translation", "")).strip()
                agent_logs[agent.key]["revisions"].append(revision)
                vprint(
                    f"[paragraph {idx}] [iter {it}] [{agent.key}] revised translation:",
                    agent_key=agent.key,
                )
                vprint(revised[agent.key], agent_key=agent.key)
                vprint(
                    f"[paragraph {idx}] [iter {it}] [{agent.key}] change summary: "
                    f"{revision.get('change_summary', '')}",
                    agent_key=agent.key,
                )
                vprint(
                    f"[paragraph {idx}] [iter {it}] [{agent.key}] revised scores: "
                    f"{score_line(revision.get('self_scores'))}",
                    agent_key=agent.key,
                )

            debate_round_summaries.append(
                {
                    "iteration": it,
                    "summaries": {
                        agent_key: round_debates[agent_key].get("round_summary", "")
                        for agent_key in round_debates
                    },
                }
            )
            current = revised

        vprint(f"[paragraph {idx}] final synthesis...", stage="final")

        agent_summaries: dict[str, Any] = {}
        for agent in AGENTS:
            key = agent.key
            agent_summaries[key] = {
                "priority": agent.priority,
                "initial_observations": agent_logs[key]["initial"].get("observations", ""),
                "revision_summaries": [
                    rev.get("change_summary", "") for rev in agent_logs[key]["revisions"]
                ],
                "debate_plans": [
                    deb.get("self_revision_plan", "") for deb in agent_logs[key]["debates"]
                ],
            }

        system, user = final_synthesis_prompt(
            greek=greek,
            paragraph_index=idx,
            final_translations=current,
            agent_summaries=agent_summaries,
            debate_summaries=debate_round_summaries,
            reference_translations=reference_translations,
            user_preference=normalized_preference,
            goals_guidance=goals_guidance,
        )
        final_result = call_json_fn(client, model, system, user, temperature=0.4)
        vprint(f"[paragraph {idx}] final candidate agent versions:", stage="final")
        for agent in AGENTS:
            vprint(f"[paragraph {idx}] [{agent.key}] {current[agent.key]}", agent_key=agent.key)
        vprint(f"[paragraph {idx}] final synthesis translation:", stage="final")
        vprint(str(final_result.get("final_translation", "")).strip(), stage="final")
        vprint(
            f"[paragraph {idx}] final synthesis justification: "
            f"{final_result.get('justification', '')}",
            stage="final",
        )
        vprint(
            f"[paragraph {idx}] final synthesis scores: "
            f"{score_line(final_result.get('balance_scores'))}",
            stage="final",
        )

        paragraphs.append(
            {
                "paragraph_index": idx,
                "greek": greek,
                "reference_translations": reference_translations,
                "agents": agent_logs,
                "final_agent_versions": current,
                "debate_round_summaries": debate_round_summaries,
                "final_synthesis": final_result,
            }
        )

    full_translation = "\n\n".join(
        p["final_synthesis"].get("final_translation", "").strip() for p in paragraphs
    ).strip()

    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline": "debate",
        "model": model,
        "iterations": iterations,
        "agent_count": len(AGENTS),
        "user_preference": normalized_preference,
        "agents": [agent.__dict__ for agent in AGENTS],
        "paragraph_count": len(greek_paragraphs),
        "paragraphs": paragraphs,
        "final_translation": full_translation,
    }

