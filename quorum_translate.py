#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

DEFAULT_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_ITERATIONS = 2
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_USER_PREFERENCE = "No additional user preference provided."
GOALS_GUIDANCE = (
    "- faithfulness: how strictly similar to the source language is it?\n"
    "- readability: how well does it flow, does it minimize convoluted sentences, "
    "does it make things easy to understand?\n"
    "- modernity: does it minimize archaic words and structure, does it feel like "
    "reading a modern article, book or post?"
)

# Plutarch, Theseus 1.1-1.3
DEFAULT_GREEK_PARAGRAPHS = [
    (
        "ὥσπερ ἐν ταῖς γεωγραφίαις, ὦ Σόσσιε Σενεκίων, οἱ ἱστορικοὶ τὰ διαφεύγοντα "
        "τὴν γνῶσιν αὐτῶν τοῖς ἐσχάτοις μέρεσι τῶν πινάκων πιεζοῦντες, αἰτίας "
        "παραγράφουσιν ὅτι \"τὰ δ᾽ ἐπέκεινα θῖνες ἄνυδροι καὶ θηριώδεις\" ἢ "
        "\"πηλὸς ἀϊδνὴς\" ἢ \"σκυθικὸν κρύος\" ἢ \"πέλαγος πεπηγός,\" οὕτως ἐμοὶ "
        "περὶ τὴν τῶν βίων τῶν παραλλήλων γραφήν, τὸν ἐφικτὸν εἰκότι λόγῳ καὶ "
        "βάσιμον ἱστορίᾳ πραγμάτων ἐχομένῃ χρόνον διελθόντι, περὶ τῶν ἀνωτέρω "
        "καλῶς εἶχεν εἰπεῖν· \"τὰ δ᾽ ἐπέκεινα τερατώδη καὶ τραγικὰ ποιηταὶ καὶ "
        "μυθογράφοι νέμονται, καὶ οὐκέτ᾽ ἔχει πίστιν οὐδὲ σαφήνειαν.\""
    ),
    (
        "ἐπεὶ δὲ τὸν περὶ Λυκούργου τοῦ νομοθέτου καὶ Νομᾶ τοῦ βασιλέως λόγον "
        "ἐκδόντες, ἐδοκοῦμεν οὐκ ἂν ἀλόγως τῷ Ῥωμύλῳ προσαναβῆναι, πλησίον τῶν "
        "χρόνων αὐτοῦ τῇ ἱστορίᾳ γεγονότες, σκοποῦντι δέ μοι τοιῷδε φωτί "
        "(κατ᾽ Αἰσχύλον) τίς ξυμβήσεται; τίν᾽ ἀντιτάξω τῷδε; τίς φερέγγυος; "
        "ἐφαίνετο τὸν τῶν καλῶν καὶ ἀοιδίμων οἰκιστὴν Ἀθηνῶν ἀντιστῆσαι καὶ "
        "παραβαλεῖν τῷ πατρὶ τῆς ἀνικήτου καὶ μεγαλοδόξου Ῥώμης,"
    ),
    (
        "εἴη μὲν οὖν ἡμῖν ἐκκαθαιρόμενον λόγῳ τὸ μυθῶδες ὑπακοῦσαι καὶ λαβεῖν "
        "ἱστορίας ὄψιν, ὅπου δ᾽ ἂν αὐθαδῶς τοῦ πιθανοῦ περιφρονῇ καὶ μὴ δέχηται "
        "τὴν πρὸς τὸ εἰκὸς μῖξιν, εὐγνωμόνων ἀκροατῶν δεησόμεθα καὶ πρᾴως τὴν "
        "ἀρχαιολογίαν προσδεχομένων."
    ),
]

# Dryden/Clough (1859), aligned to Theseus 1.1-1.3
DEFAULT_DRYDEN_CLOUGH_PARAGRAPHS = [
    (
        "As geographers, Sosius Senecio, crowd into the edges of their maps parts of the world "
        "which escape their knowledge, adding notes, in the margin, to the effect, that beyond "
        "this lie sandy deserts full of wild beasts, unapproachable bogs, Scythian ice, and "
        "frozen sea, so, in the chart of my lives, from those periods which probable reasoning "
        "can reach to, and where the history of facts can find firm footing, in passing those "
        "remote ages which are accessible only to conjecture, I might well say, Beyond this there "
        "is nothing but prodigies and fictions, the only inhabitants are the poets and inventors "
        "of fables, there is no other certainty, or reality."
    ),
    (
        "After I had published my account of Lycurgus the lawgiver and Numa the king, it seemed "
        "to me not unreasonable if, now that my history had brought me down to Romulus, I should "
        "pass in review and compare with him, as it were, the man who gave Athens its beautiful "
        "and famous city."
    ),
    (
        "May I therefore succeed in purifying fable, making it submit to reason so as to assume "
        "the face of history. Where it cannot be reduced to any probable likeness, and refuses to "
        "admit any element of the possible, I shall beg my readers to be indulgent to antiquity in "
        "its records."
    ),
]

# Bernadotte Perrin (1914), aligned to Theseus 1.1-1.3
DEFAULT_PERRIN_PARAGRAPHS = [
    (
        "Just as geographers, O Sossius Senecio, crowd on to the outer edges of their maps the "
        "parts of the earth which elude their knowledge, with explanatory notes that \"What lies "
        "beyond is sandy desert without water and full of wild beasts,\" or \"blind marsh,\" or "
        "\"Scythian cold,\" or \"frozen sea,\" so in the writing of my Parallel Lives, now that I "
        "have traversed those periods of time which are accessible to probable reasoning and which "
        "afford basis for a history dealing with facts, I might well say of the earlier periods: "
        "\"What lies beyond is full of marvels and unreality, a land of poets and fabulists, of "
        "doubt and obscurity.\""
    ),
    (
        "But after publishing my account of Lycurgus the lawgiver and Numa the king, I thought I "
        "might not unreasonably go back still farther to Romulus, now that my history had brought "
        "me near his times. And as I asked myself, \"With such a warrior\" (as Aeschylus says) "
        "\"who will dare to fight?\" \"Whom shall I set against him? Who is competent?\" it seemed "
        "to me that I must make the founder of lovely and famous Athens the counterpart and "
        "parallel to the father of invincible and glorious Rome."
    ),
    (
        "May I therefore succeed in purifying Fable, making her submit to reason and take on the "
        "semblance of History. But where she obstinately disdains to make herself credible, and "
        "refuses to admit any element of probability, I shall pray for kindly readers, and such as "
        "receive with indulgence the tales of antiquity."
    ),
]


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

ANSI_RESET = "\033[0m"
AGENT_COLORS = {
    "faithful": "\033[91m",
    "readable": "\033[92m",
    "modern": "\033[96m",
}
STAGE_COLORS = {
    "reference": "\033[93m",
    "iteration": "\033[94m",
    "final": "\033[95m",
}


def should_use_color(color_mode: str) -> bool:
    if color_mode == "always":
        return True
    if color_mode == "never":
        return False
    return sys.stderr.isatty()


def colorize(text: str, color: str | None, enabled: bool) -> str:
    if not enabled or not color:
        return text
    return f"{color}{text}{ANSI_RESET}"


def normalize_user_preference(preference: str) -> str:
    cleaned = preference.strip()
    if cleaned:
        return cleaned
    return DEFAULT_USER_PREFERENCE


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


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def parse_json_object(text: str) -> dict[str, Any]:
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


def call_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.5,
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
            )
            content = resp.choices[0].message.content or ""
            return parse_json_object(content)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
            else:
                break
    assert last_error is not None
    raise last_error


def reference_context_block(reference_translations: dict[str, str]) -> str:
    dryden = reference_translations.get("dryden_clough", "").strip()
    perrin = reference_translations.get("perrin", "").strip()
    return (
        "Reference translations for context:\n"
        f"- Dryden/Clough: {dryden}\n"
        f"- Perrin: {perrin}"
    )


def initial_translation_prompt(
    agent: Agent,
    greek: str,
    paragraph_index: int,
    reference_translations: dict[str, str],
    user_preference: str,
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
2) Then translate this paragraph into modern English, using those observations.
3) Keep all 3 goals in view:
{GOALS_GUIDANCE}
4) Consider the user preference prompt while balancing the 3 goals above.
5) Your personal priority is: {agent.priority}.
6) Keep to one paragraph.

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
{GOALS_GUIDANCE}
Also assess how well each translation follows the user preference prompt.

Return strict JSON with exactly these keys:
{{
  "round_summary": "3-6 sentences summarizing best arguments",
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
  "self_revision_plan": "2-4 concrete edits you will make next"
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
{GOALS_GUIDANCE}
Also satisfy the user preference prompt while balancing those goals.

Return strict JSON with exactly these keys:
{{
  "translation": "...",
  "change_summary": "2-4 sentences on what changed and why",
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
{GOALS_GUIDANCE}
- Prioritize the user preference prompt above when tradeoffs conflict.
- Then balance the three goals while preserving core meaning, argument flow, and key imagery.
- Resolve disagreements using the debate summaries.

Return strict JSON with exactly these keys:
{{
  "final_translation": "...",
  "justification": "3-6 sentences explaining balance choices",
  "balance_scores": {{
    "faithfulness": 1-10,
    "readability": 1-10,
    "modernity": 1-10
  }}
}}
""".strip()
    return system, user


def run_quorum(
    client: OpenAI,
    model: str,
    greek_paragraphs: list[str],
    iterations: int,
    verbose: bool,
    color_mode: str,
    user_preference: str,
) -> dict[str, Any]:
    paragraphs: list[dict[str, Any]] = []
    color_enabled = should_use_color(color_mode)
    normalized_preference = normalize_user_preference(user_preference)

    def vprint(
        message: str,
        agent_key: str | None = None,
        stage: str | None = None,
    ) -> None:
        if not verbose:
            return
        color: str | None = None
        if agent_key:
            color = AGENT_COLORS.get(agent_key)
        elif stage:
            color = STAGE_COLORS.get(stage)
        print(colorize(message, color, color_enabled), file=sys.stderr)

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

        reference_translations = {
            "dryden_clough": (
                DEFAULT_DRYDEN_CLOUGH_PARAGRAPHS[idx - 1]
                if idx - 1 < len(DEFAULT_DRYDEN_CLOUGH_PARAGRAPHS)
                else ""
            ),
            "perrin": (
                DEFAULT_PERRIN_PARAGRAPHS[idx - 1] if idx - 1 < len(DEFAULT_PERRIN_PARAGRAPHS) else ""
            ),
        }

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
            )
            return call_json(client, model, system, user, temperature=0.45)

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
                )
                return call_json(client, model, system, user, temperature=0.35)

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
                )
                return call_json(client, model, system, user, temperature=0.45)

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

        agent_summaries = {}
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
        )
        final_result = call_json(client, model, system, user, temperature=0.4)
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
        "model": model,
        "iterations": iterations,
        "agent_count": len(AGENTS),
        "user_preference": normalized_preference,
        "agents": [agent.__dict__ for agent in AGENTS],
        "paragraph_count": len(greek_paragraphs),
        "paragraphs": paragraphs,
        "final_translation": full_translation,
    }


def render_markdown_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Quorum Translation Report")
    lines.append("")
    lines.append(f"- Model: `{result['model']}`")
    lines.append(f"- Translators/Debaters: `{result['agent_count']}`")
    lines.append(f"- Debate iterations: `{result['iterations']}`")
    lines.append(f"- User preference prompt: `{result['user_preference']}`")
    lines.append(f"- Generated (UTC): `{result['created_at_utc']}`")
    lines.append("")
    lines.append("## Final Translation")
    lines.append("")
    lines.append(result["final_translation"])
    lines.append("")

    for paragraph in result["paragraphs"]:
        idx = paragraph["paragraph_index"]
        lines.append(f"## Paragraph {idx}")
        lines.append("")
        lines.append("### Greek")
        lines.append("")
        lines.append(paragraph["greek"])
        lines.append("")
        lines.append("### Final Synthesis")
        lines.append("")
        lines.append(paragraph["final_synthesis"].get("final_translation", ""))
        lines.append("")
        lines.append("### Agent Final Versions")
        lines.append("")
        for key, text in paragraph["final_agent_versions"].items():
            lines.append(f"- `{key}`: {text}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-agent quorum translation workflow (translate -> debate -> revise -> synthesize)."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS)
    parser.add_argument(
        "--output-prefix",
        default="quorum_translation",
        help="Prefix for output files (.json and .md).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress to stderr.",
    )
    parser.add_argument(
        "--color",
        choices=["always", "auto", "never"],
        default="always",
        help="Colorize verbose stderr output.",
    )
    parser.add_argument(
        "--preference",
        default="",
        help="User preference prompt to prioritize while balancing faithfulness/readability/modernity.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.iterations < 1:
        print("--iterations must be >= 1", file=sys.stderr)
        return 2

    load_dotenv(Path(".env"))
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing OPENROUTER_API_KEY (or OPENAI_API_KEY) in environment/.env.", file=sys.stderr)
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )

    result = run_quorum(
        client=client,
        model=args.model,
        greek_paragraphs=DEFAULT_GREEK_PARAGRAPHS,
        iterations=args.iterations,
        verbose=args.verbose,
        color_mode=args.color,
        user_preference=args.preference,
    )

    prefix = Path(args.output_prefix)
    json_path = prefix.with_suffix(".json")
    md_path = prefix.with_suffix(".md")

    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown_report(result), encoding="utf-8")

    print(result["final_translation"])
    print()
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
