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
DEFAULT_SEQUENTIAL_ITERATIONS = 3
DEFAULT_PIPELINE = "debate"
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
                timeout=120,
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
2) In those observations, compare multiple plausible wording/structure options before choosing one.
3) In observations, start with a concise true plain-language restatement as if explaining the meaning in everyday conversation.
   This restatement must not mirror source syntax or source-shaped abstractions.
4) Then add a source-close sketch and interpolate between the two to choose final phrasing.
5) Then translate this paragraph into modern English, using those observations.
6) Keep all 3 goals in view:
{GOALS_GUIDANCE}
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


def sequential_translate_prompt(
    greek: str,
    paragraph_index: int,
    reference_translations: dict[str, str],
    user_preference: str,
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
        previous_judgment_block = (
            "\nPrevious iteration self-judgment JSON:\n"
            f"{json.dumps(previous_judgment, ensure_ascii=False, indent=2)}\n"
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
{GOALS_GUIDANCE}
5) Preserve core meaning, key contrasts, and imagery. Do not add new meaning.
6) Translate by meaning, not Greek structure. Rewrite syntax freely for natural English.
7) Prefer clear, concrete, audience-appropriate wording. Keep a dignified tone (not cutesy, slangy, or bookish).
8) If faithfulness and readability conflict, keep core meaning/relations but favor natural readability for the stated audience.
9) Treat reference translations as semantic checks only, not style targets.
10) If figurative carryover sounds stiff, restate the same intent directly in natural prose.
11) Final wording should stay closer to your plain-language restatement than to your source-close sketch unless meaning would be lost.
12) Final self-check: if the result still sounds source-shaped, rewrite once more in plain natural English.

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
{GOALS_GUIDANCE}
Also judge with these priorities:
1) Prioritize the user preference prompt when scoring and planning revisions.
2) Prefer natural modern English over Greek-shaped phrasing, while preserving meaning.
3) Penalize literal calques, abstraction stacks, forced personification, unclear pronouns, and vocabulary above the target audience level.
4) Penalize cutesy/slangy/bookish phrasing when a plain dignified alternative preserves meaning.
5) Do not penalize merged wording when it preserves the same meaning and improves flow.
6) Do not reward simplification that drops any core relation or contrast from the source.
7) Reward direct intent-level phrasing when figurative carryover sounds stiff.
8) Treat reference translations as semantic checks only, not style targets.
9) Give concrete rewrite proposals, not generic feedback.

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
    refs = reference_context_block(reference_translations)
    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

{refs}

User preference prompt:
{user_preference}

Selected translation to polish:
{selected_translation}

Task:
1) Rewrite this translation once for final publication quality.
2) Keep one paragraph and preserve the same core meaning, relations, and contrasts.
3) Do not add new meaning or remove core meaning.
4) Prioritize natural readability for the target audience and keep tone clear and dignified.
5) Treat reference translations as semantic checks only, not style targets.
6) Keep phrasing closer to plain natural English than source-shaped diction.
7) Assume written prose unless the user preference says spoken delivery; if an audience noun is needed, prefer readers over listeners.
8) Prefer direct plain verbs over metaphor-mechanics wording when meaning is unchanged.
9) If any phrase sounds translated, rewrite it into natural modern English while preserving meaning.

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


def run_debate_pipeline(
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


def run_sequential_pipeline(
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
        vprint(f"[paragraph {idx}] sequential iteration pipeline...", stage="iteration")
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
                iteration=it,
                previous_translation=current_translation or None,
                previous_judgment=current_judgment,
            )
            translation_result = call_json(client, model, system, user, temperature=0.45)
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
                iteration=it,
            )
            current_judgment = call_json(client, model, system, user, temperature=0.3)
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
        selection_result = call_json(client, model, system, user, temperature=0.25)
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
        polish_result = call_json(client, model, system, user, temperature=0.2)
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


def run_pipeline(
    client: OpenAI,
    model: str,
    greek_paragraphs: list[str],
    iterations: int,
    verbose: bool,
    color_mode: str,
    user_preference: str,
    pipeline: str,
) -> dict[str, Any]:
    if pipeline == "debate":
        return run_debate_pipeline(
            client=client,
            model=model,
            greek_paragraphs=greek_paragraphs,
            iterations=iterations,
            verbose=verbose,
            color_mode=color_mode,
            user_preference=user_preference,
        )
    if pipeline == "sequential":
        return run_sequential_pipeline(
            client=client,
            model=model,
            greek_paragraphs=greek_paragraphs,
            iterations=iterations,
            verbose=verbose,
            color_mode=color_mode,
            user_preference=user_preference,
        )
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def render_markdown_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Quorum Translation Report")
    lines.append("")
    lines.append(f"- Pipeline: `{result.get('pipeline', 'debate')}`")
    lines.append(f"- Model: `{result['model']}`")
    lines.append(f"- Translators/Debaters: `{result['agent_count']}`")
    lines.append(f"- Iterations: `{result['iterations']}`")
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
            "Run a modular Greek translation workflow with swappable pipelines."
        )
    )
    parser.add_argument(
        "--pipeline",
        choices=["debate", "sequential"],
        default=DEFAULT_PIPELINE,
        help="Translation pipeline to run.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help=(
            "Iteration count. Defaults: debate=2, sequential=3."
        ),
    )
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
    iterations = args.iterations
    if iterations is None:
        if args.pipeline == "sequential":
            iterations = DEFAULT_SEQUENTIAL_ITERATIONS
        else:
            iterations = DEFAULT_ITERATIONS
    if iterations < 1:
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

    result = run_pipeline(
        client=client,
        model=args.model,
        greek_paragraphs=DEFAULT_GREEK_PARAGRAPHS,
        iterations=iterations,
        verbose=args.verbose,
        color_mode=args.color,
        user_preference=args.preference,
        pipeline=args.pipeline,
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
