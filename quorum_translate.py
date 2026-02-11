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
FINALIZER_WEIGHTS = {
    "faithfulness": 0.20,
    "readability": 0.60,
    "modernity": 0.20,
}

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


def initial_translation_prompt(agent: Agent, greek: str, paragraph_index: int) -> tuple[str, str]:
    system = (
        "You are one member of a translation quorum translating Ancient Greek into English. "
        "You always output JSON only."
    )
    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

Task:
1) Translate this paragraph into modern English.
2) Keep all 3 goals in view: faithfulness, readability, modernity.
3) Your personal priority is: {agent.priority}.
4) Keep to one paragraph.

Return strict JSON with exactly these keys:
{{
  "translation": "...",
  "summary": "2-4 sentences on your key choices and tradeoffs",
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
) -> tuple[str, str]:
    system = (
        "You are a translator-debater in a quorum. "
        "Critique rigorously but constructively. Output JSON only."
    )
    payload = json.dumps(translations, ensure_ascii=False, indent=2)
    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

Current translations by agent:
{payload}

Debate iteration: {iteration}
Your personal priority: {agent.priority}

Assess every translation (including your own) on:
- faithfulness to Greek meaning
- readability in modern English
- modern tone

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
) -> tuple[str, str]:
    system = (
        "You are revising your translation after debate. "
        "Preserve meaning while improving according to your priority. Output JSON only."
    )
    translations_json = json.dumps(current_translations, ensure_ascii=False, indent=2)
    debates_json = json.dumps(debate_round, ensure_ascii=False, indent=2)
    user = f"""
Paragraph {paragraph_index} Greek:
{greek}

Your previous translation:
{own_previous}

Current translations:
{translations_json}

Debate outputs this round:
{debates_json}

Debate iteration: {iteration}
Your personal priority: {agent.priority}

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
    finalizer_weights: dict[str, float],
) -> tuple[str, str]:
    system = (
        "You are the final synthesis agent. "
        "Balance faithfulness, readability, and modernity equally. Output JSON only."
    )
    payload = {
        "paragraph_index": paragraph_index,
        "final_translations": final_translations,
        "agent_summaries": agent_summaries,
        "debate_summaries": debate_summaries,
    }
    user = f"""
Greek paragraph:
{greek}

Quorum context (JSON):
{json.dumps(payload, ensure_ascii=False, indent=2)}

Task:
- Produce one final translation for this paragraph.
- Use this priority weighting:
  - faithfulness: {finalizer_weights["faithfulness"]:.2f}
  - readability: {finalizer_weights["readability"]:.2f}
  - modernity: {finalizer_weights["modernity"]:.2f}
- In close tradeoffs, prefer readability/modernity slightly over strict literalism,
  while preserving core meaning, argument flow, and key imagery.
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
) -> dict[str, Any]:
    paragraphs: list[dict[str, Any]] = []

    def score_line(scores: Any) -> str:
        if not isinstance(scores, dict):
            return "n/a"
        f = scores.get("faithfulness", "n/a")
        r = scores.get("readability", "n/a")
        m = scores.get("modernity", "n/a")
        return f"faithfulness={f}, readability={r}, modernity={m}"

    for idx, greek in enumerate(greek_paragraphs, start=1):
        if verbose:
            print(f"[paragraph {idx}] initial translations...", file=sys.stderr)

        current: dict[str, str] = {}
        agent_logs: dict[str, Any] = {}

        def initial_task(agent: Agent) -> dict[str, Any]:
            system, user = initial_translation_prompt(agent, greek, idx)
            return call_json(client, model, system, user, temperature=0.45)

        initial_results = run_agent_tasks_parallel(AGENTS, initial_task)
        for agent in AGENTS:
            result = initial_results[agent.key]
            current[agent.key] = str(result.get("translation", "")).strip()
            agent_logs[agent.key] = {
                "priority": agent.priority,
                "initial": result,
                "debates": [],
                "revisions": [],
            }
            if verbose:
                print(f"[paragraph {idx}] [{agent.key}] initial translation:", file=sys.stderr)
                print(current[agent.key], file=sys.stderr)
                print(
                    f"[paragraph {idx}] [{agent.key}] initial summary: {result.get('summary', '')}",
                    file=sys.stderr,
                )
                print(
                    f"[paragraph {idx}] [{agent.key}] initial scores: "
                    f"{score_line(result.get('self_scores'))}",
                    file=sys.stderr,
                )

        debate_round_summaries: list[dict[str, Any]] = []

        for it in range(1, iterations + 1):
            if verbose:
                print(f"[paragraph {idx}] debate iteration {it}...", file=sys.stderr)

            def debate_task(agent: Agent) -> dict[str, Any]:
                system, user = debate_prompt(agent, greek, idx, current, it)
                return call_json(client, model, system, user, temperature=0.35)

            round_debates = run_agent_tasks_parallel(AGENTS, debate_task)
            for agent in AGENTS:
                debate = round_debates[agent.key]
                agent_logs[agent.key]["debates"].append(debate)
                if verbose:
                    print(
                        f"[paragraph {idx}] [iter {it}] [{agent.key}] debate summary: "
                        f"{debate.get('round_summary', '')}",
                        file=sys.stderr,
                    )
                    print(
                        f"[paragraph {idx}] [iter {it}] [{agent.key}] revision plan: "
                        f"{debate.get('self_revision_plan', '')}",
                        file=sys.stderr,
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
                            print(
                                f"[paragraph {idx}] [iter {it}] [{agent.key}] critique of [{target}] "
                                f"scores: {scores}",
                                file=sys.stderr,
                            )
                            print(
                                f"[paragraph {idx}] [iter {it}] [{agent.key}] critique strengths: {strengths}",
                                file=sys.stderr,
                            )
                            print(
                                f"[paragraph {idx}] [iter {it}] [{agent.key}] critique concerns: {concerns}",
                                file=sys.stderr,
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
                )
                return call_json(client, model, system, user, temperature=0.45)

            revision_results = run_agent_tasks_parallel(AGENTS, revision_task)
            for agent in AGENTS:
                revision = revision_results[agent.key]
                revised[agent.key] = str(revision.get("translation", "")).strip()
                agent_logs[agent.key]["revisions"].append(revision)
                if verbose:
                    print(
                        f"[paragraph {idx}] [iter {it}] [{agent.key}] revised translation:",
                        file=sys.stderr,
                    )
                    print(revised[agent.key], file=sys.stderr)
                    print(
                        f"[paragraph {idx}] [iter {it}] [{agent.key}] change summary: "
                        f"{revision.get('change_summary', '')}",
                        file=sys.stderr,
                    )
                    print(
                        f"[paragraph {idx}] [iter {it}] [{agent.key}] revised scores: "
                        f"{score_line(revision.get('self_scores'))}",
                        file=sys.stderr,
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

        if verbose:
            print(f"[paragraph {idx}] final synthesis...", file=sys.stderr)

        agent_summaries = {}
        for agent in AGENTS:
            key = agent.key
            agent_summaries[key] = {
                "priority": agent.priority,
                "initial_summary": agent_logs[key]["initial"].get("summary", ""),
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
            finalizer_weights=FINALIZER_WEIGHTS,
        )
        final_result = call_json(client, model, system, user, temperature=0.4)
        if verbose:
            print(f"[paragraph {idx}] final candidate agent versions:", file=sys.stderr)
            for agent in AGENTS:
                print(f"[paragraph {idx}] [{agent.key}] {current[agent.key]}", file=sys.stderr)
            print(f"[paragraph {idx}] final synthesis translation:", file=sys.stderr)
            print(str(final_result.get("final_translation", "")).strip(), file=sys.stderr)
            print(
                f"[paragraph {idx}] final synthesis justification: "
                f"{final_result.get('justification', '')}",
                file=sys.stderr,
            )
            print(
                f"[paragraph {idx}] final synthesis scores: "
                f"{score_line(final_result.get('balance_scores'))}",
                file=sys.stderr,
            )

        paragraphs.append(
            {
                "paragraph_index": idx,
                "greek": greek,
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
        "finalizer_weights": FINALIZER_WEIGHTS,
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
    lines.append(
        "- Finalizer weights: "
        f"`faithfulness={result['finalizer_weights']['faithfulness']:.2f}, "
        f"readability={result['finalizer_weights']['readability']:.2f}, "
        f"modernity={result['finalizer_weights']['modernity']:.2f}`"
    )
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
