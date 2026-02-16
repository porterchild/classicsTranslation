#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from pipelines.cognitive_dualloop import run_dualloop_cognitive_pipeline
from pipelines.cognitive_user import run_user_cognitive_pipeline
from pipelines.debate import run_debate_pipeline
from pipelines.sequential import run_sequential_pipeline
from translation_feedback_mechanisms import compute_smoothness_feedback_from_perplexity

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


def get_api_key(dotenv_path: Path = Path(".env")) -> str | None:
    load_dotenv(dotenv_path)
    api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
    if api_key:
        return api_key
    fallback = (os.getenv("OPENAI_API_KEY") or "").strip()
    if fallback:
        return fallback
    return None


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


def run_pipeline(
    client: OpenAI,
    model: str,
    greek_paragraphs: list[str],
    iterations: int,
    verbose: bool,
    color_mode: str,
    user_preference: str,
    sequential_feedback_model: str | None,
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
            call_json_fn=call_json,
            normalize_user_preference_fn=normalize_user_preference,
            should_use_color_fn=should_use_color,
            colorize_fn=colorize,
            stage_colors=STAGE_COLORS,
            agent_colors=AGENT_COLORS,
            goals_guidance=GOALS_GUIDANCE,
            dryden_paragraphs=DEFAULT_DRYDEN_CLOUGH_PARAGRAPHS,
            perrin_paragraphs=DEFAULT_PERRIN_PARAGRAPHS,
        )
    if pipeline == "sequential":
        if sequential_feedback_model:
            preflight = compute_smoothness_feedback_from_perplexity(
                client=client,
                model=sequential_feedback_model,
                text="Perplexity preflight check sentence.",
                timeout=45,
            )
            if not preflight.get("available"):
                reason = str(preflight.get("reason", "unavailable")).strip()
                raise ValueError(
                    "Perplexity feedback preflight failed for "
                    f"model '{sequential_feedback_model}': {reason}"
                )
            if verbose:
                ppl = preflight.get("perplexity")
                tok = preflight.get("token_count")
                resolved = str(preflight.get("resolved_model", "")).strip()
                ppl_str = f"{float(ppl):.3f}" if isinstance(ppl, (int, float)) else "n/a"
                tok_str = str(tok) if isinstance(tok, int) else "n/a"
                resolved_note = f", resolved_model={resolved}" if resolved else ""
                print(
                    "[preflight] perplexity feedback ready: "
                    f"model={sequential_feedback_model}{resolved_note}, "
                    f"perplexity={ppl_str}, token_count={tok_str}",
                    file=sys.stderr,
                )
        return run_sequential_pipeline(
            client=client,
            model=model,
            greek_paragraphs=greek_paragraphs,
            iterations=iterations,
            verbose=verbose,
            color_mode=color_mode,
            user_preference=user_preference,
            call_json_fn=call_json,
            normalize_user_preference_fn=normalize_user_preference,
            should_use_color_fn=should_use_color,
            colorize_fn=colorize,
            stage_colors=STAGE_COLORS,
            agent_colors=AGENT_COLORS,
            goals_guidance=GOALS_GUIDANCE,
            dryden_paragraphs=DEFAULT_DRYDEN_CLOUGH_PARAGRAPHS,
            perrin_paragraphs=DEFAULT_PERRIN_PARAGRAPHS,
            feedback_model=sequential_feedback_model,
        )
    if pipeline == "cognitive_user":
        return run_user_cognitive_pipeline(
            client=client,
            model=model,
            greek_paragraphs=greek_paragraphs,
            iterations=iterations,
            verbose=verbose,
            color_mode=color_mode,
            user_preference=user_preference,
            call_json_fn=call_json,
            normalize_user_preference_fn=normalize_user_preference,
            should_use_color_fn=should_use_color,
            colorize_fn=colorize,
            stage_colors=STAGE_COLORS,
            goals_guidance=GOALS_GUIDANCE,
            dryden_paragraphs=DEFAULT_DRYDEN_CLOUGH_PARAGRAPHS,
            perrin_paragraphs=DEFAULT_PERRIN_PARAGRAPHS,
        )
    if pipeline == "cognitive_dualloop":
        return run_dualloop_cognitive_pipeline(
            client=client,
            model=model,
            greek_paragraphs=greek_paragraphs,
            iterations=iterations,
            verbose=verbose,
            color_mode=color_mode,
            user_preference=user_preference,
            call_json_fn=call_json,
            normalize_user_preference_fn=normalize_user_preference,
            should_use_color_fn=should_use_color,
            colorize_fn=colorize,
            stage_colors=STAGE_COLORS,
            goals_guidance=GOALS_GUIDANCE,
            dryden_paragraphs=DEFAULT_DRYDEN_CLOUGH_PARAGRAPHS,
            perrin_paragraphs=DEFAULT_PERRIN_PARAGRAPHS,
        )
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def render_markdown_report(result: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Translation Report")
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

    is_single_agent = int(result.get("agent_count", 0)) == 1
    if is_single_agent:
        lines.append("## Greek Source")
        lines.append("")
        for paragraph in result["paragraphs"]:
            idx = paragraph["paragraph_index"]
            lines.append(f"### Paragraph {idx}")
            lines.append("")
            lines.append(paragraph["greek"])
            lines.append("")
        return "\n".join(lines).strip() + "\n"

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
        choices=["debate", "sequential", "cognitive_user", "cognitive_dualloop"],
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
        default="runs/quorum_translation",
        help="Prefix for output file (.md).",
    )
    parser.add_argument(
        "--color",
        choices=["always", "auto", "never"],
        default="always",
        help="Colorize verbose stderr output.",
    )
    parser.set_defaults(verbose=True)
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        help="Print progress to stderr (default).",
    )
    verbosity_group.add_argument(
        "--quiet",
        dest="verbose",
        action="store_false",
        help="Disable progress logs.",
    )
    parser.add_argument(
        "--preference",
        default="",
        help="User preference prompt to prioritize while balancing faithfulness/readability/modernity.",
    )
    parser.add_argument(
        "--sequential-feedback-model",
        default="",
        help=(
            "Optional model for sequential perplexity feedback. "
            "Use 'local_model' for llama.cpp at localhost:8081, "
            "If unavailable, the run fails before translation starts."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    iterations = args.iterations
    if iterations is None:
        if args.pipeline in {"sequential", "cognitive_user", "cognitive_dualloop"}:
            iterations = DEFAULT_SEQUENTIAL_ITERATIONS
        else:
            iterations = DEFAULT_ITERATIONS
    if iterations < 1:
        print("--iterations must be >= 1", file=sys.stderr)
        return 2

    api_key = get_api_key(Path(".env"))
    if not api_key:
        print("Missing OPENROUTER_API_KEY (or OPENAI_API_KEY) in environment/.env.", file=sys.stderr)
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
    )

    try:
        result = run_pipeline(
            client=client,
            model=args.model,
            greek_paragraphs=DEFAULT_GREEK_PARAGRAPHS,
            iterations=iterations,
            verbose=args.verbose,
            color_mode=args.color,
            user_preference=args.preference,
            sequential_feedback_model=(args.sequential_feedback_model or "").strip() or None,
            pipeline=args.pipeline,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path = prefix.with_suffix(".md")

    md_path.write_text(render_markdown_report(result), encoding="utf-8")

    print(result["final_translation"])
    print()
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
