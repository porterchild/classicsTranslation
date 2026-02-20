"""Main evaluation loop.

Usage:
    python odyssey_eval/evaluate.py [--translator butler|butcher_lang|chapman|all] [--passages 5] [--verbose]

For each translator:
  - Randomly selects N passages from the pool
  - Runs the sequential pipeline on each with that translator's values
  - Scores each against the known translation via the comparison agent
  - Logs results to runs/odyssey_eval_<run_id>.json + .md
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI

from odyssey_eval.compare import compare
from odyssey_eval.corpus import load_pool, passage_label, sample_passages
from odyssey_eval.pipeline import run_passage
from odyssey_eval.profiles import PROFILES

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "x-ai/grok-4.1-fast"


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def evaluate_translator(
    *,
    client: OpenAI,
    translator_key: str,
    pool: list[dict],
    n_passages: int,
    model: str,
    pipeline_iterations: int,
    verbose: bool,
    used_indices: set[int],
    rng: random.Random,
) -> dict:
    profile = PROFILES[translator_key]
    translator_name = profile["name"]

    # Append actual sample passages from this translator so the LLM has
    # concrete style examples, not just a description.
    sample_block = ""
    samples = profile.get("sample_passages", [])
    if samples:
        parts = ["---\nACTUAL SAMPLE PASSAGES FROM THIS TRANSLATOR (study these closely):\n"]
        for s in samples:
            parts.append(f"Od. {s['book']}.{s['lines']}:\n{s['text']}\n")
        sample_block = "\n".join(parts)
    values_profile = profile["values_profile"] + "\n" + sample_block

    print(f"\n{'='*60}", flush=True)
    print(f"Evaluating: {translator_name}", flush=True)
    print(f"{'='*60}", flush=True)

    selected = sample_passages(pool, n=n_passages, exclude_indices=used_indices, rng=rng)
    for idx, _ in selected:
        used_indices.add(idx)

    passage_results = []
    scores = []

    for pool_idx, passage in selected:
        label = passage_label(passage)
        known_text = passage[translator_key]
        greek = passage["greek"]

        print(f"\n  Passage {label}", flush=True)
        if verbose:
            print(f"  Greek: {greek[:100]}...", flush=True)

        print(f"  Running pipeline...", flush=True)
        pipeline_out = run_passage(
            client=client,
            greek=greek,
            values_profile=values_profile,
            model=model,
            iterations=pipeline_iterations,
            verbose=verbose,
        )
        final_translation = pipeline_out["final_translation"]

        print(f"  Comparing against {translator_name}...", flush=True)
        comparison = compare(
            client=client,
            values_profile=values_profile,
            known_passage=known_text,
            pipeline_output=final_translation,
        )
        score = comparison.get("score", 0)
        scores.append(score)

        print(f"  Score: {score}/10", flush=True)
        print(f"  Rationale: {comparison.get('rationale', '')}", flush=True)
        if verbose:
            print(f"  Key gaps: {comparison.get('key_gaps', [])}", flush=True)
            print(f"  Key matches: {comparison.get('key_matches', [])}", flush=True)

        passage_results.append(
            {
                "passage_label": label,
                "book": passage["book"],
                "start_line": passage["start_line"],
                "end_line": passage["end_line"],
                "greek": greek,
                "known_translation": known_text,
                "pipeline_output": final_translation,
                "comparison": comparison,
                "score": score,
                "pipeline_detail": pipeline_out,
            }
        )

    avg_score = sum(scores) / len(scores) if scores else 0.0
    worst = min(passage_results, key=lambda r: r["score"])
    best = max(passage_results, key=lambda r: r["score"])

    print(f"\n  {translator_name} average score: {avg_score:.1f}/10", flush=True)
    print(f"  Best passage: {best['passage_label']} ({best['score']}/10)", flush=True)
    print(f"  Worst passage: {worst['passage_label']} ({worst['score']}/10)", flush=True)

    return {
        "translator": translator_key,
        "translator_name": translator_name,
        "n_passages": n_passages,
        "avg_score": round(avg_score, 2),
        "scores": scores,
        "best_passage": best["passage_label"],
        "worst_passage": worst["passage_label"],
        "passages": passage_results,
    }


def write_markdown(results: list[dict], run_id: str, model: str, pipeline_iterations: int) -> str:
    lines = [
        f"# Odyssey Evaluation Run: {run_id}",
        f"Model: {model} | Pipeline iterations: {pipeline_iterations}",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Score Summary",
        "",
        "| Translator | Avg Score | Best | Worst |",
        "|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['translator_name']} | {r['avg_score']:.1f}/10 "
            f"| {r['best_passage']} ({max(r['scores'])}/10) "
            f"| {r['worst_passage']} ({min(r['scores'])}/10) |"
        )
    lines.append("")

    for r in results:
        lines += [
            f"## {r['translator_name']}",
            "",
            f"Average: **{r['avg_score']:.1f}/10**",
            "",
        ]
        for p in r["passages"]:
            comp = p["comparison"]
            lines += [
                f"### {p['passage_label']} — Score: {p['score']}/10",
                "",
                "**Greek:**",
                f"> {p['greek']}",
                "",
                f"**Known translation ({r['translator_name']}):**",
                f"> {p['known_translation']}",
                "",
                "**Pipeline output:**",
                f"> {p['pipeline_output']}",
                "",
                f"**Rationale:** {comp.get('rationale', '')}",
                "",
            ]
            gaps = comp.get("key_gaps", [])
            matches = comp.get("key_matches", [])
            if gaps:
                lines.append(f"**Key gaps:** {'; '.join(gaps)}")
            if matches:
                lines.append(f"**Key matches:** {'; '.join(matches)}")
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Odyssey translation evaluation loop.")
    parser.add_argument(
        "--translator",
        default="all",
        choices=["butler", "butcher_lang", "chapman", "all"],
        help="Which translator to evaluate against (default: all)",
    )
    parser.add_argument("--passages", type=int, default=5, help="Passages per translator")
    parser.add_argument("--iterations", type=int, default=2, help="Pipeline iterations per passage")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="LLM model for pipeline and comparison")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    api_key = (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        sys.exit("Missing OPENROUTER_API_KEY")

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    pool = load_pool()
    print(f"Loaded passage pool: {len(pool)} passages", flush=True)

    translators = (
        ["butler", "butcher_lang", "chapman"]
        if args.translator == "all"
        else [args.translator]
    )

    rng = random.Random(args.seed)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    all_results = []
    for translator_key in translators:
        # Each translator samples independently — passages may overlap across
        # translators (evaluating different target styles on the same Greek is fine).
        result = evaluate_translator(
            client=client,
            translator_key=translator_key,
            pool=pool,
            n_passages=args.passages,
            model=args.model,
            pipeline_iterations=args.iterations,
            verbose=args.verbose,
            used_indices=set(),  # fresh per translator
            rng=rng,
        )
        all_results.append(result)

    # Write outputs
    runs_dir = ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)

    json_path = runs_dir / f"odyssey_eval_{run_id}.json"
    md_path = runs_dir / f"odyssey_eval_{run_id}.md"

    full_output = {
        "run_id": run_id,
        "model": args.model,
        "pipeline_iterations": args.iterations,
        "n_passages_per_translator": args.passages,
        "translators_evaluated": translators,
        "results": all_results,
    }
    json_path.write_text(json.dumps(full_output, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(
        write_markdown(all_results, run_id, args.model, args.iterations), encoding="utf-8"
    )

    print(f"\nResults written to:", flush=True)
    print(f"  {json_path}", flush=True)
    print(f"  {md_path}", flush=True)


if __name__ == "__main__":
    main()
