#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openai import OpenAI
import main

DEFAULT_PREFERENCE = (
    "This should be readable by a 7 year old. "
    "Use smooth plain clauses, and express abstract ideas in everyday language that fits the passage."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run focused sequential tuning on Theseus 1.3 (the third paragraph)."
    )
    parser.add_argument("--model", default=main.DEFAULT_MODEL)
    parser.add_argument(
        "--iterations",
        type=int,
        default=4,
        help="Number of sequential refinement iterations.",
    )
    parser.add_argument(
        "--preference",
        default=DEFAULT_PREFERENCE,
        help="User preference prompt to steer the translation.",
    )
    parser.add_argument(
        "--output-prefix",
        default="runs/theseus_paragraph3",
        help="Prefix for markdown output (.md).",
    )
    parser.add_argument(
        "--color",
        choices=["always", "auto", "never"],
        default="always",
        help="Colorize verbose stderr output.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose progress output.",
    )
    return parser.parse_args()


def run_cli() -> int:
    args = parse_args()
    if args.iterations < 1:
        print("--iterations must be >= 1", file=sys.stderr)
        return 2

    api_key = main.get_api_key(Path(".env"))
    if not api_key:
        print("Missing OPENROUTER_API_KEY (or OPENAI_API_KEY) in environment/.env.", file=sys.stderr)
        return 2

    client = OpenAI(api_key=api_key, base_url=main.OPENROUTER_BASE_URL)

    # Keep reference alignment with the selected Greek paragraph.
    main.DEFAULT_DRYDEN_CLOUGH_PARAGRAPHS = [main.DEFAULT_DRYDEN_CLOUGH_PARAGRAPHS[2]]
    main.DEFAULT_PERRIN_PARAGRAPHS = [main.DEFAULT_PERRIN_PARAGRAPHS[2]]

    try:
        result = main.run_pipeline(
            client=client,
            model=args.model,
            greek_paragraphs=[main.DEFAULT_GREEK_PARAGRAPHS[2]],
            iterations=args.iterations,
            verbose=not args.quiet,
            color_mode=args.color,
            user_preference=args.preference,
            sequential_feedback_model=None,
            pipeline="sequential",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    md_path = prefix.with_suffix(".md")
    md_path.write_text(main.render_markdown_report(result), encoding="utf-8")
    print(result["final_translation"])
    print()
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
