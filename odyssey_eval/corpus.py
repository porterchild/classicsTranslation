"""Load the aligned passage pool and provide random passage selection."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

POOL_PATH = Path(__file__).parent.parent / "passages_pool.json"

# Keys each pool entry must have
REQUIRED_KEYS = {"book", "start_line", "end_line", "greek", "butler", "butcher_lang", "chapman"}
TRANSLATOR_KEYS = {"butler", "butcher_lang", "chapman"}


def load_pool(path: Path = POOL_PATH) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Passage pool not found at {path}. Run build_pool.py first."
        )
    with path.open(encoding="utf-8") as f:
        pool = json.load(f)
    if not isinstance(pool, list) or not pool:
        raise ValueError("Passage pool is empty or malformed.")
    valid = []
    for i, entry in enumerate(pool):
        missing = REQUIRED_KEYS - set(entry.keys())
        if missing:
            raise ValueError(f"Pool entry {i} missing keys: {missing}")
        # Skip entries where any translation is empty (extraction failed)
        if any(not entry.get(k, "").strip() for k in TRANSLATOR_KEYS):
            empty = [k for k in TRANSLATOR_KEYS if not entry.get(k, "").strip()]
            print(f"  [pool] skipping Od. {entry['book']}.{entry['start_line']}: empty {empty}")
            continue
        valid.append(entry)
    if not valid:
        raise ValueError("All pool entries have incomplete translations.")
    return valid


def sample_passages(
    pool: list[dict[str, Any]],
    n: int = 5,
    *,
    exclude_indices: set[int] | None = None,
    rng: random.Random | None = None,
) -> list[tuple[int, dict[str, Any]]]:
    """Return n (index, passage) tuples sampled without replacement.

    Excludes indices in exclude_indices to prevent repeating passages
    within a session.
    """
    rng = rng or random.Random()
    available = [
        (i, p) for i, p in enumerate(pool)
        if exclude_indices is None or i not in exclude_indices
    ]
    if len(available) < n:
        raise ValueError(
            f"Not enough passages in pool (have {len(available)}, need {n}). "
            "Add more passages to passages_pool.json."
        )
    return rng.sample(available, n)


def get_passage(pool: list[dict[str, Any]], book: int, start_line: int) -> dict[str, Any] | None:
    """Find a specific passage by book + start_line."""
    for p in pool:
        if p["book"] == book and p["start_line"] == start_line:
            return p
    return None


def passage_label(passage: dict[str, Any]) -> str:
    return f"Od. {passage['book']}.{passage['start_line']}-{passage['end_line']}"
