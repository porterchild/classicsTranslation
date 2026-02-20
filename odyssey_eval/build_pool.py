"""One-time script to build the aligned passage pool (passages_pool.json).

Downloads the three English translations from Project Gutenberg, fetches
Greek passages from Perseus, then uses an LLM to extract matching English
passages for each Greek selection.

Usage:
    python odyssey_eval/build_pool.py [--passages-per-book 2] [--books 1,5,9,11,12,17,22,23]

Output:
    passages_pool.json in the project root
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "x-ai/grok-4.1-fast"

# Project Gutenberg plain-text URLs
GUTENBERG_URLS = {
    "butler": "https://www.gutenberg.org/cache/epub/1727/pg1727.txt",
    "butcher_lang": "https://www.gutenberg.org/cache/epub/3160/pg3160.txt",
    "chapman": "https://www.gutenberg.org/cache/epub/48895/pg48895.txt",
}

# Books to sample from and approx line counts (for bounds checking)
BOOK_SIZES = {
    1: 444, 2: 434, 3: 497, 4: 847, 5: 493,
    6: 331, 7: 347, 8: 586, 9: 566, 10: 574,
    11: 640, 12: 453, 13: 440, 14: 533, 15: 557,
    16: 481, 17: 606, 18: 428, 19: 604, 20: 394,
    21: 434, 22: 501, 23: 372, 24: 548,
}

# Curated passage starts: (book, start_line, end_line, description)
# Chosen for variety of content, character, and narrative register
PASSAGE_SPECS = [
    (1,   1,  21, "Invocation and introduction of Odysseus's plight"),
    (1,  32,  57, "Zeus speaks of Aegisthus, the gods discuss fate"),
    (1,  96, 124, "Athena appears to Telemachus as Mentes"),
    (5,   1,  27, "Dawn rises; gods convene; Hermes sent to Calypso"),
    (5,  43,  75, "Hermes finds Calypso; she receives him"),
    (5, 151, 179, "Calypso releases Odysseus; he weeps on the shore"),
    (6,  85, 114, "Nausicaa and her maids at the river; Odysseus wakes"),
    (9,   1,  38, "Odysseus identifies himself to the Phaeacians"),
    (9, 105, 135, "The Cyclops's cave; Polyphemus returns"),
    (9, 375, 414, "The blinding of Polyphemus"),
    (11,  1,  33, "Odysseus reaches the Underworld; the ritual"),
    (11, 140, 179, "Odysseus meets his mother Anticleia"),
    (12,  1,  35, "Circe advises Odysseus on the Sirens and Scylla"),
    (12, 165, 200, "The Sirens passage"),
    (17, 290, 327, "Argos the dog recognizes Odysseus and dies"),
    (19, 96, 130, "Penelope interrogates the disguised Odysseus"),
    (21, 400, 434, "Odysseus strings the bow"),
    (22,  1,  33, "The slaughter of the suitors begins"),
    (23,  1,  40, "Eurynome wakes Penelope; Odysseus has returned"),
    (23, 166, 206, "Penelope tests Odysseus with the bed"),
    (24, 130, 168, "Odysseus reveals himself to his father Laertes"),
]


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _fetch_url(url: str, timeout: int = 30) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (research)"})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def _call_llm(client: OpenAI, model: str, system: str, user: str,
               temperature: float = 0.2, retries: int = 3) -> str:
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                timeout=120,
                extra_body={"reasoning": {"enabled": True}},
            )
            return resp.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            if attempt < retries:
                time.sleep(1.5 * attempt)
    assert last_err is not None
    raise last_err


def _call_json(client: OpenAI, model: str, system: str, user: str,
               temperature: float = 0.2) -> dict:
    text = _call_llm(client, model, system, user, temperature=temperature)
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        v = json.loads(raw)
        if isinstance(v, dict):
            return v
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if m:
        try:
            v = json.loads(m.group(0))
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            pass
    raise ValueError(f"No JSON in response:\n{text[:300]}")


def _strip_html_tags(html: str) -> str:
    """Remove HTML tags and decode common entities."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("&#39;", "'").replace("&quot;", '"')
    text = re.sub(r"\s+", " ", text).strip()
    return text


def fetch_greek_passage_from_perseus(book: int, start: int, end: int) -> str:
    """Fetch Greek lines from Perseus HTML using linenumber span markers.

    Perseus serves ~40 lines per page chunk. Each chunk has markers like:
        <span class="linenumber"><span class="english">N</span></span>
    every 5 lines. We fetch both the page for 'start' and 'end' (which may
    differ), combine all segments, and return those that overlap [start, end].
    """
    base = "https://www.perseus.tufts.edu/hopper/text"
    # Split pattern: <span class="linenumber">...<span class="english">N</span>...
    marker_pat = re.compile(
        r'<span[^>]*class="linenumber"[^>]*>.*?<span[^>]*class="english"[^>]*>(\d+)</span>',
        re.IGNORECASE | re.DOTALL,
    )

    print(f"    Fetching Greek from Perseus: Od. {book}.{start}-{end}...", flush=True)

    all_segments: dict[int, str] = {}  # marker_line_num -> stripped text

    # Fetch pages covering start and end (they may be different Perseus chunks)
    query_lines = sorted({start, end})
    for query_line in query_lines:
        url = (
            f"{base}?doc=Perseus%3Atext%3A1999.01.0135%3Abook%3D{book}"
            f"%3Aline%3D{query_line}&lang=original"
        )
        try:
            html = _fetch_url(url, timeout=20)
        except Exception as exc:
            print(f"    WARNING: Perseus fetch failed for line {query_line}: {exc}", flush=True)
            continue

        # Split the HTML at each line-number marker
        parts = marker_pat.split(html)
        # parts = [pre-first-marker-text, line_num1, text1, line_num2, text2, ...]
        i = 1
        while i + 1 < len(parts):
            try:
                ln = int(parts[i])
                if 1 <= ln <= 700 and ln not in all_segments:
                    seg_raw = _strip_html_tags(parts[i + 1])
                    # Keep only Greek characters and basic punctuation
                    seg = re.sub(
                        r"[^\u0370-\u03FF\u1F00-\u1FFF\u0300-\u036F\s,·'.;:—()\-]+",
                        " ",
                        seg_raw,
                    )
                    seg = re.sub(r"\s+", " ", seg).strip()
                    if seg:
                        all_segments[ln] = seg
            except (ValueError, IndexError):
                pass
            i += 2

    if not all_segments:
        return ""

    # Collect segments overlapping [start, end].
    # Each marker N represents approximately lines N..N+5.
    relevant: list[str] = []
    for marker_ln in sorted(all_segments.keys()):
        seg_end = marker_ln + 5
        if marker_ln <= end and seg_end >= start:
            relevant.append(all_segments[marker_ln])

    if not relevant:
        return ""

    return re.sub(r"\s+", " ", " ".join(relevant)).strip()


def fetch_greek_passage(client: OpenAI, model: str, book: int, start: int, end: int) -> str:
    """Fetch Greek passage from Perseus. Returns empty string on failure."""
    greek = fetch_greek_passage_from_perseus(book, start, end)
    if greek and len(greek) > 30:
        return greek
    print(f"    WARNING: Perseus returned thin Greek for {book}.{start}-{end} (len={len(greek)})", flush=True)
    return greek


def split_translation_by_books(text: str, translator_key: str) -> dict[int, str]:
    """Split a Gutenberg translation text into books."""
    books: dict[int, str] = {}

    # Different translations use different book markers.
    # Try patterns from most to least specific; accept if we find >= MIN_BOOKS distinct books.
    # The ordinal pattern (Chapman) is last to avoid matching prose like "the fifth book".
    MIN_BOOKS = 5
    patterns = [
        r"BOOK\s+([IVXLCDM]+|[0-9]+)[\s\.]",
        r"Book\s+([IVXLCDM]+|[0-9]+)[\s\.]",
        r"BOOK\s+([IVXLCDM]+)",
        r"\bBOOK\s+([IVXLCDM]+)\b",
        # Chapman: "THE FIRST BOOK OF HOMER'S ODYSSEYS" — last because it can false-positive
        # on prose like "the fifth book" in other texts.
        r"THE\s+(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH|NINTH|TENTH|"
        r"ELEVENTH|TWELFTH|THIRTEENTH|FOURTEENTH|FIFTEENTH|SIXTEENTH|SEVENTEENTH|"
        r"EIGHTEENTH|NINETEENTH|TWENTIETH|TWENTY-FIRST|TWENTY-SECOND|TWENTY-THIRD|"
        r"TWENTY-FOURTH)\s+BOOK",
    ]

    _ORDINAL_TO_INT = {
        "FIRST": 1, "SECOND": 2, "THIRD": 3, "FOURTH": 4, "FIFTH": 5,
        "SIXTH": 6, "SEVENTH": 7, "EIGHTH": 8, "NINTH": 9, "TENTH": 10,
        "ELEVENTH": 11, "TWELFTH": 12, "THIRTEENTH": 13, "FOURTEENTH": 14,
        "FIFTEENTH": 15, "SIXTEENTH": 16, "SEVENTEENTH": 17, "EIGHTEENTH": 18,
        "NINETEENTH": 19, "TWENTIETH": 20, "TWENTY-FIRST": 21,
        "TWENTY-SECOND": 22, "TWENTY-THIRD": 23, "TWENTY-FOURTH": 24,
    }

    def roman_to_int(s: str) -> int | None:
        roman_vals = {
            'I': 1, 'V': 5, 'X': 10, 'L': 50,
            'C': 100, 'D': 500, 'M': 1000
        }
        s = s.upper().strip()
        if s.isdigit():
            return int(s)
        result = 0
        prev = 0
        for ch in reversed(s):
            val = roman_vals.get(ch, 0)
            if val < prev:
                result -= val
            else:
                result += val
            prev = val
        return result if result > 0 else None

    splits: list[tuple[int, int]] = []  # (book_num, char_pos)
    for pattern in patterns:
        candidate: list[tuple[int, int]] = []
        for m in re.finditer(pattern, text, re.IGNORECASE):
            captured = m.group(1).upper().strip()
            # Try ordinal word first (Chapman format)
            book_num = _ORDINAL_TO_INT.get(captured)
            if book_num is None:
                book_num = roman_to_int(captured)
            if book_num and 1 <= book_num <= 24:
                candidate.append((book_num, m.start()))
        # Only accept if we found enough distinct books (avoids prose false-positives)
        if len({bn for bn, _ in candidate}) >= MIN_BOOKS:
            splits = candidate
            break

    if not splits:
        print(f"    WARNING: Could not split {translator_key} by books. Will treat as single block.")
        books[0] = text  # fallback: unsplit
        return books

    splits.sort(key=lambda x: x[1])
    for i, (book_num, pos) in enumerate(splits):
        end_pos = splits[i + 1][1] if i + 1 < len(splits) else len(text)
        books[book_num] = text[pos:end_pos]

    return books


def extract_matching_english(
    client: OpenAI,
    model: str,
    greek_passage: str,
    book_text: str,
    translator_name: str,
    book: int,
    start: int,
    end: int,
) -> str:
    """Use an LLM to extract the English passage from the actual downloaded text.

    Provides a proportionally-selected window from the actual downloaded text and
    instructs the LLM to quote from it verbatim, not from memory.
    """
    book_size = BOOK_SIZES.get(book, 500)
    proportion = start / book_size
    # Use a larger window and bias toward beginning of book for early passages
    window_size = min(10000, len(book_text))
    raw_center = int(len(book_text) * proportion)
    window_start = max(0, raw_center - window_size // 3)
    window_end = min(len(book_text), window_start + window_size)
    book_window = book_text[window_start:window_end]

    system = (
        "You are a classical scholar. Your job is to find and quote a specific passage "
        "from a provided excerpt of an English translation. "
        "You MUST quote only text that appears verbatim in the provided excerpt. "
        "Do NOT use your memory of other translations. Do NOT paraphrase. "
        "If the passage is not in the excerpt, say 'NOT FOUND'."
    )
    user = f"""I need the passage from {translator_name}'s Odyssey translation that corresponds to
Book {book}, lines {start}-{end}. The passage covers this content (based on the Greek):

GREEK CONTENT SUMMARY: {greek_passage[:300]}...

Here is an EXCERPT from {translator_name}'s actual translation of Book {book}
(copy exactly from this, do not use any other source):

---BEGIN EXCERPT---
{book_window}
---END EXCERPT---

Find and copy the portion of the EXCERPT ABOVE that corresponds to Book {book} lines {start}-{end}.
Quote it exactly as it appears. Do not add anything from memory. Do not paraphrase.
Return just the passage text, nothing else. If not found in the excerpt, say 'NOT FOUND'."""

    result = _call_llm(client, model, system, user, temperature=0.0)
    result = result.strip()
    if "NOT FOUND" in result.upper() or len(result) < 20:
        return ""
    return result


def build_pool(
    *,
    client: OpenAI,
    model: str,
    passage_specs: list[tuple],
    output_path: Path,
) -> None:
    print("Fetching translation texts from Project Gutenberg...", flush=True)
    translation_texts: dict[str, str] = {}
    for key, url in GUTENBERG_URLS.items():
        print(f"  Downloading {key}...", flush=True)
        translation_texts[key] = _fetch_url(url)
        print(f"  Done ({len(translation_texts[key])} chars)", flush=True)

    print("\nSplitting translations by book...", flush=True)
    translation_books: dict[str, dict[int, str]] = {}
    for key, text in translation_texts.items():
        translation_books[key] = split_translation_by_books(text, key)
        found = sorted(translation_books[key].keys())
        print(f"  {key}: found books {found[:10]}{'...' if len(found) > 10 else ''}", flush=True)

    pool: list[dict] = []

    for book, start, end, description in passage_specs:
        print(f"\nPassage: Od. {book}.{start}-{end} — {description}", flush=True)

        # Fetch Greek
        print("  Fetching Greek...", flush=True)
        greek = fetch_greek_passage(client, model, book, start, end)
        if not greek:
            print(f"  SKIP: empty Greek for {book}.{start}-{end}")
            continue
        print(f"  Greek: {greek[:80]}...", flush=True)

        entry: dict = {
            "book": book,
            "start_line": start,
            "end_line": end,
            "description": description,
            "greek": greek,
        }

        # Extract matching English from each translation
        for key in ["butler", "butcher_lang", "chapman"]:
            books_dict = translation_books[key]
            book_text = books_dict.get(book, books_dict.get(0, ""))
            if not book_text:
                print(f"  WARNING: No Book {book} text for {key}")
                entry[key] = ""
                continue

            print(f"  Extracting {key} passage...", flush=True)
            eng = extract_matching_english(
                client, model, greek, book_text,
                translator_name=key.replace("_", " ").title(),
                book=book, start=start, end=end,
            )
            entry[key] = eng
            print(f"  {key}: {eng[:80]}...", flush=True)
            time.sleep(0.5)  # be kind to the API

        pool.append(entry)
        print(f"  Added passage ({len(pool)} total)", flush=True)

        # Save incrementally in case of interruption
        output_path.write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nPool complete: {len(pool)} passages written to {output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Odyssey passage pool.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default=str(ROOT / "passages_pool.json"))
    parser.add_argument(
        "--books",
        default=None,
        help="Comma-separated book numbers to include (e.g. 1,5,9). Default: all curated passages.",
    )
    args = parser.parse_args()

    _load_dotenv(ROOT / ".env")
    api_key = (os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        sys.exit("Missing OPENROUTER_API_KEY")

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    specs = PASSAGE_SPECS
    if args.books:
        allowed = set(int(b.strip()) for b in args.books.split(","))
        specs = [(b, s, e, d) for b, s, e, d in PASSAGE_SPECS if b in allowed]
        print(f"Filtering to books: {allowed} → {len(specs)} passages", flush=True)

    output_path = Path(args.output)
    build_pool(client=client, model=args.model, passage_specs=specs, output_path=output_path)


if __name__ == "__main__":
    main()
