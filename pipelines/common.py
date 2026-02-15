from __future__ import annotations


def reference_context_block(reference_translations: dict[str, str]) -> str:
    dryden = reference_translations.get("dryden_clough", "").strip()
    perrin = reference_translations.get("perrin", "").strip()
    return (
        "Reference translations for context:\n"
        f"- Dryden/Clough: {dryden}\n"
        f"- Perrin: {perrin}"
    )


def reference_translations_for_index(
    dryden_paragraphs: list[str],
    perrin_paragraphs: list[str],
    paragraph_index: int,
) -> dict[str, str]:
    at = paragraph_index - 1
    return {
        "dryden_clough": dryden_paragraphs[at] if at < len(dryden_paragraphs) else "",
        "perrin": perrin_paragraphs[at] if at < len(perrin_paragraphs) else "",
    }

