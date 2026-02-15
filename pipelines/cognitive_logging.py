from __future__ import annotations

import sys
from typing import Any, Callable


def make_vprint(
    *,
    verbose: bool,
    color_enabled: bool,
    colorize_fn: Callable[[str, str | None, bool], str],
    stage_colors: dict[str, str],
) -> Callable[[str, str | None], None]:
    def vprint(message: str, stage: str | None = None) -> None:
        if not verbose:
            return
        color = stage_colors.get(stage) if stage else None
        print(colorize_fn(message, color, color_enabled), file=sys.stderr)

    return vprint


def log_reference_inputs(
    vprint: Callable[[str, str | None], None],
    *,
    paragraph_index: int,
    pipeline_label: str,
    user_preference: str,
    reference_translations: dict[str, str],
) -> None:
    vprint(f"[paragraph {paragraph_index}] {pipeline_label}...", "iteration")
    vprint(
        f"[paragraph {paragraph_index}] user preference prompt: {user_preference}",
        "reference",
    )
    vprint(
        f"[paragraph {paragraph_index}] reference input [dryden_clough]: "
        f"{reference_translations.get('dryden_clough', '')}",
        "reference",
    )
    vprint(
        f"[paragraph {paragraph_index}] reference input [perrin]: "
        f"{reference_translations.get('perrin', '')}",
        "reference",
    )


def log_user_iteration(
    vprint: Callable[[str, str | None], None],
    *,
    paragraph_index: int,
    iteration: int,
    translation_result: dict[str, Any],
    translation: str,
) -> None:
    notes = translation_result.get("phrase_process_notes", [])
    vprint(
        f"[paragraph {paragraph_index}] [iter {iteration}] process notes count: "
        f"{len(notes) if isinstance(notes, list) else 0}",
        "iteration",
    )
    if isinstance(notes, list):
        for note_idx, note in enumerate(notes, start=1):
            if not isinstance(note, dict):
                continue
            vprint(
                f"[paragraph {paragraph_index}] [iter {iteration}] phrase {note_idx} source: "
                f"{str(note.get('source_phrase', '')).strip()}",
                "iteration",
            )
            vprint(
                f"[paragraph {paragraph_index}] [iter {iteration}] phrase {note_idx} context: "
                f"{str(note.get('context_note', '')).strip()}",
                "iteration",
            )
            vprint(
                f"[paragraph {paragraph_index}] [iter {iteration}] phrase {note_idx} simple anchor: "
                f"{str(note.get('simple_anchor', '')).strip()}",
                "iteration",
            )
            vprint(
                f"[paragraph {paragraph_index}] [iter {iteration}] phrase {note_idx} connotation targets: "
                f"{str(note.get('connotation_targets', '')).strip()}",
                "iteration",
            )
            candidates = note.get("candidate_options", [])
            if isinstance(candidates, list):
                for cand_idx, cand in enumerate(candidates, start=1):
                    vprint(
                        f"[paragraph {paragraph_index}] [iter {iteration}] phrase {note_idx} candidate {cand_idx}: "
                        f"{str(cand).strip()}",
                        "iteration",
                    )
            vprint(
                f"[paragraph {paragraph_index}] [iter {iteration}] phrase {note_idx} chosen: "
                f"{str(note.get('chosen_phrase', '')).strip()}",
                "iteration",
            )
    vprint(
        f"[paragraph {paragraph_index}] [iter {iteration}] zoom-out notes: "
        f"{str(translation_result.get('zoom_out_notes', '')).strip()}",
        "iteration",
    )
    vprint(
        f"[paragraph {paragraph_index}] [iter {iteration}] draft translation:\n{translation}",
        "iteration",
    )


def log_dualloop_iteration(
    vprint: Callable[[str, str | None], None],
    *,
    paragraph_index: int,
    iteration: int,
    translation_result: dict[str, Any],
    translation: str,
) -> None:
    vprint(
        f"[paragraph {paragraph_index}] [iter {iteration}] scene model: "
        f"{str(translation_result.get('scene_model', '')).strip()}",
        "iteration",
    )
    claim_map = translation_result.get("claim_map", [])
    if isinstance(claim_map, list):
        for claim_idx, claim in enumerate(claim_map, start=1):
            vprint(
                f"[paragraph {paragraph_index}] [iter {iteration}] claim {claim_idx}: {str(claim).strip()}",
                "iteration",
            )
    vprint(
        f"[paragraph {paragraph_index}] [iter {iteration}] plain restatement: "
        f"{str(translation_result.get('plain_restatement', '')).strip()}",
        "iteration",
    )
    ledger = translation_result.get("constraint_ledger", {})
    if isinstance(ledger, dict):
        non_negotiables = ledger.get("non_negotiables", [])
        negotiables = ledger.get("negotiables", [])
        if isinstance(non_negotiables, list):
            for nn_idx, item in enumerate(non_negotiables, start=1):
                vprint(
                    f"[paragraph {paragraph_index}] [iter {iteration}] non-negotiable {nn_idx}: {str(item).strip()}",
                    "iteration",
                )
        if isinstance(negotiables, list):
            for ng_idx, item in enumerate(negotiables, start=1):
                vprint(
                    f"[paragraph {paragraph_index}] [iter {iteration}] negotiable {ng_idx}: {str(item).strip()}",
                    "iteration",
                )
    drafts = translation_result.get("drafts", {})
    if isinstance(drafts, dict):
        vprint(
            f"[paragraph {paragraph_index}] [iter {iteration}] draft source_close: "
            f"{str(drafts.get('source_close', '')).strip()}",
            "iteration",
        )
        vprint(
            f"[paragraph {paragraph_index}] [iter {iteration}] draft plain_natural: "
            f"{str(drafts.get('plain_natural', '')).strip()}",
            "iteration",
        )
        vprint(
            f"[paragraph {paragraph_index}] [iter {iteration}] draft balanced: "
            f"{str(drafts.get('balanced', '')).strip()}",
            "iteration",
        )
    vprint(
        f"[paragraph {paragraph_index}] [iter {iteration}] draft translation:\n{translation}",
        "iteration",
    )


def log_iteration_focus(
    vprint: Callable[[str, str | None], None],
    *,
    paragraph_index: int,
    iteration: int,
    focus_text: str,
) -> None:
    if focus_text:
        vprint(
            f"[paragraph {paragraph_index}] [iter {iteration}] carry-forward focus: {focus_text}",
            "final",
        )


def log_final_selection(
    vprint: Callable[[str, str | None], None],
    *,
    paragraph_index: int,
    translation_label: str,
    final_translation: str,
    selected_iteration: int,
    selection_notes: str,
) -> None:
    vprint(f"[paragraph {paragraph_index}] final {translation_label} translation:", "final")
    vprint(final_translation, "final")
    vprint(f"[paragraph {paragraph_index}] selected iteration: {selected_iteration}", "final")
    if selection_notes:
        vprint(
            f"[paragraph {paragraph_index}] final selection notes: {selection_notes}",
            "final",
        )

