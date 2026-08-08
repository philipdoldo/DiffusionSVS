"""Adapters for the `m4singer` TextGrid + metadata family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.types import CanonicalExample, Interval, NoteInterval
from preprocessing.audio import read_audio_metadata
from preprocessing.labels.textgrid import TextGrid, TextGridTier, parse_textgrid
from preprocessing.phonesets import (
    normalize_mandarin_dataset_phone,
)
from preprocessing.phonesets.phonesets import MANDARIN_CANONICAL_PHONE_TOKENS


def load_m4singer_metadata(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Load the top-level `m4singer` metadata table."""
    dataset_root_obj = Path(dataset_root)
    with (dataset_root_obj / "meta.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("m4singer meta.json must decode to a list")
    return [dict(item) for item in payload]


def _note_intervals_from_item(item: dict[str, Any]) -> tuple[NoteInterval, ...]:
    notes = tuple(int(note) for note in item["notes"])
    note_durations = tuple(float(duration) for duration in item["notes_dur"])
    slur_flags = tuple(bool(flag) for flag in item["is_slur"])

    if not (len(notes) == len(note_durations) == len(slur_flags)):
        raise ValueError("m4singer notes, note durations, and slur flags must align")

    cursor = 0.0
    note_intervals: list[NoteInterval] = []
    for note, duration, is_slur in zip(notes, note_durations, slur_flags, strict=True):
        start_sec = cursor
        end_sec = cursor + duration
        # TODO: lyrics
        note_intervals.append(
            NoteInterval(
                start_sec=start_sec,
                end_sec=end_sec,
                pitch_midi=note,
                is_slur=is_slur,
            )
        )
        cursor = end_sec
    return tuple(note_intervals)


def _normalize_phone_intervals(
    phone_intervals: tuple[Interval, ...],
    *,
    source_dataset: str,
) -> tuple[Interval, ...]:
    return tuple(
        Interval(
            label=normalize_mandarin_dataset_phone(
                interval.label, source_dataset=source_dataset
            ),
            start_sec=interval.start_sec,
            end_sec=interval.end_sec,
        )
        for interval in phone_intervals
    )


def _normalize_manifest_phone_sequence(item: dict[str, Any]) -> tuple[str, ...]:
    return tuple(
        normalize_mandarin_dataset_phone(str(phone), source_dataset="m4singer")
        for phone in item["phs"]
    )


def _select_m4singer_phone_and_word_tiers(
    textgrid: TextGrid,
    *,
    expected_phone_sequence: tuple[str, ...],
) -> tuple[TextGridTier, TextGridTier | None]:
    non_empty_tiers = [
        tier for tier in textgrid.tiers if tier.non_empty_interval_count > 0
    ]
    if not non_empty_tiers:
        raise ValueError("m4singer TextGrid did not expose any labeled tiers")

    phone_tier = None
    for tier in non_empty_tiers:
        normalized_labels = tuple(
            normalize_mandarin_dataset_phone(interval.label, source_dataset="m4singer")
            for interval in tier.labeled_intervals()
        )
        if normalized_labels == expected_phone_sequence:
            phone_tier = tier
            break

    if phone_tier is None:
        canonical_phone_set = set(MANDARIN_CANONICAL_PHONE_TOKENS) | {"AP", "SP"}
        ranked_tiers = []
        for tier in non_empty_tiers:
            normalized_labels = tuple(
                normalize_mandarin_dataset_phone(
                    interval.label, source_dataset="m4singer"
                )
                for interval in tier.labeled_intervals()
            )
            canonical_hits = sum(
                1 for label in normalized_labels if label in canonical_phone_set
            )
            ranked_tiers.append(
                (
                    canonical_hits / max(len(normalized_labels), 1),
                    canonical_hits,
                    -len(normalized_labels),
                    tier,
                )
            )
        ranked_tiers.sort(reverse=True)
        if ranked_tiers and ranked_tiers[0][0] > 0.0:
            phone_tier = ranked_tiers[0][3]
        else:
            phone_tier = textgrid.choose_tier(
                ("phones", "phone", "phonemes", "phoneme")
            )
    if phone_tier is None:
        raise ValueError("m4singer item did not expose a phone tier")

    word_tier = None
    for tier in non_empty_tiers:
        if tier is not phone_tier:
            word_tier = tier
            break
    if word_tier is None:
        fallback_word_tier = textgrid.choose_tier(
            ("words", "word", "lyrics", "lyric", "text")
        )
        if fallback_word_tier is not phone_tier:
            word_tier = fallback_word_tier
    return phone_tier, word_tier


def adapt_m4singer_item(
    item: dict[str, Any],
    *,
    dataset_root: str | Path,
    audio_suffix: str = ".wav",
    include_audio_metadata: bool = False,
) -> CanonicalExample:
    """Adapt one `m4singer` metadata item into a canonical example."""
    dataset_root_obj = Path(dataset_root)
    item_name = str(item["item_name"])
    song_name, clip_name = item_name.rsplit("#", maxsplit=1)

    textgrid_path = dataset_root_obj / song_name / f"{clip_name}.TextGrid"
    audio_path = dataset_root_obj / song_name / f"{clip_name}{audio_suffix}"
    audio_metadata = (
        read_audio_metadata(audio_path, resolve=False)
        if include_audio_metadata
        else None
    )

    with textgrid_path.open("r", encoding="utf-8") as handle:
        textgrid = parse_textgrid(handle.read())

    expected_phone_sequence = _normalize_manifest_phone_sequence(item)
    phone_tier, word_tier = _select_m4singer_phone_and_word_tiers(
        textgrid,
        expected_phone_sequence=expected_phone_sequence,
    )

    phone_intervals = _normalize_phone_intervals(
        phone_tier.labeled_intervals(),
        source_dataset="m4singer",
    )
    word_intervals = None if word_tier is None else word_tier.labeled_intervals()
    note_intervals = _note_intervals_from_item(item)

    metadata = {
        "phone_tier_name": phone_tier.name,
        "raw_manifest_fields": {
            "phs": list(item["phs"]),
            "ph_dur": list(item["ph_dur"]),
            "notes": list(item["notes"]),
            "notes_dur": list(item["notes_dur"]),
            "is_slur": list(item["is_slur"]),
        },
    }
    if word_tier is not None:
        metadata["word_tier_name"] = word_tier.name

    return CanonicalExample(
        audio_path=str(audio_path),
        utterance_id=item_name,
        source_dataset="m4singer",
        raw_format="m4singer_textgrid",
        audio_sampling_rate=None
        if audio_metadata is None
        else audio_metadata.sample_rate,
        audio_num_samples=None
        if audio_metadata is None
        else audio_metadata.num_samples,
        lyrics_text=str(item["txt"]),
        phone_intervals=phone_intervals,
        word_intervals=word_intervals,
        note_intervals=note_intervals,
        line_start_sec=phone_intervals[0].start_sec,
        line_end_sec=phone_intervals[-1].end_sec,
        source_paths={
            "textgrid_path": str(textgrid_path),
            "meta_path": str(dataset_root_obj / "meta.json"),
        },
        metadata=metadata,
    )


__all__ = [
    "adapt_m4singer_item",
    "load_m4singer_metadata",
]
