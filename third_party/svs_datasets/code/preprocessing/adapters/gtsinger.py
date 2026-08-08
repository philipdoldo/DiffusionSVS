"""Adapters for the `GTSinger` TextGrid + JSON sidecar family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.types import CanonicalExample, Interval, NoteInterval
from preprocessing.audio import read_audio_metadata
from preprocessing.labels.textgrid import parse_textgrid
from preprocessing.phonesets import (
    DEFAULT_JAPANESE_LAB_NORMALIZATION,
    normalize_gtsinger_chinese_phone,
    normalize_gtsinger_english_phone,
    normalize_gtsinger_japanese_phone,
    normalize_mandarin_phone_intervals,
)


def _flatten_gtsinger_notes(items: list[dict[str, Any]]) -> tuple[NoteInterval, ...]:
    note_intervals: list[NoteInterval] = []
    for item in items:
        if "note" not in item or "note_start" not in item or "note_end" not in item:
            continue
        word = str(item["word"])
        note_values = item["note"]
        note_starts = item["note_start"]
        note_ends = item["note_end"]
        if not (len(note_values) == len(note_starts) == len(note_ends)):
            raise ValueError("GTSinger note metadata lengths must match")

        for note_value, start_sec, end_sec in zip(
            note_values,
            note_starts,
            note_ends,
            strict=True,
        ):
            note_int = int(note_value)
            note_intervals.append(
                NoteInterval(
                    start_sec=float(start_sec),
                    end_sec=float(end_sec),
                    pitch_midi=None if note_int == 0 else note_int,
                    lyric=word,
                )
            )
    return tuple(note_intervals)


def _compact_gtsinger_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact_items: list[dict[str, Any]] = []
    for item in items:
        compact_items.append(
            {
                "word": item.get("word"),
                "start_time": item.get("start_time"),
                "end_time": item.get("end_time"),
                "ph_count": len(item.get("ph", [])),
                "mix": list(item.get("mix", [])),
                "falsetto": list(item.get("falsetto", [])),
                "breathy": list(item.get("breathy", [])),
                "pharyngeal": list(item.get("pharyngeal", [])),
                "glissando": list(item.get("glissando", [])),
                "vibrato": list(item.get("vibrato", [])),
                "tech": item.get("tech"),
                "singing_method": item.get("singing_method"),
                "pace": item.get("pace"),
                "range": item.get("range"),
                "emotion": item.get("emotion"),
            }
        )
    return compact_items


def _normalize_gtsinger_japanese_default(label: str) -> str:
    return normalize_gtsinger_japanese_phone(
        label,
        config=DEFAULT_JAPANESE_LAB_NORMALIZATION,
    )


def _normalize_gtsinger_phone_intervals(
    phone_intervals: tuple[Interval, ...],
    *,
    source_dataset: str,
) -> tuple[Interval, ...]:
    if source_dataset == "GTSinger_English":
        normalizer = normalize_gtsinger_english_phone
    elif source_dataset == "GTSinger_Japanese":
        normalizer = _normalize_gtsinger_japanese_default
    elif source_dataset == "GTSinger_Chinese":
        return normalize_mandarin_phone_intervals(
            tuple(
                Interval(
                    label=normalize_gtsinger_chinese_phone(interval.label),
                    start_sec=interval.start_sec,
                    end_sec=interval.end_sec,
                )
                for interval in phone_intervals
            ),
            source_dataset=source_dataset,
        )
    else:
        return phone_intervals

    return tuple(
        Interval(
            label=normalizer(interval.label),
            start_sec=interval.start_sec,
            end_sec=interval.end_sec,
        )
        for interval in phone_intervals
    )


def _derive_gtsinger_utterance_id(audio_path: Path, source_dataset: str) -> str:
    audio_path = Path(audio_path)
    stem_path = audio_path.with_suffix("")
    parts = stem_path.parts
    if source_dataset in parts:
        source_index = parts.index(source_dataset)
        relative_parts = parts[source_index + 1 :]
        if relative_parts:
            return "/".join(relative_parts)
    return f"{audio_path.parent.name}/{audio_path.stem}"


def adapt_gtsinger_segment(
    *,
    textgrid_path: str | Path,
    json_path: str | Path,
    audio_path: str | Path,
    source_dataset: str,
    speaker_id: str | None = None,
    include_audio_metadata: bool = False,
) -> CanonicalExample:
    """Adapt one `GTSinger` segment into a canonical example."""
    textgrid_path_obj = Path(textgrid_path)
    json_path_obj = Path(json_path)
    audio_path_obj = Path(audio_path)

    with textgrid_path_obj.open("r", encoding="utf-8") as handle:
        textgrid = parse_textgrid(handle.read())
    with json_path_obj.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if not isinstance(payload, list) or not payload:
        raise ValueError("GTSinger JSON sidecar must decode to a non-empty list")
    items = [dict(item) for item in payload]

    word_tier = textgrid.choose_tier(("word", "words", "lyric", "lyrics", "text"))
    phone_tier = textgrid.choose_tier(("phone", "phones", "phoneme", "phonemes"))
    if phone_tier is None:
        raise ValueError("GTSinger segment did not expose a phone tier")

    phone_intervals = _normalize_gtsinger_phone_intervals(
        phone_tier.labeled_intervals(),
        source_dataset=source_dataset,
    )
    word_intervals = None if word_tier is None else word_tier.labeled_intervals()
    note_intervals = _flatten_gtsinger_notes(items)

    style_source = next(
        (item for item in items if item.get("word") != "<SP>"), items[0]
    )
    metadata = {
        "phone_tier_name": phone_tier.name,
        "gtsinger_style": {
            "singing_method": style_source.get("singing_method"),
            "pace": style_source.get("pace"),
            "range": style_source.get("range"),
            "emotion": style_source.get("emotion"),
        },
        # Keep only the fields needed downstream for word timing and style labels.
        "gtsinger_items": _compact_gtsinger_items(items),
    }
    if word_tier is not None:
        metadata["word_tier_name"] = word_tier.name

    audio_metadata = (
        read_audio_metadata(audio_path_obj, resolve=False)
        if include_audio_metadata
        else None
    )
    utterance_id = _derive_gtsinger_utterance_id(audio_path_obj, source_dataset)

    return CanonicalExample(
        audio_path=str(audio_path_obj),
        utterance_id=utterance_id,
        source_dataset=source_dataset,
        raw_format="gtsinger_textgrid_json",
        speaker_id=speaker_id,
        audio_sampling_rate=None if audio_metadata is None else audio_metadata.sample_rate,
        audio_num_samples=None if audio_metadata is None else audio_metadata.num_samples,
        phone_intervals=phone_intervals,
        word_intervals=word_intervals,
        note_intervals=note_intervals,
        line_start_sec=phone_intervals[0].start_sec,
        line_end_sec=phone_intervals[-1].end_sec,
        source_paths={
            "textgrid_path": str(textgrid_path_obj),
            "json_path": str(json_path_obj),
        },
        metadata=metadata,
    )


__all__ = [
    "adapt_gtsinger_segment",
]
