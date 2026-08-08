"""Adapters for the `sungandspoken` seconds-based phone-text family."""

from __future__ import annotations

from pathlib import Path

from core.types import CanonicalExample, Interval
from preprocessing.audio import read_audio_metadata
from preprocessing.labels.lab import parse_lab_text
from preprocessing.phonesets import normalize_english_dataset_phone


def _clip_overlapping_intervals(intervals: tuple[Interval, ...]) -> tuple[Interval, ...]:
    clipped: list[Interval] = []
    previous_end = 0.0
    for interval in intervals:
        start_sec = max(interval.start_sec, previous_end)
        end_sec = max(interval.end_sec, start_sec)
        clipped.append(
            Interval(
                label=interval.label,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )
        previous_end = end_sec
    return tuple(clipped)


def adapt_sungandspoken_pair(
    *,
    audio_path: str | Path,
    label_path: str | Path,
    source_dataset: str = "sungandspoken",
    include_audio_metadata: bool = False,
) -> object:
    """Adapt one `sungandspoken` audio + `.txt` label pair into a canonical example."""
    audio_path_obj = Path(audio_path)
    label_path_obj = Path(label_path)
    audio_metadata = (
        read_audio_metadata(audio_path_obj, resolve=False)
        if include_audio_metadata
        else None
    )

    with label_path_obj.open("r", encoding="utf-8") as handle:
        label_text = handle.read()

    subject_id = (
        audio_path_obj.parents[1].name if len(audio_path_obj.parents) >= 2 else None
    )
    performance_mode = audio_path_obj.parent.name

    phone_intervals = _clip_overlapping_intervals(
        parse_lab_text(
            label_text,
            time_unit="seconds",
            phone_normalizer=lambda label: normalize_english_dataset_phone(
                label,
                source_dataset=source_dataset,
            ),
        )
    )

    return CanonicalExample(
        audio_path=str(audio_path_obj),
        utterance_id=f"{subject_id}/{performance_mode}/{audio_path_obj.stem}",
        source_dataset=source_dataset,
        raw_format="lab",
        audio_sampling_rate=None if audio_metadata is None else audio_metadata.sample_rate,
        audio_num_samples=None if audio_metadata is None else audio_metadata.num_samples,
        speaker_id=subject_id,
        phone_intervals=phone_intervals,
        line_start_sec=phone_intervals[0].start_sec,
        line_end_sec=phone_intervals[-1].end_sec,
        source_paths={"label_path": str(label_path_obj)},
        metadata={"performance_mode": performance_mode},
    )


__all__ = [
    "adapt_sungandspoken_pair",
]
