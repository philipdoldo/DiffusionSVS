"""Adapters for the `NGYY_ENG_Dataset` manifest family."""

from __future__ import annotations

import csv
from pathlib import Path

from preprocessing.audio import read_audio_metadata
from preprocessing.labels.sequence_durations import parse_sequence_duration_row
from preprocessing.phonesets import normalize_english_dataset_phone


def load_ngyy_rows(speaker_root: str | Path) -> list[dict[str, str]]:
    """Load one speaker shard from `NGYY_ENG_Dataset`."""
    speaker_root_obj = Path(speaker_root)
    csv_path = speaker_root_obj / "transcriptions.csv"
    with csv_path.open("r", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def adapt_ngyy_row(
    row: dict[str, object],
    *,
    speaker_root: str | Path,
    audio_dir_name: str = "wavs",
    audio_suffix: str = ".wav",
    include_audio_metadata: bool = True,
) -> object:
    """Adapt one `NGYY_ENG_Dataset` row into a canonical example."""
    speaker_root_obj = Path(speaker_root)
    audio_path = speaker_root_obj / audio_dir_name / f"{row['name']}{audio_suffix}"
    audio_metadata = (
        read_audio_metadata(audio_path, resolve=False)
        if include_audio_metadata
        else None
    )

    return parse_sequence_duration_row(
        row,
        audio_path=str(audio_path),
        source_dataset="NGYY_ENG_Dataset",
        raw_format="ngyy_transcriptions",
        speaker_id=speaker_root_obj.name,
        source_paths={
            "transcriptions_path": str(speaker_root_obj / "transcriptions.csv")
        },
        audio_sampling_rate=None if audio_metadata is None else audio_metadata.sample_rate,
        audio_num_samples=None if audio_metadata is None else audio_metadata.num_samples,
        phone_normalizer=lambda label: normalize_english_dataset_phone(
            label,
            source_dataset="NGYY_ENG_Dataset",
        ),
    )


__all__ = [
    "adapt_ngyy_row",
    "load_ngyy_rows",
]
