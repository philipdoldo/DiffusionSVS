"""Adapters for the `Alex_Floarea_EN_Public_Corpus` manifest family."""

from __future__ import annotations

import csv
from pathlib import Path

from preprocessing.audio import read_audio_metadata
from preprocessing.labels.sequence_durations import parse_sequence_duration_row
from preprocessing.phonesets import normalize_english_dataset_phone


def load_alex_public_rows(dataset_root: str | Path) -> list[dict[str, str]]:
    """Load the top-level transcription table for the public Alex corpus."""
    dataset_root_obj = Path(dataset_root)
    csv_path = dataset_root_obj / "transcriptions.csv"
    with csv_path.open("r", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def adapt_alex_public_row(
    row: dict[str, object],
    *,
    dataset_root: str | Path,
    audio_dir_name: str = "wavs",
    audio_suffix: str = ".wav",
    include_audio_metadata: bool = False,
) -> object:
    """Adapt one public Alex corpus row into a canonical example."""
    dataset_root_obj = Path(dataset_root)
    clip_name = str(row["name"])
    audio_path = dataset_root_obj / audio_dir_name / f"{clip_name}{audio_suffix}"
    label_path = dataset_root_obj / "lab" / f"{clip_name}.lab"
    audio_metadata = (
        read_audio_metadata(audio_path, resolve=False)
        if include_audio_metadata
        else None
    )

    source_paths = {
        "transcriptions_path": str(dataset_root_obj / "transcriptions.csv"),
    }
    source_paths["label_path"] = str(label_path)

    return parse_sequence_duration_row(
        row,
        audio_path=str(audio_path),
        source_dataset="Alex_Floarea_EN_Public_Corpus",
        raw_format="alex_public_transcriptions",
        source_paths=source_paths,
        audio_sampling_rate=None if audio_metadata is None else audio_metadata.sample_rate,
        audio_num_samples=None if audio_metadata is None else audio_metadata.num_samples,
        phone_normalizer=lambda label: normalize_english_dataset_phone(
            label,
            source_dataset="Alex_Floarea_EN_Public_Corpus",
        ),
    )


__all__ = [
    "adapt_alex_public_row",
    "load_alex_public_rows",
]
