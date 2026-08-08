"""Adapters for the `opencpop` sequence-duration transcription family."""

from __future__ import annotations

from pathlib import Path

from preprocessing.audio import read_audio_metadata
from preprocessing.labels.sequence_durations import (
    _split_floats,
    _split_tokens,
    parse_sequence_duration_row,
)
from preprocessing.phonesets import normalize_mandarin_phone_sequence_with_durations


def parse_opencpop_line(line: str) -> dict[str, object]:
    """Parse one `opencpop` transcription row."""
    stripped = line.strip()
    if not stripped:
        raise ValueError("opencpop transcription line must not be empty")

    parts = stripped.split("|")
    if len(parts) != 7:
        raise ValueError(
            f"opencpop transcription rows must contain 7 fields, got {len(parts)}"
        )

    return {
        "name": parts[0],
        "txt": parts[1],
        "ph_seq": parts[2],
        "note_seq": parts[3],
        "note_dur": parts[4],
        "ph_dur": parts[5],
        "note_slur": parts[6],
    }


def adapt_opencpop_line(
    line: str,
    *,
    dataset_root: str | Path,
    transcriptions_path: str | Path | None = None,
    audio_dir_name: str = "wavs",
    audio_suffix: str = ".wav",
    include_audio_metadata: bool = False,
) -> object:
    """Adapt one `opencpop` transcription row into a canonical example."""
    row = parse_opencpop_line(line)
    phone_sequence = _split_tokens(row["ph_seq"], separator=" ")
    phone_durations = _split_floats(row["ph_dur"], separator=" ")
    merged_phones, merged_durations = normalize_mandarin_phone_sequence_with_durations(
        phone_sequence,
        phone_durations,
        source_dataset="opencpop",
    )
    row = {
        **row,
        "ph_seq": " ".join(merged_phones),
        "ph_dur": " ".join(f"{duration:.10g}" for duration in merged_durations),
    }
    dataset_root_obj = Path(dataset_root)
    audio_path = dataset_root_obj / audio_dir_name / f"{row['name']}{audio_suffix}"
    audio_metadata = (
        read_audio_metadata(audio_path, resolve=False)
        if include_audio_metadata
        else None
    )

    source_paths = {}
    if transcriptions_path is not None:
        source_paths["transcriptions_path"] = str(Path(transcriptions_path))

    return parse_sequence_duration_row(
        row,
        audio_path=str(audio_path),
        source_dataset="opencpop",
        raw_format="opencpop_transcriptions",
        text_key="txt",
        source_paths=source_paths,
        audio_sampling_rate=None if audio_metadata is None else audio_metadata.sample_rate,
        audio_num_samples=None if audio_metadata is None else audio_metadata.num_samples,
    )


__all__ = [
    "adapt_opencpop_line",
    "parse_opencpop_line",
]
