"""Helpers for HTS-style full-context label files."""

from __future__ import annotations

import re

from core.types import CanonicalExample, Interval
from preprocessing.labels.lab import parse_lab_text

_CENTRAL_PHONE_PATTERN = re.compile(r"-(?P<phone>[^+]+)\+")


def extract_central_phone(label: str) -> str:
    """Extract the monophone identity from one HTS full-context label string."""
    prefix = label.split("/", maxsplit=1)[0]
    match = _CENTRAL_PHONE_PATTERN.search(prefix)
    if match is None:
        raise ValueError(f"could not extract a central phone from label {label!r}")

    phone = match.group("phone").strip()
    if not phone:
        raise ValueError(f"central phone must not be empty in label {label!r}")
    return phone


def parse_hts_fullcontext_lab_text(
    text: str,
    *,
    time_unit: str = "auto",
) -> tuple[Interval, ...]:
    """Parse HTS-style full-context labels into monophone intervals."""
    normalized_lines: list[str] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 3:
            raise ValueError(
                f"full-context label line {line_number} must contain start, end, and label"
            )

        start_raw, end_raw = parts[0], parts[1]
        label = " ".join(parts[2:])
        normalized_lines.append(f"{start_raw} {end_raw} {extract_central_phone(label)}")

    return parse_lab_text("\n".join(normalized_lines), time_unit=time_unit)


def parse_hts_fullcontext_example(
    text: str,
    *,
    audio_path: str,
    utterance_id: str,
    source_dataset: str,
    label_path: str | None = None,
    time_unit: str = "auto",
    audio_sampling_rate: int | None = None,
    audio_num_samples: int | None = None,
    speaker_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> CanonicalExample:
    """Parse one HTS-style full-context label payload into a canonical example."""
    phone_intervals = parse_hts_fullcontext_lab_text(text, time_unit=time_unit)
    if not phone_intervals:
        raise ValueError(
            "a full-context label example must contain at least one interval"
        )

    source_paths = {}
    if label_path is not None:
        source_paths["label_path"] = label_path

    metadata_dict = {} if metadata is None else dict(metadata)
    metadata_dict["label_encoding"] = "hts_fullcontext"

    return CanonicalExample(
        audio_path=audio_path,
        utterance_id=utterance_id,
        source_dataset=source_dataset,
        raw_format="hts_fullcontext_lab",
        speaker_id=speaker_id,
        audio_sampling_rate=audio_sampling_rate,
        audio_num_samples=audio_num_samples,
        phone_intervals=phone_intervals,
        line_start_sec=phone_intervals[0].start_sec,
        line_end_sec=phone_intervals[-1].end_sec,
        source_paths=source_paths,
        metadata=metadata_dict,
    )


__all__ = [
    "extract_central_phone",
    "parse_hts_fullcontext_example",
    "parse_hts_fullcontext_lab_text",
]
