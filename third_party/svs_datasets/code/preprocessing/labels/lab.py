"""Parser utilities for simple mono-phone `.lab` files."""

from __future__ import annotations

from collections.abc import Callable

from core.types import CanonicalExample, Interval

HTK_100NS_PER_SECOND = 10_000_000.0


def _is_integer_like(value: str) -> bool:
    stripped = value.strip()
    return bool(stripped) and stripped.lstrip("+-").isdigit()


def _infer_lab_time_unit(rows: list[tuple[str, str, str]]) -> str:
    if not rows:
        raise ValueError("cannot infer a time unit from an empty .lab payload")

    integer_like = all(
        _is_integer_like(start) and _is_integer_like(end) for start, end, _ in rows
    )
    if not integer_like:
        return "seconds"

    max_endpoint = max(abs(int(end)) for _, end, _ in rows)
    if max_endpoint > 1_000:
        return "htk_100ns"
    return "seconds"


def _convert_time(raw_value: str, *, time_unit: str) -> float:
    value = float(raw_value)
    if time_unit == "seconds":
        return value
    if time_unit == "htk_100ns":
        return value / HTK_100NS_PER_SECOND
    raise ValueError(f"unsupported time_unit {time_unit!r}")


def parse_lab_text(
    text: str,
    *,
    time_unit: str = "auto",
    drop_empty: bool = True,
    phone_normalizer: Callable[[str], str] | None = None,
    repair_invalid_intervals: bool = False,
) -> tuple[Interval, ...]:
    """Parse a simple `.lab` payload into canonical intervals.

    Supported line format:
    - `<start> <end> <label>`

    Time units:
    - `"auto"`: infer seconds vs HTK 100 ns ticks
    - `"seconds"`
    - `"htk_100ns"`
    """
    rows: list[tuple[str, str, str]] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 3:
            raise ValueError(
                f".lab line {line_number} must contain start, end, and label, got {raw_line!r}"
            )
        start_raw, end_raw = parts[0], parts[1]
        label = " ".join(parts[2:])
        if phone_normalizer is not None:
            label = phone_normalizer(label)
        rows.append((start_raw, end_raw, label))

    actual_time_unit = _infer_lab_time_unit(rows) if time_unit == "auto" else time_unit
    intervals: list[Interval] = []
    previous_end: float | None = None
    for start_raw, end_raw, label in rows:
        if drop_empty and not label:
            continue
        start_sec = _convert_time(start_raw, time_unit=actual_time_unit)
        end_sec = _convert_time(end_raw, time_unit=actual_time_unit)
        if repair_invalid_intervals:
            if previous_end is not None and start_sec < previous_end:
                start_sec = previous_end
            if end_sec < start_sec:
                end_sec = start_sec
        intervals.append(
            Interval(
                label=label,
                start_sec=start_sec,
                end_sec=end_sec,
            )
        )
        previous_end = end_sec
    return tuple(intervals)


def parse_lab_example(
    text: str,
    *,
    audio_path: str,
    utterance_id: str,
    source_dataset: str,
    label_path: str | None = None,
    time_unit: str = "auto",
    phone_normalizer: Callable[[str], str] | None = None,
    audio_sampling_rate: int | None = None,
    audio_num_samples: int | None = None,
    speaker_id: str | None = None,
    metadata: dict[str, object] | None = None,
    repair_invalid_intervals: bool = False,
) -> CanonicalExample:
    """Parse one simple `.lab` payload into a canonical full-label example."""
    phone_intervals = parse_lab_text(
        text,
        time_unit=time_unit,
        phone_normalizer=phone_normalizer,
        repair_invalid_intervals=repair_invalid_intervals,
    )
    if not phone_intervals:
        raise ValueError("a .lab example must contain at least one phone interval")

    source_paths = {}
    if label_path is not None:
        source_paths["label_path"] = label_path

    return CanonicalExample(
        audio_path=audio_path,
        utterance_id=utterance_id,
        source_dataset=source_dataset,
        raw_format="lab",
        speaker_id=speaker_id,
        audio_sampling_rate=audio_sampling_rate,
        audio_num_samples=audio_num_samples,
        phone_intervals=phone_intervals,
        line_start_sec=phone_intervals[0].start_sec,
        line_end_sec=phone_intervals[-1].end_sec,
        source_paths=source_paths,
        metadata={} if metadata is None else dict(metadata),
    )


__all__ = [
    "HTK_100NS_PER_SECOND",
    "parse_lab_example",
    "parse_lab_text",
]
