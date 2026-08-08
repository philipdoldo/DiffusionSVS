"""Sequence-plus-duration label parsing helpers, for csv or txt or other various tabular formats."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping

from core.types import CanonicalExample, Interval, NoteInterval


def _split_tokens(value: object, *, separator: str | None = " ") -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ()
        if separator is None:
            parts = text.split()
        else:
            parts = [part.strip() for part in text.split(separator)]
        return tuple(part for part in parts if part)
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    raise TypeError(f"unsupported token field type: {type(value).__name__}")


def _split_floats(value: object, *, separator: str | None = " ") -> tuple[float, ...]:
    return tuple(float(token) for token in _split_tokens(value, separator=separator))


def _parse_bool_token(token: str) -> bool:
    lowered = token.lower()
    if lowered in {"1", "true", "t", "yes"}:
        return True
    if lowered in {"0", "false", "f", "no"}:
        return False
    if lowered and set(lowered) <= {"0"}:
        return False
    if lowered and set(lowered) <= {"1"}:
        return True
    raise ValueError(f"could not parse boolean token {token!r}")


def _split_bools(
    value: object,
    *,
    separator: str | None = " ",
    expected_count: int | None = None,
) -> tuple[bool, ...]:
    tokens = _split_tokens(value, separator=separator)
    if expected_count is not None and len(tokens) == expected_count:
        return tuple(_parse_bool_token(token) for token in tokens)

    result: list[bool] = []
    for token in tokens:
        lowered = token.lower()
        if (
            expected_count is not None
            and len(tokens) < expected_count
            and set(lowered) <= {"0", "1"}
            and len(lowered) > 1
        ):
            result.extend(character == "1" for character in lowered)
        else:
            result.append(_parse_bool_token(token))
    return tuple(result)


def _durations_to_intervals(
    labels: tuple[str, ...],
    durations: tuple[float, ...],
) -> tuple[Interval, ...]:
    if len(labels) != len(durations):
        raise ValueError(
            f"duration count must match label count, got {len(labels)} labels and "
            f"{len(durations)} durations"
        )

    cursor = 0.0
    intervals: list[Interval] = []
    for label, duration in zip(labels, durations, strict=True):
        if duration < 0.0:
            raise ValueError(f"durations must be non-negative, got {duration!r}")
        start_sec = cursor
        end_sec = cursor + duration
        intervals.append(Interval(label=label, start_sec=start_sec, end_sec=end_sec))
        cursor = end_sec
    return tuple(intervals)


def _durations_to_note_intervals(
    notes: tuple[str, ...],
    durations: tuple[float, ...],
    is_slur: tuple[bool, ...] | None = None,
) -> tuple[NoteInterval, ...]:
    if len(notes) != len(durations):
        raise ValueError(
            f"note duration count must match note count, got {len(notes)} notes and "
            f"{len(durations)} durations"
        )
    if is_slur is not None and len(is_slur) != len(notes):
        raise ValueError(
            f"note slur count must match note count, got {len(notes)} notes and "
            f"{len(is_slur)} slur flags"
        )

    cursor = 0.0
    note_intervals: list[NoteInterval] = []
    for index, (note, duration) in enumerate(zip(notes, durations, strict=True)):
        if duration < 0.0:
            raise ValueError(f"durations must be non-negative, got {duration!r}")
        start_sec = cursor
        end_sec = cursor + duration
        note_intervals.append(
            NoteInterval(
                start_sec=start_sec,
                end_sec=end_sec,
                # TODO: this isn't actually a lyric sometimes (eg "B3", "G#4")
                lyric=note,
                is_slur=None if is_slur is None else is_slur[index],
            )
        )
        cursor = end_sec
    return tuple(note_intervals)


def parse_sequence_duration_row(
    row: Mapping[str, object],
    *,
    audio_path: str,
    source_dataset: str,
    raw_format: str = "sequence_duration_table",
    utterance_id_key: str = "name",
    text_key: str | None = None,
    phone_sequence_key: str = "ph_seq",
    phone_duration_key: str | None = "ph_dur",
    note_sequence_key: str | None = "note_seq",
    note_duration_key: str | None = "note_dur",
    note_slur_key: str | None = "note_slur",
    token_separator: str | None = " ",
    duration_separator: str | None = " ",
    source_paths: Mapping[str, str] | None = None,
    speaker_id: str | None = None,
    audio_sampling_rate: int | None = None,
    audio_num_samples: int | None = None,
    phone_normalizer: Callable[[str], str] | None = None,
) -> CanonicalExample:
    """Convert one sequence-duration row into the canonical schema."""
    if utterance_id_key not in row:
        raise ValueError(
            f"sequence-duration row is missing the utterance key {utterance_id_key!r}"
        )

    if phone_duration_key is None:
        raise ValueError("sequence-duration rows must provide phone durations")

    utterance_id = str(row[utterance_id_key])
    phone_sequence = _split_tokens(
        row.get(phone_sequence_key), separator=token_separator
    )
    if not phone_sequence:
        raise ValueError(
            "sequence-duration row must provide a non-empty phone sequence"
        )
    if phone_normalizer is not None:
        phone_sequence = tuple(phone_normalizer(phone) for phone in phone_sequence)

    phone_durations = _split_floats(
        row.get(phone_duration_key),
        separator=duration_separator,
    )
    if not phone_durations:
        raise ValueError("sequence-duration row must provide non-empty phone durations")

    phone_intervals = _durations_to_intervals(phone_sequence, phone_durations)

    note_sequence = (
        ()
        if note_sequence_key is None
        else _split_tokens(row.get(note_sequence_key), separator=token_separator)
    )
    note_durations = (
        ()
        if note_duration_key is None
        else _split_floats(row.get(note_duration_key), separator=duration_separator)
    )
    note_slur = (
        None
        if note_slur_key is None or row.get(note_slur_key) is None
        else _split_bools(
            row.get(note_slur_key),
            separator=duration_separator,
            expected_count=len(note_sequence),
        )
    )

    note_intervals = None
    if note_sequence and note_durations:
        note_intervals = _durations_to_note_intervals(
            note_sequence, note_durations, note_slur
        )

    consumed_keys = {
        utterance_id_key,
        phone_sequence_key,
        phone_duration_key,
    }
    if text_key is not None:
        consumed_keys.add(text_key)
    if note_sequence_key is not None:
        consumed_keys.add(note_sequence_key)
    if note_duration_key is not None:
        consumed_keys.add(note_duration_key)
    if note_slur_key is not None:
        consumed_keys.add(note_slur_key)

    metadata = {
        "raw_manifest_fields": {
            str(key): value for key, value in row.items() if key not in consumed_keys
        }
    }

    return CanonicalExample(
        audio_path=audio_path,
        utterance_id=utterance_id,
        source_dataset=source_dataset,
        raw_format=raw_format,
        speaker_id=speaker_id,
        audio_sampling_rate=audio_sampling_rate,
        audio_num_samples=audio_num_samples,
        lyrics_text=None
        if text_key is None or row.get(text_key) is None
        else str(row[text_key]),
        phone_sequence=phone_sequence,
        phone_intervals=phone_intervals,
        note_intervals=note_intervals,
        line_start_sec=0.0,
        line_end_sec=phone_intervals[-1].end_sec,
        source_paths={} if source_paths is None else dict(source_paths),
        metadata=metadata,
    )


__all__ = [
    "parse_sequence_duration_row",
]
