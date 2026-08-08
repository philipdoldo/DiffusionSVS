from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Any, TypeAlias

Seconds: TypeAlias = float
FrameIndex: TypeAlias = int
PhoneId: TypeAlias = int


def _validate_time(name: str, value: float) -> None:
    if not isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


def _validate_interval_order(name: str, start_sec: float, end_sec: float) -> None:
    _validate_time(f"{name}.start_sec", start_sec)
    _validate_time(f"{name}.end_sec", end_sec)
    if end_sec < start_sec:
        raise ValueError(
            f"{name}.end_sec must be greater than or equal to start_sec, "
            f"got {start_sec!r} -> {end_sec!r}"
        )


@dataclass(frozen=True, slots=True)
class Interval:
    """A labeled half-open-ish interval in seconds."""

    label: str
    start_sec: Seconds
    end_sec: Seconds

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("interval label must not be empty")
        _validate_interval_order("interval", self.start_sec, self.end_sec)

    @property
    def duration_sec(self) -> Seconds:
        """Return the interval duration in seconds."""
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict[str, object]:
        """Serialize the interval to a JSON-friendly mapping."""
        return {
            "label": self.label,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Interval:
        """Construct an interval from a serialized mapping."""
        return cls(
            label=str(data["label"]),
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
        )


@dataclass(frozen=True, slots=True)
class NoteInterval:
    """A note-aligned interval optionally associated with one phonetic unit."""

    start_sec: Seconds
    end_sec: Seconds
    pitch_midi: int | None = None
    lyric: str | None = None
    is_slur: bool | None = None

    def __post_init__(self) -> None:
        _validate_interval_order("note_interval", self.start_sec, self.end_sec)
        if self.pitch_midi is not None and self.pitch_midi < 0:
            raise ValueError(
                f"pitch_midi must be non-negative when provided, got {self.pitch_midi!r}"
            )

    @property
    def duration_sec(self) -> Seconds:
        """Return the note duration in seconds."""
        return self.end_sec - self.start_sec

    def to_dict(self) -> dict[str, object]:
        """Serialize the note interval to a JSON-friendly mapping."""
        return {
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
            "pitch_midi": self.pitch_midi,
            "lyric": self.lyric,
            "is_slur": self.is_slur,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> NoteInterval:
        """Construct a note interval from a serialized mapping."""
        pitch = data.get("pitch_midi")
        return cls(
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            pitch_midi=None if pitch is None else int(pitch),
            lyric=None if data.get("lyric") is None else str(data["lyric"]),
            is_slur=None if data.get("is_slur") is None else bool(data["is_slur"]),
        )


def _validate_monotonic_intervals(
    name: str,
    intervals: tuple[Interval, ...] | tuple[NoteInterval, ...] | None,
) -> None:
    if intervals is None:
        return

    previous_end: float | None = None
    for index, interval in enumerate(intervals):
        if previous_end is not None and interval.start_sec < previous_end:
            raise ValueError(
                f"{name}[{index}] starts before the previous interval ends: "
                f"{interval.start_sec!r} < {previous_end!r}"
            )
        previous_end = interval.end_sec


@dataclass(slots=True)
class CanonicalExample:
    """Canonical corpus object shared across all raw dataset adapters."""

    audio_path: str
    utterance_id: str
    source_dataset: str
    raw_format: str
    speaker_id: str | None = None
    audio_sampling_rate: int | None = None
    audio_num_samples: int | None = None
    lyrics_text: str | None = None
    phone_sequence: tuple[str, ...] | list[str] | None = None
    phone_intervals: tuple[Interval, ...] | list[Interval] | None = None
    word_intervals: tuple[Interval, ...] | list[Interval] | None = None
    note_intervals: tuple[NoteInterval, ...] | list[NoteInterval] | None = None
    line_start_sec: Seconds | None = None
    line_end_sec: Seconds | None = None
    source_paths: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.audio_path:
            raise ValueError("audio_path must not be empty")
        if not self.utterance_id:
            raise ValueError("utterance_id must not be empty")
        if not self.source_dataset:
            raise ValueError("source_dataset must not be empty")
        if not self.raw_format:
            raise ValueError("raw_format must not be empty")

        if self.audio_sampling_rate is not None and self.audio_sampling_rate <= 0:
            raise ValueError(
                f"audio_sampling_rate must be positive, got {self.audio_sampling_rate!r}"
            )
        if self.audio_num_samples is not None and self.audio_num_samples < 0:
            raise ValueError(
                f"audio_num_samples must be non-negative, got {self.audio_num_samples!r}"
            )

        if self.line_start_sec is not None:
            _validate_time("line_start_sec", self.line_start_sec)
        if self.line_end_sec is not None:
            _validate_time("line_end_sec", self.line_end_sec)
        if (
            self.line_start_sec is not None
            and self.line_end_sec is not None
            and self.line_end_sec < self.line_start_sec
        ):
            raise ValueError(
                "line_end_sec must be greater than or equal to line_start_sec"
            )

        if self.phone_sequence is not None:
            self.phone_sequence = tuple(self.phone_sequence)
        if self.phone_intervals is not None:
            self.phone_intervals = tuple(self.phone_intervals)
        if self.word_intervals is not None:
            self.word_intervals = tuple(self.word_intervals)
        if self.note_intervals is not None:
            self.note_intervals = tuple(self.note_intervals)

        self.source_paths = dict(self.source_paths)
        self.metadata = dict(self.metadata)

        self.validate()

    @property
    def effective_phone_sequence(self) -> tuple[str, ...] | None:
        """Return an explicit phone sequence or derive it from phone intervals."""
        if self.phone_sequence is not None:
            return self.phone_sequence
        if self.phone_intervals is not None:
            return tuple(interval.label for interval in self.phone_intervals)
        return None

    @property
    def has_phone_boundaries(self) -> bool:
        """Return whether the example includes explicit phone intervals."""
        return self.phone_intervals is not None

    @property
    def duration_sec(self) -> Seconds | None:
        """Best-effort duration estimate for the example."""
        if self.line_start_sec is not None and self.line_end_sec is not None:
            return self.line_end_sec - self.line_start_sec
        if self.phone_intervals:
            return self.phone_intervals[-1].end_sec - self.phone_intervals[0].start_sec
        if self.word_intervals:
            return self.word_intervals[-1].end_sec - self.word_intervals[0].start_sec
        if self.note_intervals:
            return self.note_intervals[-1].end_sec - self.note_intervals[0].start_sec
        return None

    def validate(self) -> None:
        """Run lightweight consistency checks on the canonical example."""
        if not self.phone_intervals:
            raise ValueError("canonical singing examples must include phone_intervals")

        _validate_monotonic_intervals("phone_intervals", self.phone_intervals)
        _validate_monotonic_intervals("word_intervals", self.word_intervals)
        _validate_monotonic_intervals("note_intervals", self.note_intervals)

    def to_dict(self) -> dict[str, object]:
        """Serialize the canonical example to a JSON-friendly mapping."""
        return {
            "audio_path": self.audio_path,
            "utterance_id": self.utterance_id,
            "source_dataset": self.source_dataset,
            "raw_format": self.raw_format,
            "speaker_id": self.speaker_id,
            "audio_sampling_rate": self.audio_sampling_rate,
            "audio_num_samples": self.audio_num_samples,
            "lyrics_text": self.lyrics_text,
            "phone_sequence": (
                None if self.phone_sequence is None else list(self.phone_sequence)
            ),
            "phone_intervals": (
                None
                if self.phone_intervals is None
                else [interval.to_dict() for interval in self.phone_intervals]
            ),
            "word_intervals": (
                None
                if self.word_intervals is None
                else [interval.to_dict() for interval in self.word_intervals]
            ),
            "note_intervals": (
                None
                if self.note_intervals is None
                else [interval.to_dict() for interval in self.note_intervals]
            ),
            "line_start_sec": self.line_start_sec,
            "line_end_sec": self.line_end_sec,
            "source_paths": dict(self.source_paths),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> CanonicalExample:
        """Construct a canonical example from a serialized mapping."""
        phone_intervals = data.get("phone_intervals")
        word_intervals = data.get("word_intervals")
        note_intervals = data.get("note_intervals")

        return cls(
            audio_path=str(data["audio_path"]),
            utterance_id=str(data["utterance_id"]),
            source_dataset=str(data["source_dataset"]),
            raw_format=str(data["raw_format"]),
            speaker_id=None
            if data.get("speaker_id") is None
            else str(data["speaker_id"]),
            audio_sampling_rate=(
                None
                if data.get("audio_sampling_rate") is None
                else int(data["audio_sampling_rate"])
            ),
            audio_num_samples=(
                None
                if data.get("audio_num_samples") is None
                else int(data["audio_num_samples"])
            ),
            lyrics_text=None
            if data.get("lyrics_text") is None
            else str(data["lyrics_text"]),
            phone_sequence=(
                None
                if data.get("phone_sequence") is None
                else tuple(str(token) for token in data["phone_sequence"])
            ),
            phone_intervals=(
                None
                if phone_intervals is None
                else tuple(Interval.from_dict(item) for item in phone_intervals)
            ),
            word_intervals=(
                None
                if word_intervals is None
                else tuple(Interval.from_dict(item) for item in word_intervals)
            ),
            note_intervals=(
                None
                if note_intervals is None
                else tuple(NoteInterval.from_dict(item) for item in note_intervals)
            ),
            line_start_sec=(
                None
                if data.get("line_start_sec") is None
                else float(data["line_start_sec"])
            ),
            line_end_sec=(
                None
                if data.get("line_end_sec") is None
                else float(data["line_end_sec"])
            ),
            source_paths=(
                {} if data.get("source_paths") is None else dict(data["source_paths"])
            ),
            metadata=({} if data.get("metadata") is None else dict(data["metadata"])),
        )

    def __str__(self) -> str:
        return (
            f"CanonicalExample(\n"
            f"  utterance_id={self.utterance_id!r},\n"
            f"  source_dataset={self.source_dataset!r},\n"
            f"  raw_format={self.raw_format!r},\n"
            f"  audio_path={self.audio_path!r},\n"
            "  intervals=[\n    "
            + ",\n    ".join(str(interval) for interval in (self.phone_intervals or []))
            + "\n  ],\n"
            "  word_intervals=[\n    "
            + ",\n    ".join(str(interval) for interval in (self.word_intervals or []))
            + "\n  ],\n"
            "  note_intervals=[\n    "
            + ",\n    ".join(str(interval) for interval in (self.note_intervals or []))
            + "\n  ]\n"
            ")"
        )


__all__ = [
    "CanonicalExample",
    "FrameIndex",
    "Interval",
    "NoteInterval",
    "PhoneId",
    "Seconds",
]
