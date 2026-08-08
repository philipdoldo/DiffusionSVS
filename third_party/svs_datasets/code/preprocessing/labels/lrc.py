"""Inference-time `.lrc` parsing and lyric-window helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

_TIMESTAMP_PATTERN = re.compile(r"\[(\d+):(\d+(?:\.\d+)?)\]")
_TAG_PATTERN = re.compile(r"^\[([A-Za-z]+):(.*)\]$")


def _parse_timestamp_minutes_seconds(minutes_raw: str, seconds_raw: str) -> float:
    minutes = int(minutes_raw)
    seconds = float(seconds_raw)
    if seconds < 0.0 or seconds >= 60.0:
        raise ValueError(f"LRC seconds field must be in [0, 60), got {seconds!r}")
    return minutes * 60.0 + seconds


@dataclass(frozen=True, slots=True)
class LrcCue:
    """One timestamped lyric cue parsed from `.lrc`.

    This is an inference-time object only. It is not training supervision.
    """

    start_sec: float
    text: str

    def __post_init__(self) -> None:
        if self.start_sec < 0.0:
            raise ValueError(
                f"LrcCue.start_sec must be non-negative, got {self.start_sec!r}"
            )


@dataclass(frozen=True, slots=True)
class LrcLineWindow:
    """One inference-time lyric line window derived from LRC cues."""

    start_sec: float
    end_sec: float | None
    text: str
    cue_index: int

    def __post_init__(self) -> None:
        if self.start_sec < 0.0:
            raise ValueError(
                f"LrcLineWindow.start_sec must be non-negative, got {self.start_sec!r}"
            )
        if self.end_sec is not None and self.end_sec < self.start_sec:
            raise ValueError(
                "LrcLineWindow.end_sec must be greater than or equal to start_sec"
            )

    @property
    def duration_sec(self) -> float | None:
        if self.end_sec is None:
            return None
        return self.end_sec - self.start_sec


@dataclass(frozen=True, slots=True)
class LrcDocument:
    """Parsed `.lrc` document containing metadata and inference cues."""

    cues: tuple[LrcCue, ...]
    metadata: dict[str, str]
    offset_ms: int = 0

    def __post_init__(self) -> None:
        sorted_cues = tuple(sorted(self.cues, key=lambda cue: cue.start_sec))
        object.__setattr__(self, "cues", sorted_cues)
        object.__setattr__(self, "metadata", dict(self.metadata))

    def to_line_windows(
        self,
        *,
        song_duration_sec: float | None = None,
        trailing_window_sec: float | None = None,
        drop_empty_text: bool = True,
    ) -> tuple[LrcLineWindow, ...]:
        """Convert cues into inference windows.

        End times are inferred from the next cue. For the last cue:
        - use `song_duration_sec` when available
        - else use `start_sec + trailing_window_sec` when provided
        - else leave `end_sec` as `None`
        """
        if song_duration_sec is not None and song_duration_sec < 0.0:
            raise ValueError(
                f"song_duration_sec must be non-negative, got {song_duration_sec!r}"
            )
        if trailing_window_sec is not None and trailing_window_sec < 0.0:
            raise ValueError(
                f"trailing_window_sec must be non-negative, got {trailing_window_sec!r}"
            )

        filtered_cues = tuple(
            cue for cue in self.cues if cue.text or not drop_empty_text
        )
        windows: list[LrcLineWindow] = []
        for index, cue in enumerate(filtered_cues):
            if index + 1 < len(filtered_cues):
                end_sec = filtered_cues[index + 1].start_sec
            elif song_duration_sec is not None:
                end_sec = song_duration_sec
            elif trailing_window_sec is not None:
                end_sec = cue.start_sec + trailing_window_sec
            else:
                end_sec = None
            windows.append(
                LrcLineWindow(
                    start_sec=cue.start_sec,
                    end_sec=end_sec,
                    text=cue.text,
                    cue_index=index,
                )
            )
        return tuple(windows)


def parse_lrc(text: str) -> LrcDocument:
    """Parse an `.lrc` payload into inference-time lyric cues.

    Supported constructs:
    - timestamped lyric lines such as `[00:12.34]hello`
    - repeated timestamps on the same line
    - simple metadata tags such as `[ar:artist]`, `[ti:title]`, `[offset:500]`

    The `offset` tag is interpreted as milliseconds and applied to all cues.
    """
    metadata: dict[str, str] = {}
    offset_ms = 0
    raw_cues: list[tuple[float, str]] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue

        tag_match = _TAG_PATTERN.match(stripped)
        timestamp_matches = list(_TIMESTAMP_PATTERN.finditer(stripped))
        if tag_match and not timestamp_matches:
            tag_name = tag_match.group(1).strip().lower()
            tag_value = tag_match.group(2).strip()
            metadata[tag_name] = tag_value
            if tag_name == "offset":
                try:
                    offset_ms = int(tag_value)
                except ValueError as exc:
                    raise ValueError(
                        f"LRC offset tag must be an integer number of milliseconds, got "
                        f"{tag_value!r} on line {line_number}"
                    ) from exc
            continue

        if not timestamp_matches:
            raise ValueError(
                f"LRC line {line_number} must contain at least one timestamp, got {raw_line!r}"
            )

        lyric_text = _TIMESTAMP_PATTERN.sub("", stripped).strip()
        for match in timestamp_matches:
            start_sec = _parse_timestamp_minutes_seconds(match.group(1), match.group(2))
            raw_cues.append((start_sec, lyric_text))

    adjusted_cues = []
    offset_sec = offset_ms / 1000.0
    for start_sec, lyric_text in raw_cues:
        adjusted_start_sec = start_sec + offset_sec
        if adjusted_start_sec < 0.0:
            raise ValueError(
                f"LRC offset would make a cue timestamp negative: {adjusted_start_sec!r}"
            )
        adjusted_cues.append(LrcCue(start_sec=adjusted_start_sec, text=lyric_text))

    return LrcDocument(
        cues=tuple(adjusted_cues), metadata=metadata, offset_ms=offset_ms
    )


__all__ = [
    "LrcCue",
    "LrcDocument",
    "LrcLineWindow",
    "parse_lrc",
]
