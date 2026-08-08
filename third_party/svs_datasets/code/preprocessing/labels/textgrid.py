"""Minimal long-form TextGrid parser for interval-tier singing corpora."""

from __future__ import annotations

from dataclasses import dataclass

from core.types import CanonicalExample, Interval


def _require_non_negative(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}")


@dataclass(frozen=True, slots=True)
class TextGridInterval:
    """One interval from a TextGrid tier, allowing empty labels."""

    text: str
    start_sec: float
    end_sec: float

    def __post_init__(self) -> None:
        _require_non_negative("TextGridInterval.start_sec", self.start_sec)
        _require_non_negative("TextGridInterval.end_sec", self.end_sec)
        if self.end_sec < self.start_sec:
            raise ValueError(
                "TextGridInterval.end_sec must be greater than or equal to start_sec"
            )

    def to_interval(self) -> Interval:
        if not self.text:
            raise ValueError("cannot convert an empty TextGrid interval into Interval")
        return Interval(label=self.text, start_sec=self.start_sec, end_sec=self.end_sec)


@dataclass(frozen=True, slots=True)
class TextGridTier:
    """One TextGrid interval tier."""

    name: str
    intervals: tuple[TextGridInterval, ...]

    def labeled_intervals(self) -> tuple[Interval, ...]:
        """Return only the non-empty labeled intervals."""
        return tuple(
            interval.to_interval() for interval in self.intervals if interval.text
        )

    @property
    def non_empty_interval_count(self) -> int:
        return sum(1 for interval in self.intervals if interval.text)


@dataclass(frozen=True, slots=True)
class TextGrid:
    """Parsed long-form TextGrid with interval tiers only."""

    tiers: tuple[TextGridTier, ...]

    def get_tier(self, name: str) -> TextGridTier | None:
        normalized_name = name.strip().lower()
        for tier in self.tiers:
            if tier.name.strip().lower() == normalized_name:
                return tier
        return None

    def choose_tier(self, preferred_names: tuple[str, ...]) -> TextGridTier | None:
        for preferred_name in preferred_names:
            tier = self.get_tier(preferred_name)
            if tier is not None:
                return tier

        non_empty_tiers = [
            tier for tier in self.tiers if tier.non_empty_interval_count > 0
        ]
        if not non_empty_tiers:
            return None
        return max(non_empty_tiers, key=lambda tier: tier.non_empty_interval_count)


def _unquote(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith('"') and stripped.endswith('"'):
        return stripped[1:-1].replace('""', '"')
    return stripped


def _parse_field_value(line: str, field_name: str) -> str:
    prefix = f"{field_name} ="
    if prefix not in line:
        raise ValueError(f"expected {field_name!r} assignment, got {line!r}")
    return line.split("=", maxsplit=1)[1].strip()


def _next_non_empty_line(lines: list[str], index: int) -> tuple[int, str]:
    while index < len(lines):
        stripped = lines[index].strip()
        if stripped:
            return index + 1, stripped
        index += 1
    raise ValueError("unexpected end of TextGrid while parsing")


def parse_textgrid(text: str) -> TextGrid:
    """Parse a long-form TextGrid payload into interval tiers.

    Supported format:
    - Praat long text format with `IntervalTier` entries
    - point tiers are ignored
    """
    lines = text.splitlines()
    index = 0
    tiers: list[TextGridTier] = []

    while index < len(lines):
        stripped = lines[index].strip()
        if stripped.startswith("item [") and stripped != "item []:":
            index, class_line = _next_non_empty_line(lines, index + 1)
            tier_class = _unquote(_parse_field_value(class_line, "class"))

            index, name_line = _next_non_empty_line(lines, index)
            tier_name = _unquote(_parse_field_value(name_line, "name"))

            index, _ = _next_non_empty_line(lines, index)  # xmin
            index, _ = _next_non_empty_line(lines, index)  # xmax
            index, size_line = _next_non_empty_line(lines, index)

            if tier_class != "IntervalTier":
                interval_count = int(
                    float(_parse_field_value(size_line, "points: size"))
                )
                for _ in range(interval_count):
                    index, _ = _next_non_empty_line(lines, index)  # points [n]:
                    index, _ = _next_non_empty_line(lines, index)  # number
                    index, _ = _next_non_empty_line(lines, index)  # mark
                continue

            interval_count = int(
                float(_parse_field_value(size_line, "intervals: size"))
            )
            intervals: list[TextGridInterval] = []
            for _ in range(interval_count):
                index, _ = _next_non_empty_line(lines, index)  # intervals [n]:
                index, xmin_line = _next_non_empty_line(lines, index)
                index, xmax_line = _next_non_empty_line(lines, index)
                index, text_line = _next_non_empty_line(lines, index)
                intervals.append(
                    TextGridInterval(
                        text=_unquote(_parse_field_value(text_line, "text")).strip(),
                        start_sec=float(_parse_field_value(xmin_line, "xmin")),
                        end_sec=float(_parse_field_value(xmax_line, "xmax")),
                    )
                )
            tiers.append(TextGridTier(name=tier_name, intervals=tuple(intervals)))
        else:
            index += 1

    if not tiers:
        raise ValueError("no interval tiers were found in the TextGrid payload")

    return TextGrid(tiers=tuple(tiers))


def parse_textgrid_example(
    text: str,
    *,
    audio_path: str,
    utterance_id: str,
    source_dataset: str,
    textgrid_path: str | None = None,
    phone_tier_names: tuple[str, ...] = ("phones", "phone", "phonemes", "phoneme"),
    word_tier_names: tuple[str, ...] = ("words", "word", "lyrics", "lyric", "text"),
    audio_sampling_rate: int | None = None,
    audio_num_samples: int | None = None,
    speaker_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> CanonicalExample:
    """Parse one TextGrid payload into a canonical full-label example."""
    textgrid = parse_textgrid(text)
    phone_tier = textgrid.choose_tier(phone_tier_names)
    if phone_tier is None:
        raise ValueError("could not identify a phone tier in the TextGrid payload")

    word_tier = textgrid.choose_tier(word_tier_names)
    if word_tier is phone_tier:
        word_tier = None

    phone_intervals = phone_tier.labeled_intervals()
    word_intervals = None if word_tier is None else word_tier.labeled_intervals()
    if not phone_intervals:
        raise ValueError(
            "the selected phone tier did not contain any labeled intervals"
        )

    source_paths = {}
    if textgrid_path is not None:
        source_paths["textgrid_path"] = textgrid_path

    metadata_dict = {} if metadata is None else dict(metadata)
    metadata_dict["phone_tier_name"] = phone_tier.name
    if word_tier is not None:
        metadata_dict["word_tier_name"] = word_tier.name

    return CanonicalExample(
        audio_path=audio_path,
        utterance_id=utterance_id,
        source_dataset=source_dataset,
        raw_format="textgrid",
        speaker_id=speaker_id,
        audio_sampling_rate=audio_sampling_rate,
        audio_num_samples=audio_num_samples,
        phone_intervals=phone_intervals,
        word_intervals=word_intervals,
        line_start_sec=phone_intervals[0].start_sec,
        line_end_sec=phone_intervals[-1].end_sec,
        source_paths=source_paths,
        metadata=metadata_dict,
    )


__all__ = [
    "TextGrid",
    "TextGridInterval",
    "TextGridTier",
    "parse_textgrid",
    "parse_textgrid_example",
]
