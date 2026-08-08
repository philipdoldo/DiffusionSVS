"""Adapters for the `popcs` TextGrid + sidecar-text family."""

from __future__ import annotations

from pathlib import Path

from core.types import CanonicalExample, Interval
from preprocessing.audio import read_audio_metadata
from preprocessing.labels.textgrid import TextGrid, TextGridTier, parse_textgrid
from preprocessing.phonesets import normalize_mandarin_dataset_phone

_POPCS_SPECIAL_TOKENS = frozenset({"<BOS>", "<EOS>", "<SEP>"})


def _split_popcs_lyrics(text: str) -> tuple[str, ...]:
    return tuple(
        segment for segment in (part.strip() for part in text.split("@")) if segment
    )


def _split_popcs_phone_sidecar(text: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for raw_token in text.split():
        token = raw_token.strip()
        if not token or token == "|" or token in _POPCS_SPECIAL_TOKENS:
            continue
        tokens.append(token)
    return tuple(tokens)


def _normalize_popcs_phone_intervals(
    phone_tier: TextGridTier,
) -> tuple[Interval, ...]:
    return tuple(
        Interval(
            label=normalize_mandarin_dataset_phone(
                interval.text or "SP", source_dataset="popcs"
            ),
            start_sec=interval.start_sec,
            end_sec=interval.end_sec,
        )
        for interval in phone_tier.intervals
    )


def _choose_popcs_tiers(
    textgrid: TextGrid,
) -> tuple[tuple[int, TextGridTier], tuple[int, TextGridTier] | None]:
    ranked_non_empty_tiers = sorted(
        (
            (index, tier)
            for index, tier in enumerate(textgrid.tiers)
            if tier.non_empty_interval_count > 0
        ),
        key=lambda item: item[1].non_empty_interval_count,
        reverse=True,
    )
    if not ranked_non_empty_tiers:
        raise ValueError("popcs TextGrid did not expose any labeled intervals")

    phone_tier = ranked_non_empty_tiers[0]
    syllable_tier = (
        None if len(ranked_non_empty_tiers) < 2 else ranked_non_empty_tiers[1]
    )
    return phone_tier, syllable_tier


def adapt_popcs_example(
    *,
    textgrid_path: str | Path,
    audio_path: str | Path,
    lyrics_path: str | Path | None = None,
    phone_text_path: str | Path | None = None,
    source_dataset: str = "popcs",
    include_audio_metadata: bool = False,
) -> CanonicalExample:
    """Adapt one `popcs` example into the canonical full-label schema."""
    textgrid_path_obj = Path(textgrid_path)
    lyrics_path_obj = None if lyrics_path is None else Path(lyrics_path)
    phone_text_path_obj = None if phone_text_path is None else Path(phone_text_path)
    audio_path_obj = Path(audio_path)

    textgrid_text = textgrid_path_obj.read_text(encoding="utf-8")
    lyrics_text = (
        None if lyrics_path_obj is None else lyrics_path_obj.read_text(encoding="utf-8").strip()
    )
    phone_text = (
        None
        if phone_text_path_obj is None
        else phone_text_path_obj.read_text(encoding="utf-8").strip()
    )

    textgrid = parse_textgrid(textgrid_text)
    (phone_tier_index, phone_tier), syllable_tier_info = _choose_popcs_tiers(textgrid)

    phone_intervals = _normalize_popcs_phone_intervals(phone_tier)
    if not phone_intervals:
        raise ValueError("popcs phone tier did not contain any labeled intervals")

    word_intervals = None
    metadata: dict[str, object] = {
        "phone_tier_name": phone_tier.name,
        "phone_tier_index": phone_tier_index,
    }
    if lyrics_text is not None:
        metadata["lyrics_segments"] = list(_split_popcs_lyrics(lyrics_text))

    if syllable_tier_info is not None:
        syllable_tier_index, syllable_tier = syllable_tier_info
        word_intervals = syllable_tier.labeled_intervals()
        metadata["word_tier_name"] = syllable_tier.name
        metadata["word_tier_index"] = syllable_tier_index
        metadata["word_tier_semantics"] = "syllable_pinyin_units"

    sidecar_phone_tokens = None
    if phone_text is not None:
        sidecar_phone_tokens = tuple(
            normalize_mandarin_dataset_phone(token, source_dataset="popcs")
            for token in _split_popcs_phone_sidecar(phone_text)
        )
        metadata["phone_sidecar_count"] = len(sidecar_phone_tokens)
        metadata["textgrid_phone_count"] = len(phone_intervals)
        metadata["phone_sidecar_matches_textgrid"] = len(sidecar_phone_tokens) == len(
            phone_intervals
        )

    phone_sequence = tuple(interval.label for interval in phone_intervals)
    clip_stem = audio_path_obj.stem.removesuffix("_wf0")
    audio_metadata = (
        read_audio_metadata(audio_path_obj, resolve=False)
        if include_audio_metadata
        else None
    )

    return CanonicalExample(
        audio_path=str(audio_path_obj),
        utterance_id=f"{audio_path_obj.parent.name}/{clip_stem}",
        source_dataset=source_dataset,
        raw_format=(
            "popcs_textgrid_sidecars"
            if lyrics_path_obj is not None or phone_text_path_obj is not None
            else "popcs_textgrid"
        ),
        audio_sampling_rate=None if audio_metadata is None else audio_metadata.sample_rate,
        audio_num_samples=None if audio_metadata is None else audio_metadata.num_samples,
        lyrics_text=lyrics_text,
        phone_sequence=phone_sequence,
        phone_intervals=phone_intervals,
        word_intervals=word_intervals,
        line_start_sec=phone_intervals[0].start_sec,
        line_end_sec=phone_intervals[-1].end_sec,
        source_paths={
            "textgrid_path": str(textgrid_path_obj),
            **({} if lyrics_path_obj is None else {"lyrics_path": str(lyrics_path_obj)}),
            **({} if phone_text_path_obj is None else {"phone_text_path": str(phone_text_path_obj)}),
        },
        metadata=metadata,
    )


__all__ = [
    "adapt_popcs_example",
]
