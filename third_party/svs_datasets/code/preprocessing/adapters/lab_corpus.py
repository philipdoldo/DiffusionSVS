"""Adapters for simple `.lab` + audio corpus families."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from preprocessing.audio import read_audio_metadata
from preprocessing.labels.lab import parse_lab_example
from preprocessing.phonesets import (
    normalize_english_dataset_phone,
    normalize_japanese_dataset_phone,
)


def _default_phone_normalizer(source_dataset: str) -> Callable[[str], str] | None:
    if source_dataset in {
        "AlexFloarea-AI-SVS",
        "Project-AIdol-Public-English-Dataset",
        "tiger_en",
    }:
        return lambda label: normalize_english_dataset_phone(
            label,
            source_dataset=source_dataset,
        )
    if source_dataset in {
        "Amaboshi_CipherDB",
        "Kurotake_Kouga_AI_Song",
        "OFUTON_P_UTAGOE_DB",
        "ONIKU_KURUMI_UTAGOE_DB",
        "PJS_corpus_ver1.1",
        "enunu_kodoku_database_20220807-2",
        # TODO: itako has some cases where it has back-to-back SP or AP (eg SP SP SP AP AP)
        # and these should be merged
        "itako_singing",
        "nit070_db",
        "ritsu",
        "tiger_jp",
        "no7singing",
    }:
        return lambda label: normalize_japanese_dataset_phone(
            label,
            source_dataset=source_dataset,
        )
    return None


def adapt_simple_lab_pair(
    *,
    audio_path: str | Path,
    label_path: str | Path,
    source_dataset: str,
    utterance_id: str | None = None,
    speaker_id: str | None = None,
    metadata: dict[str, object] | None = None,
    phone_normalizer: Callable[[str], str] | None = None,
    include_audio_metadata: bool = False,
    repair_invalid_intervals: bool = False,
) -> object:
    """Adapt a simple `.lab` + audio pair into a canonical example."""
    audio_path_obj = Path(audio_path)
    label_path_obj = Path(label_path)
    audio_metadata = (
        read_audio_metadata(audio_path_obj, resolve=False)
        if include_audio_metadata
        else None
    )
    resolved_phone_normalizer = phone_normalizer
    if resolved_phone_normalizer is None:
        resolved_phone_normalizer = _default_phone_normalizer(source_dataset)

    with label_path_obj.open("r", encoding="utf-8") as handle:
        label_text = handle.read()

    return parse_lab_example(
        label_text,
        audio_path=str(audio_path_obj),
        utterance_id=audio_path_obj.stem if utterance_id is None else utterance_id,
        source_dataset=source_dataset,
        label_path=str(label_path_obj),
        phone_normalizer=resolved_phone_normalizer,
        audio_sampling_rate=None
        if audio_metadata is None
        else audio_metadata.sample_rate,
        audio_num_samples=None
        if audio_metadata is None
        else audio_metadata.num_samples,
        speaker_id=speaker_id,
        metadata={} if metadata is None else dict(metadata),
        repair_invalid_intervals=repair_invalid_intervals,
    )


__all__ = [
    "adapt_simple_lab_pair",
]
