"""Adapter for the `no7singing` corpus with multiple waveform variants."""

from __future__ import annotations

from pathlib import Path

from preprocessing.adapters.lab_corpus import adapt_simple_lab_pair

NO7SINGING_AUDIO_VARIANTS = (
    "wav_PT",
    "wav_P",
    "wav_T",
    "wav_O_re",
    "wav_O",
)
DEFAULT_NO7SINGING_AUDIO_VARIANT = "wav_PT"

_NO7SINGING_AUDIO_VARIANT_ALIASES = {
    "pt": "wav_PT",
    "wav_pt": "wav_PT",
    "p": "wav_P",
    "wav_p": "wav_P",
    "t": "wav_T",
    "wav_t": "wav_T",
    "o_re": "wav_O_re",
    "wav_o_re": "wav_O_re",
    "ore": "wav_O_re",
    "o": "wav_O",
    "wav_o": "wav_O",
}


def normalize_no7singing_audio_variant(audio_variant: str) -> str:
    """Return a canonical `no7singing` waveform-variant directory name."""
    normalized = audio_variant.strip()
    if normalized in NO7SINGING_AUDIO_VARIANTS:
        return normalized

    alias_key = normalized.lower().replace("-", "_")
    try:
        return _NO7SINGING_AUDIO_VARIANT_ALIASES[alias_key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported no7singing audio variant {audio_variant!r}; "
            f"expected one of {NO7SINGING_AUDIO_VARIANTS!r}"
        ) from exc


def resolve_no7singing_paths(
    *,
    dataset_root: str | Path,
    utterance_id: str,
    audio_variant: str = DEFAULT_NO7SINGING_AUDIO_VARIANT,
    audio_suffix: str = ".wav",
) -> dict[str, Path]:
    """Resolve the dataset-specific sidecar paths for one `no7singing` item."""
    dataset_root_obj = Path(dataset_root)
    normalized_variant = normalize_no7singing_audio_variant(audio_variant)
    stem = Path(utterance_id).stem

    paths = {
        "audio_path": dataset_root_obj / normalized_variant / f"{stem}{audio_suffix}",
        "label_path": dataset_root_obj / "mono_label" / f"{stem}.lab",
        "midi_path": dataset_root_obj / "midi_label" / f"{stem}.mid",
        "lyrics_pdf_path": dataset_root_obj / "lyric" / f"{stem}.pdf",
        "lyrics_docx_path": dataset_root_obj / "lyric" / f"{stem}.docx",
        "pitchshift_path": dataset_root_obj / "pitchshift.txt",
        "musicxml_path": dataset_root_obj / "musicxml" / f"{stem}.musicxml",
    }
    return paths


def adapt_no7singing_example(
    *,
    dataset_root: str | Path,
    utterance_id: str,
    audio_variant: str = DEFAULT_NO7SINGING_AUDIO_VARIANT,
    audio_suffix: str = ".wav",
    source_dataset: str = "no7singing",
    speaker_id: str = "no7",
    include_audio_metadata: bool = False,
) -> object:
    """Adapt one `no7singing` item into a canonical example."""
    normalized_variant = normalize_no7singing_audio_variant(audio_variant)
    paths = resolve_no7singing_paths(
        dataset_root=dataset_root,
        utterance_id=utterance_id,
        audio_variant=normalized_variant,
        audio_suffix=audio_suffix,
    )
    stem = Path(utterance_id).stem

    example = adapt_simple_lab_pair(
        audio_path=paths["audio_path"],
        label_path=paths["label_path"],
        source_dataset=source_dataset,
        utterance_id=f"{stem}/{normalized_variant}",
        speaker_id=speaker_id,
        include_audio_metadata=include_audio_metadata,
        metadata={
            "audio_variant": normalized_variant,
            "available_audio_variants": NO7SINGING_AUDIO_VARIANTS,
            # Upstream documentation says labels correspond to the PT waveform.
            "preferred_supervised_audio_variant": DEFAULT_NO7SINGING_AUDIO_VARIANT,
        },
    )
    example.source_paths["midi_path"] = str(paths["midi_path"])
    example.source_paths["pitchshift_path"] = str(paths["pitchshift_path"])
    example.source_paths["lyrics_pdf_path"] = str(paths["lyrics_pdf_path"])
    example.source_paths["lyrics_docx_path"] = str(paths["lyrics_docx_path"])
    example.source_paths["musicxml_path"] = str(paths["musicxml_path"])
    return example


__all__ = [
    "DEFAULT_NO7SINGING_AUDIO_VARIANT",
    "NO7SINGING_AUDIO_VARIANTS",
    "adapt_no7singing_example",
    "normalize_no7singing_audio_variant",
    "resolve_no7singing_paths",
]
