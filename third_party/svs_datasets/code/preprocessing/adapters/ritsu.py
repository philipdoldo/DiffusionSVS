"""Adapter for the multi-variant `ritsu` singing databases."""

from __future__ import annotations

from pathlib import Path

from preprocessing.adapters.lab_corpus import adapt_simple_lab_pair

RITSU_VARIANTS = (
    "base",
    "ver2",
    "ver2_0_2",
    "normal",
    "soft",
)
DEFAULT_RITSU_VARIANT = "ver2_0_2"

_RITSU_VARIANT_LAYOUTS = {
    "base": {
        "root_dir": "「波音リツ」歌声データベース",
        "database_dir": "DATABASE",
        "speaker_id": "namine_ritsu_base",
    },
    "ver2": {
        "root_dir": "「波音リツ」歌声データベースVer2",
        "database_dir": "DATABASE",
        "speaker_id": "namine_ritsu_ver2",
    },
    "ver2_0_2": {
        "root_dir": "「波音リツ」歌声データベースVer2.0.2",
        "database_dir": "DATABASE",
        "speaker_id": "namine_ritsu_ver2_0_2",
    },
    "normal": {
        "root_dir": "「波音リツ Normal」歌声データベース",
        "database_dir": "DATABASE_normal",
        "speaker_id": "namine_ritsu_normal",
    },
    "soft": {
        "root_dir": "「波音リツ Soft」歌声データベース",
        "database_dir": "DATABASE_soft",
        "speaker_id": "namine_ritsu_soft",
    },
}

_RITSU_VARIANT_ALIASES = {
    "1": "base",
    "v1": "base",
    "ver1": "base",
    "ver_1": "base",
    "v2": "ver2",
    "ver2": "ver2",
    "ver_2": "ver2",
    "2.0.2": "ver2_0_2",
    "v2.0.2": "ver2_0_2",
    "ver2.0.2": "ver2_0_2",
    "ver2_0_2": "ver2_0_2",
}


def normalize_ritsu_variant(variant: str) -> str:
    """Return a canonical `ritsu` variant name."""
    normalized = variant.strip()
    if normalized in RITSU_VARIANTS:
        return normalized

    alias_key = normalized.lower().replace("-", "_")
    try:
        return _RITSU_VARIANT_ALIASES[alias_key]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported ritsu variant {variant!r}; expected one of {RITSU_VARIANTS!r}"
        ) from exc


def resolve_ritsu_paths(
    *,
    dataset_root: str | Path,
    song_id: str,
    variant: str = DEFAULT_RITSU_VARIANT,
    audio_suffix: str = ".wav",
) -> dict[str, Path]:
    """Resolve the dataset-specific sidecar paths for one `ritsu` song."""
    dataset_root_obj = Path(dataset_root)
    normalized_variant = normalize_ritsu_variant(variant)
    layout = _RITSU_VARIANT_LAYOUTS[normalized_variant]
    stem = Path(song_id).stem

    variant_root = dataset_root_obj / layout["root_dir"]
    song_root = variant_root / layout["database_dir"] / stem

    paths = {
        "audio_path": song_root / f"{stem}{audio_suffix}",
        "label_path": song_root / f"{stem}.lab",
        "midi_path": song_root / f"{stem}.mid",
        "musicxml_path": song_root / f"{stem}.musicxml",
        "ust_path": song_root / f"{stem}.ust",
        "readme_path": variant_root / "readme.txt",
        "release_notes_path": variant_root / "Release_notes.txt",
        "kana2phonemes_path": variant_root / "kana2phonemes_002_oto2lab.table",
        "output_txt_path": variant_root / "output.txt",
        "output_csv_path": variant_root / "output2.csv",
        "ust_bpm_and_range_path": variant_root / "ust_bpm_and_range.csv",
    }
    return paths


def iter_ritsu_song_ids(
    *,
    dataset_root: str | Path,
    variant: str = DEFAULT_RITSU_VARIANT,
) -> tuple[str, ...]:
    """Return song IDs discoverable in one `ritsu` database variant."""
    dataset_root_obj = Path(dataset_root)
    normalized_variant = normalize_ritsu_variant(variant)
    layout = _RITSU_VARIANT_LAYOUTS[normalized_variant]
    database_root = (
        dataset_root_obj / layout["root_dir"] / layout["database_dir"]
    )
    return tuple(
        label_path.parent.name
        for label_path in sorted(database_root.glob("*/*.lab"))
    )


def adapt_ritsu_example(
    *,
    dataset_root: str | Path,
    song_id: str,
    variant: str = DEFAULT_RITSU_VARIANT,
    audio_suffix: str = ".wav",
    source_dataset: str = "ritsu",
    include_audio_metadata: bool = False,
) -> object:
    """Adapt one `ritsu` song from a selected database variant."""
    normalized_variant = normalize_ritsu_variant(variant)
    layout = _RITSU_VARIANT_LAYOUTS[normalized_variant]
    paths = resolve_ritsu_paths(
        dataset_root=dataset_root,
        song_id=song_id,
        variant=normalized_variant,
        audio_suffix=audio_suffix,
    )
    stem = Path(song_id).stem

    example = adapt_simple_lab_pair(
        audio_path=paths["audio_path"],
        label_path=paths["label_path"],
        source_dataset=source_dataset,
        utterance_id=f"{normalized_variant}/{stem}",
        speaker_id=str(layout["speaker_id"]),
        include_audio_metadata=include_audio_metadata,
        metadata={
            "database_variant": normalized_variant,
            "available_variants": RITSU_VARIANTS,
        },
    )
    for key, value in paths.items():
        if key.endswith("_path"):
            example.source_paths[key] = str(value)
    return example


__all__ = [
    "DEFAULT_RITSU_VARIANT",
    "RITSU_VARIANTS",
    "adapt_ritsu_example",
    "iter_ritsu_song_ids",
    "normalize_ritsu_variant",
    "resolve_ritsu_paths",
]
