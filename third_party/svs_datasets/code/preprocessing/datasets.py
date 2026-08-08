"""Public dataset-loading entrypoints."""

from __future__ import annotations

from collections.abc import Callable
import os
from pathlib import Path

from tqdm import tqdm

from core.compat import StrEnum
from core.types import CanonicalExample
from preprocessing.adapters.alex_public import (
    adapt_alex_public_row,
    load_alex_public_rows,
)
from preprocessing.adapters.gtsinger import adapt_gtsinger_segment
from preprocessing.adapters.lab_corpus import adapt_simple_lab_pair
from preprocessing.adapters.m4singer import adapt_m4singer_item, load_m4singer_metadata
from preprocessing.adapters.ngyy import adapt_ngyy_row, load_ngyy_rows
from preprocessing.adapters.no7singing import (
    DEFAULT_NO7SINGING_AUDIO_VARIANT,
    adapt_no7singing_example,
    normalize_no7singing_audio_variant,
)
from preprocessing.adapters.opencpop import adapt_opencpop_line
from preprocessing.adapters.popcs import adapt_popcs_example
from preprocessing.adapters.ritsu import (
    DEFAULT_RITSU_VARIANT,
    adapt_ritsu_example,
    iter_ritsu_song_ids,
    resolve_ritsu_paths,
)
from preprocessing.audio import resolve_audio_path


class Dataset(StrEnum):
    ALEX_FLOAREA_AI_SVS = "AlexFloarea-AI-SVS"
    ALEX_FLOAREA_EN_PUBLIC = "Alex_Floarea_EN_Public_Corpus"
    PROJECT_AIDOL_PUBLIC_ENGLISH = "Project-AIdol-Public-English-Dataset"
    NGYY = "NGYY_ENG_Dataset"
    OPEN_CPOP = "opencpop"
    M4SINGER = "m4singer"
    NO7SINGING = "no7singing"
    RITSU = "ritsu"
    POPCS = "popcs"
    SUNG_AND_SPOKEN = "sungandspoken"
    GTSINGER_CHINESE = "GTSinger_Chinese"
    GTSINGER_ENGLISH = "GTSinger_English"
    GTSINGER_JAPANESE = "GTSinger_Japanese"
    AMABOSHI_CIPHERDB = "Amaboshi_CipherDB"
    KUROTAKE_KOUGA_AI_SONG = "Kurotake_Kouga_AI_Song"
    OFUTON_P_UTAGOE_DB = "OFUTON_P_UTAGOE_DB"
    ONIKU_KURUMI_UTAGOE_DB = "ONIKU_KURUMI_UTAGOE_DB"
    PJS_CORPUS = "PJS_corpus_ver1.1"
    ENUNU_KODOKU = "enunu_kodoku_database_20220807-2"
    ITAKO_SINGING = "itako_singing"
    NIT070_DB = "nit070_db"
    TIGER_EN = "tiger_en"
    TIGER_JP = "tiger_jp"


_SKIPPED_SCAN_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
    }
)


def _dataset_alias_key(value: object) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


_DATASET_ALIASES = {
    _dataset_alias_key(dataset.name): dataset
    for dataset in Dataset
}
_DATASET_ALIASES.update(
    {
        _dataset_alias_key(dataset.value): dataset
        for dataset in Dataset
    }
)
_DATASET_ALIASES.update(
    {
        "alex_public": Dataset.ALEX_FLOAREA_EN_PUBLIC,
        "alex_floarea_public": Dataset.ALEX_FLOAREA_EN_PUBLIC,
        "gtsinger_cn": Dataset.GTSINGER_CHINESE,
        "gtsinger_en": Dataset.GTSINGER_ENGLISH,
        "gtsinger_jp": Dataset.GTSINGER_JAPANESE,
        "m4": Dataset.M4SINGER,
        "open_cpop": Dataset.OPEN_CPOP,
    }
)


def normalize_dataset(dataset: Dataset | str) -> Dataset:
    """Normalize a dataset enum, enum value, enum name, or common alias."""
    if isinstance(dataset, Dataset):
        return dataset
    try:
        return Dataset(str(dataset))
    except ValueError:
        pass
    key = _dataset_alias_key(dataset)
    try:
        return _DATASET_ALIASES[key]
    except KeyError as exc:
        valid = ", ".join(dataset.value for dataset in Dataset)
        raise ValueError(f"unknown dataset {dataset!r}; expected one of: {valid}") from exc


def _read_nonempty_lines(path: Path) -> tuple[str, ...]:
    return tuple(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def _path_with_suffix(path: Path, suffix: str) -> Path:
    return path.with_suffix(suffix)


def _resolved_first_audio_path(candidate_path: Path) -> Path:
    return resolve_audio_path(candidate_path)


def _first_audio_suffix(candidate_path: Path) -> str:
    return _resolved_first_audio_path(candidate_path).suffix


def _first_audio_dir_name_and_suffix(candidate_path: Path) -> tuple[str, str]:
    audio_path = _resolved_first_audio_path(candidate_path)
    return audio_path.parent.name, audio_path.suffix


def _iter_files(
    root: Path,
    suffixes: tuple[str, ...],
    *,
    max_depth: int | None = None,
) -> tuple[Path, ...]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    paths: list[Path] = []
    for directory, dirnames, filenames in os.walk(root):
        current_dir = Path(directory)
        relative_dir_parts = (
            ()
            if current_dir == root
            else current_dir.relative_to(root).parts
        )
        dirnames[:] = sorted(
            dirname
            for dirname in dirnames
            if (
                dirname not in _SKIPPED_SCAN_DIRS
                and not dirname.startswith(".")
                and (max_depth is None or len(relative_dir_parts) < max_depth - 1)
            )
        )
        for filename in sorted(filenames):
            path = current_dir / filename
            if (
                path.suffix.lower() in suffix_set
                and (
                    max_depth is None
                    or len(path.relative_to(root).parts) <= max_depth
                )
            ):
                paths.append(path)
    return tuple(paths)


def _load_lab_paths(
    dataset_root: Path,
    *,
    source_dataset: str,
    label_paths: tuple[Path, ...],
    audio_path_for_label: Callable[[Path], Path],
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    examples: list[CanonicalExample] = []
    for label_path in tqdm(label_paths, desc=source_dataset, unit="examples"):
        audio_path = audio_path_for_label(label_path)
        utterance_id = label_path.relative_to(dataset_root).with_suffix("").as_posix()
        examples.append(
            adapt_simple_lab_pair(
                audio_path=audio_path,
                label_path=label_path,
                source_dataset=source_dataset,
                utterance_id=utterance_id,
                include_audio_metadata=include_audio_metadata,
                repair_invalid_intervals=source_dataset == Dataset.ITAKO_SINGING.value,
            )
        )
    return tuple(examples)


def _load_lab_dir_dataset(
    dataset_root: Path,
    *,
    source_dataset: str,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    label_paths = tuple(sorted((dataset_root / "lab").glob("*.lab")))
    if not label_paths:
        return ()
    audio_dir_name, audio_suffix = _first_audio_dir_name_and_suffix(
        dataset_root / "wav" / f"{label_paths[0].stem}.wav"
    )
    return _load_lab_paths(
        dataset_root,
        source_dataset=source_dataset,
        label_paths=label_paths,
        include_audio_metadata=include_audio_metadata,
        audio_path_for_label=lambda label_path: (
            dataset_root / audio_dir_name / f"{label_path.stem}{audio_suffix}"
        ),
    )


def _load_mono_label_dataset(
    dataset_root: Path,
    *,
    source_dataset: str,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    label_paths = tuple(sorted((dataset_root / "mono_label").glob("*.lab")))
    if not label_paths:
        return ()
    audio_dir_name, audio_suffix = _first_audio_dir_name_and_suffix(
        dataset_root / "wav" / f"{label_paths[0].stem}.wav"
    )
    return _load_lab_paths(
        dataset_root,
        source_dataset=source_dataset,
        label_paths=label_paths,
        include_audio_metadata=include_audio_metadata,
        audio_path_for_label=lambda label_path: (
            dataset_root / audio_dir_name / f"{label_path.stem}{audio_suffix}"
        ),
    )


def _load_root_lab_dataset(
    dataset_root: Path,
    *,
    source_dataset: str,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    label_paths = tuple(sorted(dataset_root.glob("*.lab")))
    if not label_paths:
        return ()
    audio_suffix = _first_audio_suffix(label_paths[0].with_suffix(".wav"))
    return _load_lab_paths(
        dataset_root,
        source_dataset=source_dataset,
        label_paths=label_paths,
        include_audio_metadata=include_audio_metadata,
        audio_path_for_label=lambda label_path: _path_with_suffix(
            label_path,
            audio_suffix,
        ),
    )


def _load_nested_lab_dataset(
    dataset_root: Path,
    *,
    source_dataset: str,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    label_paths = tuple(sorted(dataset_root.glob("*/*.lab")))
    if not label_paths:
        return ()
    audio_suffix = _first_audio_suffix(label_paths[0].with_suffix(".wav"))
    return _load_lab_paths(
        dataset_root,
        source_dataset=source_dataset,
        label_paths=label_paths,
        include_audio_metadata=include_audio_metadata,
        audio_path_for_label=lambda label_path: _path_with_suffix(
            label_path,
            audio_suffix,
        ),
    )


def _load_enunu_dataset(
    dataset_root: Path,
    *,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    label_paths = _iter_files(dataset_root, (".lab",))
    if not label_paths:
        return ()

    def audio_candidate_for_label(label_path: Path) -> Path:
        relative_label = label_path.relative_to(dataset_root)
        if relative_label.parts and relative_label.parts[0] == "__labels":
            return dataset_root / Path(*relative_label.parts[1:]).with_suffix(".wav")
        return label_path.with_suffix(".wav")

    audio_suffix = _first_audio_suffix(audio_candidate_for_label(label_paths[0]))

    def audio_path_for_label(label_path: Path) -> Path:
        relative_label = label_path.relative_to(dataset_root)
        if relative_label.parts and relative_label.parts[0] == "__labels":
            return dataset_root / _path_with_suffix(
                Path(*relative_label.parts[1:]),
                audio_suffix,
            )
        return _path_with_suffix(label_path, audio_suffix)

    return _load_lab_paths(
        dataset_root,
        source_dataset=Dataset.ENUNU_KODOKU.value,
        label_paths=label_paths,
        include_audio_metadata=include_audio_metadata,
        audio_path_for_label=audio_path_for_label,
    )


def _load_alex_public_dataset(
    dataset_root: Path,
    *,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    rows = load_alex_public_rows(dataset_root)
    if not rows:
        return ()
    audio_dir_name, audio_suffix = _first_audio_dir_name_and_suffix(
        dataset_root / "wavs" / f"{rows[0]['name']}.wav"
    )
    return tuple(
        adapt_alex_public_row(
            row,
            dataset_root=dataset_root,
            audio_dir_name=audio_dir_name,
            audio_suffix=audio_suffix,
            include_audio_metadata=include_audio_metadata,
        )
        for row in tqdm(
            rows,
            desc=Dataset.ALEX_FLOAREA_EN_PUBLIC.value,
            unit="examples",
        )
    )


def _load_ngyy_dataset(
    dataset_root: Path,
    *,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    speaker_roots = tuple(
        path.parent
        for path in sorted(dataset_root.glob("*/transcriptions.csv"))
    )
    speaker_rows: list[tuple[Path, list[dict[str, str]]]] = []
    first_speaker_root: Path | None = None
    first_row: dict[str, str] | None = None
    for speaker_root in speaker_roots:
        rows = load_ngyy_rows(speaker_root)
        speaker_rows.append((speaker_root, rows))
        if rows and first_row is None:
            first_speaker_root = speaker_root
            first_row = rows[0]
    if first_speaker_root is None or first_row is None:
        return ()

    audio_dir_name, audio_suffix = _first_audio_dir_name_and_suffix(
        first_speaker_root / "wavs" / f"{first_row['name']}.wav"
    )

    examples: list[CanonicalExample] = []
    for speaker_root, rows in speaker_rows:
        examples.extend(
            adapt_ngyy_row(
                row,
                speaker_root=speaker_root,
                audio_dir_name=audio_dir_name,
                audio_suffix=audio_suffix,
                include_audio_metadata=include_audio_metadata,
            )
            for row in tqdm(rows, desc=f"{Dataset.NGYY.value}/{speaker_root.name}", unit="examples")
        )
    return tuple(examples)


def _load_opencpop_dataset(
    dataset_root: Path,
    *,
    transcriptions_path: str | Path | None = None,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    path = (
        Path(transcriptions_path)
        if transcriptions_path is not None
        else dataset_root / "transcriptions.txt"
    )
    lines = _read_nonempty_lines(path)
    if not lines:
        return ()
    first_name = lines[0].split("|", maxsplit=1)[0]
    audio_dir_name, audio_suffix = _first_audio_dir_name_and_suffix(
        dataset_root / "wavs" / f"{first_name}.wav"
    )
    return tuple(
        adapt_opencpop_line(
            line,
            dataset_root=dataset_root,
            transcriptions_path=path,
            audio_dir_name=audio_dir_name,
            audio_suffix=audio_suffix,
            include_audio_metadata=include_audio_metadata,
        )
        for line in tqdm(lines, desc=Dataset.OPEN_CPOP.value, unit="examples")
    )


def _load_m4singer_dataset(
    dataset_root: Path,
    *,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    items = load_m4singer_metadata(dataset_root)
    if not items:
        return ()
    first_song_name, first_clip_name = str(items[0]["item_name"]).rsplit("#", maxsplit=1)
    audio_suffix = _first_audio_suffix(
        dataset_root / first_song_name / f"{first_clip_name}.wav"
    )
    return tuple(
        adapt_m4singer_item(
            item,
            dataset_root=dataset_root,
            audio_suffix=audio_suffix,
            include_audio_metadata=include_audio_metadata,
        )
        for item in tqdm(items, desc=Dataset.M4SINGER.value, unit="examples")
    )


def _load_no7singing_dataset(
    dataset_root: Path,
    *,
    audio_variant: str = DEFAULT_NO7SINGING_AUDIO_VARIANT,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    normalized_variant = normalize_no7singing_audio_variant(audio_variant)
    label_root = dataset_root / "mono_label"
    label_paths = tuple(sorted(label_root.glob("*.lab")))
    if not label_paths:
        return ()
    audio_suffix = _first_audio_suffix(
        dataset_root / normalized_variant / f"{label_paths[0].stem}.wav"
    )
    return tuple(
        adapt_no7singing_example(
            dataset_root=dataset_root,
            utterance_id=label_path.stem,
            audio_variant=normalized_variant,
            audio_suffix=audio_suffix,
            include_audio_metadata=include_audio_metadata,
        )
        for label_path in tqdm(label_paths, desc=Dataset.NO7SINGING.value, unit="examples")
    )


def _load_ritsu_dataset(
    dataset_root: Path,
    *,
    variant: str = DEFAULT_RITSU_VARIANT,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    song_ids = iter_ritsu_song_ids(dataset_root=dataset_root, variant=variant)
    if not song_ids:
        return ()
    first_audio_path = resolve_ritsu_paths(
        dataset_root=dataset_root,
        song_id=song_ids[0],
        variant=variant,
        audio_suffix=".wav",
    )["audio_path"]
    audio_suffix = _first_audio_suffix(first_audio_path)
    return tuple(
        adapt_ritsu_example(
            dataset_root=dataset_root,
            song_id=song_id,
            variant=variant,
            audio_suffix=audio_suffix,
            include_audio_metadata=include_audio_metadata,
        )
        for song_id in tqdm(song_ids, desc=Dataset.RITSU.value, unit="examples")
    )


def _load_pjs_dataset(
    dataset_root: Path,
    *,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    corpus_root = dataset_root / dataset_root.name
    label_paths = (
        tuple(sorted((dataset_root / "pjs-manual-labels" / "lab").glob("*.lab")))
        + tuple(sorted(corpus_root.glob("pjs*/pjs*.lab")))
    )
    if not label_paths:
        return ()
    first_stem = label_paths[0].stem
    audio_suffix = _first_audio_suffix(
        corpus_root / first_stem / f"{first_stem}_song.wav"
    )
    examples: list[CanonicalExample] = []
    for label_path in tqdm(label_paths, desc=Dataset.PJS_CORPUS.value, unit="examples"):
        stem = label_path.stem
        audio_path = corpus_root / stem / f"{stem}_song{audio_suffix}"
        utterance_id = label_path.relative_to(dataset_root).with_suffix("").as_posix()
        examples.append(
            adapt_simple_lab_pair(
                audio_path=audio_path,
                label_path=label_path,
                source_dataset=Dataset.PJS_CORPUS.value,
                utterance_id=utterance_id,
                include_audio_metadata=include_audio_metadata,
            )
        )
    return tuple(examples)


def _load_gtsinger_dataset(
    dataset_root: Path,
    *,
    source_dataset: str,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    textgrid_paths = tuple(sorted(dataset_root.glob("*/*/*/*/*.TextGrid")))
    if not textgrid_paths:
        return ()
    audio_suffix = _first_audio_suffix(textgrid_paths[0].with_suffix(".wav"))
    examples: list[CanonicalExample] = []
    for textgrid_path in tqdm(textgrid_paths, desc=source_dataset, unit="examples"):
        json_path = textgrid_path.with_suffix(".json")
        audio_path = _path_with_suffix(textgrid_path, audio_suffix)
        examples.append(
            adapt_gtsinger_segment(
                textgrid_path=textgrid_path,
                json_path=json_path,
                audio_path=audio_path,
                source_dataset=source_dataset,
                include_audio_metadata=include_audio_metadata,
            )
        )
    return tuple(examples)


def _load_popcs_dataset(
    dataset_root: Path,
    *,
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    textgrid_paths = tuple(sorted(dataset_root.glob("*/*.TextGrid")))
    if not textgrid_paths:
        return ()
    audio_suffix = _first_audio_suffix(
        textgrid_paths[0].with_name(f"{textgrid_paths[0].stem}_wf0.wav")
    )
    examples: list[CanonicalExample] = []
    for textgrid_path in tqdm(textgrid_paths, desc=Dataset.POPCS.value, unit="examples"):
        lyrics_path = textgrid_path.with_suffix(".txt")
        phone_text_path = textgrid_path.with_name(f"{textgrid_path.stem}_ph.txt")
        audio_path = textgrid_path.with_name(f"{textgrid_path.stem}_wf0{audio_suffix}")
        examples.append(
            adapt_popcs_example(
                textgrid_path=textgrid_path,
                lyrics_path=lyrics_path,
                phone_text_path=phone_text_path,
                audio_path=audio_path,
                include_audio_metadata=include_audio_metadata,
            )
        )
    return tuple(examples)


def _load_sungandspoken_dataset(
    dataset_root: Path,
    *,
    label_glob: str = "*/*/*.txt",
    include_audio_metadata: bool,
) -> tuple[CanonicalExample, ...]:
    if label_glob == "**/*.txt":
        label_paths = _iter_files(dataset_root, (".txt",))
    elif label_glob == "*/*/*.txt":
        label_paths = tuple(sorted(dataset_root.glob(label_glob)))
    else:
        label_paths = tuple(sorted(dataset_root.glob(label_glob)))
    if not label_paths:
        return ()
    audio_suffix = _first_audio_suffix(label_paths[0].with_suffix(".wav"))
    from preprocessing.adapters.sungandspoken import adapt_sungandspoken_pair

    examples: list[CanonicalExample] = []
    for label_path in tqdm(label_paths, desc=Dataset.SUNG_AND_SPOKEN.value, unit="examples"):
        audio_path = _path_with_suffix(label_path, audio_suffix)
        examples.append(
            adapt_sungandspoken_pair(
                audio_path=audio_path,
                label_path=label_path,
                include_audio_metadata=include_audio_metadata,
            )
        )
    return tuple(examples)


def load_dataset(
    dataset: Dataset | str,
    dataset_root: str | Path,
    *,
    include_audio_metadata: bool = False,
    **kwargs: object,
) -> tuple[CanonicalExample, ...]:
    """Load a whole dataset root into canonical examples."""
    normalized_dataset = normalize_dataset(dataset)
    root = Path(dataset_root)

    if normalized_dataset is Dataset.PJS_CORPUS:
        return _load_pjs_dataset(
            root,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset in {
        Dataset.ALEX_FLOAREA_AI_SVS,
        Dataset.PROJECT_AIDOL_PUBLIC_ENGLISH,
        Dataset.NIT070_DB,
    }:
        return _load_lab_dir_dataset(
            root,
            source_dataset=normalized_dataset.value,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset is Dataset.ITAKO_SINGING:
        return _load_mono_label_dataset(
            root,
            source_dataset=normalized_dataset.value,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset in {Dataset.TIGER_EN, Dataset.TIGER_JP}:
        return _load_root_lab_dataset(
            root,
            source_dataset=normalized_dataset.value,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset in {
        Dataset.AMABOSHI_CIPHERDB,
        Dataset.KUROTAKE_KOUGA_AI_SONG,
        Dataset.OFUTON_P_UTAGOE_DB,
        Dataset.ONIKU_KURUMI_UTAGOE_DB,
    }:
        return _load_nested_lab_dataset(
            root,
            source_dataset=normalized_dataset.value,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset is Dataset.ENUNU_KODOKU:
        return _load_enunu_dataset(
            root,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset is Dataset.ALEX_FLOAREA_EN_PUBLIC:
        return _load_alex_public_dataset(
            root,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset is Dataset.NGYY:
        return _load_ngyy_dataset(
            root,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset is Dataset.OPEN_CPOP:
        return _load_opencpop_dataset(
            root,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset is Dataset.M4SINGER:
        return _load_m4singer_dataset(
            root,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset is Dataset.NO7SINGING:
        return _load_no7singing_dataset(
            root,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset is Dataset.RITSU:
        return _load_ritsu_dataset(
            root,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset is Dataset.POPCS:
        return _load_popcs_dataset(
            root,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset is Dataset.SUNG_AND_SPOKEN:
        return _load_sungandspoken_dataset(
            root,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )
    if normalized_dataset in {
        Dataset.GTSINGER_CHINESE,
        Dataset.GTSINGER_ENGLISH,
        Dataset.GTSINGER_JAPANESE,
    }:
        return _load_gtsinger_dataset(
            root,
            source_dataset=normalized_dataset.value,
            include_audio_metadata=include_audio_metadata,
            **kwargs,
        )

    raise NotImplementedError(f"no loader registered for {normalized_dataset.value!r}")


__all__ = [
    "Dataset",
    "load_dataset",
    "normalize_dataset",
]
