from __future__ import annotations

import os
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import preprocessing.datasets as dataset_module
from preprocessing import Dataset, load_dataset
from preprocessing.adapters.ritsu import resolve_ritsu_paths


DATA_ROOT = Path(
    os.environ.get(
        "SVS_DATASET_ROOT",
        # my dataset path - Christian
        "/mnt/t/svs/phoneme_f0_extraction/singing",
    )
)
RUN_MOUNTED_DATASET_TESTS = os.environ.get("SVS_RUN_MOUNTED_DATASET_TESTS") == "1"


MOUNTED_EXPECTED_COUNTS = {
    Dataset.ALEX_FLOAREA_AI_SVS: 224,
    Dataset.ALEX_FLOAREA_EN_PUBLIC: 899,
    Dataset.PROJECT_AIDOL_PUBLIC_ENGLISH: 57,
    Dataset.NGYY: 762,
    Dataset.OPEN_CPOP: 3756,
    Dataset.M4SINGER: 20896,
    Dataset.NO7SINGING: 51,
    Dataset.RITSU: 110,
    Dataset.POPCS: 1651,
    Dataset.SUNG_AND_SPOKEN: 48,
    Dataset.GTSINGER_CHINESE: 10188,
    Dataset.GTSINGER_ENGLISH: 5209,
    Dataset.GTSINGER_JAPANESE: 2832,
    Dataset.AMABOSHI_CIPHERDB: 85,
    Dataset.KUROTAKE_KOUGA_AI_SONG: 258,
    Dataset.OFUTON_P_UTAGOE_DB: 57,
    Dataset.ONIKU_KURUMI_UTAGOE_DB: 56,
    Dataset.PJS_CORPUS: 200,
    Dataset.ENUNU_KODOKU: 218,
    Dataset.ITAKO_SINGING: 50,
    Dataset.NIT070_DB: 31,
    Dataset.TIGER_EN: 29,
    Dataset.TIGER_JP: 7,
}


@dataclass(frozen=True)
class LoadedSample:
    audio_path: str
    utterance_id: str


def _dataset_root(dataset: Dataset) -> Path:
    for language in ("en", "jp", "zh"):
        candidate = DATA_ROOT / language / dataset.value
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"{dataset.value} is not present under {DATA_ROOT}")


def _sample(audio_path: str | Path, utterance_id: str) -> LoadedSample:
    return LoadedSample(audio_path=str(audio_path), utterance_id=utterance_id)


def _fake_simple_lab_pair(
    *,
    audio_path: str | Path,
    label_path: str | Path,
    source_dataset: str,
    utterance_id: str | None = None,
    **_: object,
) -> LoadedSample:
    return _sample(audio_path, utterance_id or Path(label_path).stem)


def _fake_alex_public_row(
    row: dict[str, object],
    *,
    dataset_root: str | Path,
    audio_dir_name: str = "wavs",
    audio_suffix: str = ".wav",
    include_audio_metadata: bool = False,
) -> LoadedSample:
    name = str(row["name"])
    return _sample(Path(dataset_root) / audio_dir_name / f"{name}{audio_suffix}", name)


def _fake_ngyy_row(
    row: dict[str, object],
    *,
    speaker_root: str | Path,
    audio_dir_name: str = "wavs",
    audio_suffix: str = ".wav",
    include_audio_metadata: bool = False,
) -> LoadedSample:
    root = Path(speaker_root)
    name = str(row["name"])
    return _sample(root / audio_dir_name / f"{name}{audio_suffix}", f"{root.name}/{name}")


def _fake_opencpop_line(
    line: str,
    *,
    dataset_root: str | Path,
    audio_dir_name: str = "wavs",
    audio_suffix: str = ".wav",
    **_: object,
) -> LoadedSample:
    name = line.split("|", maxsplit=1)[0]
    return _sample(Path(dataset_root) / audio_dir_name / f"{name}{audio_suffix}", name)


def _fake_m4singer_item(
    item: dict[str, object],
    *,
    dataset_root: str | Path,
    audio_suffix: str = ".wav",
    include_audio_metadata: bool = False,
) -> LoadedSample:
    item_name = str(item["item_name"])
    song_name, clip_name = item_name.rsplit("#", maxsplit=1)
    return _sample(Path(dataset_root) / song_name / f"{clip_name}{audio_suffix}", item_name)


def _fake_no7singing_example(
    *,
    dataset_root: str | Path,
    utterance_id: str,
    audio_variant: str,
    audio_suffix: str = ".wav",
    **_: object,
) -> LoadedSample:
    return _sample(
        Path(dataset_root) / audio_variant / f"{utterance_id}{audio_suffix}",
        f"{utterance_id}/{audio_variant}",
    )


def _fake_ritsu_example(
    *,
    dataset_root: str | Path,
    song_id: str,
    variant: str,
    audio_suffix: str = ".wav",
    **_: object,
) -> LoadedSample:
    paths = resolve_ritsu_paths(
        dataset_root=dataset_root,
        song_id=song_id,
        variant=variant,
        audio_suffix=audio_suffix,
    )
    return _sample(paths["audio_path"], f"{variant}/{song_id}")


def _fake_gtsinger_segment(
    *,
    textgrid_path: str | Path,
    json_path: str | Path,
    audio_path: str | Path,
    source_dataset: str,
    **_: object,
) -> LoadedSample:
    return _sample(audio_path, str(Path(textgrid_path).with_suffix("")))


def _fake_popcs_example(
    *,
    textgrid_path: str | Path,
    audio_path: str | Path,
    **_: object,
) -> LoadedSample:
    textgrid = Path(textgrid_path)
    return _sample(audio_path, f"{textgrid.parent.name}/{textgrid.stem}")


def _fake_sungandspoken_pair(
    *,
    audio_path: str | Path,
    label_path: str | Path,
) -> LoadedSample:
    return _sample(audio_path, str(Path(label_path).with_suffix("")))


def _patched_loaders() -> list[patch]:
    return [
        patch.object(dataset_module, "adapt_simple_lab_pair", _fake_simple_lab_pair),
        patch.object(dataset_module, "adapt_alex_public_row", _fake_alex_public_row),
        patch.object(dataset_module, "adapt_ngyy_row", _fake_ngyy_row),
        patch.object(dataset_module, "adapt_opencpop_line", _fake_opencpop_line),
        patch.object(dataset_module, "adapt_m4singer_item", _fake_m4singer_item),
        patch.object(dataset_module, "adapt_no7singing_example", _fake_no7singing_example),
        patch.object(dataset_module, "adapt_ritsu_example", _fake_ritsu_example),
        patch.object(dataset_module, "adapt_gtsinger_segment", _fake_gtsinger_segment),
        patch.object(dataset_module, "adapt_popcs_example", _fake_popcs_example),
        patch.object(dataset_module, "resolve_audio_path", lambda path: Path(path)),
        patch(
            "preprocessing.adapters.sungandspoken.adapt_sungandspoken_pair",
            _fake_sungandspoken_pair,
        ),
    ]


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")


class DatasetLoaderTest(unittest.TestCase):
    def test_every_public_loader_has_a_mounted_count(self) -> None:
        self.assertEqual(set(MOUNTED_EXPECTED_COUNTS), set(Dataset))

    def test_audio_suffix_is_inferred_once_per_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stem in ("a", "b"):
                _touch(root / "lab" / f"{stem}.lab")

            resolved_paths: list[Path] = []

            def resolve_first_audio(path: str | Path) -> Path:
                path = Path(path)
                resolved_paths.append(path)
                return root / "flac" / f"{path.stem}.flac"

            patches = [
                patch.object(dataset_module, "adapt_simple_lab_pair", _fake_simple_lab_pair),
                patch.object(dataset_module, "resolve_audio_path", resolve_first_audio),
            ]
            for active_patch in patches:
                active_patch.start()
            self.addCleanup(lambda: [active_patch.stop() for active_patch in reversed(patches)])

            examples = load_dataset(Dataset.ALEX_FLOAREA_AI_SVS, root)

            self.assertEqual(resolved_paths, [root / "wav" / "a.wav"])
            self.assertEqual(
                [example.audio_path for example in examples],
                [str(root / "flac" / "a.flac"), str(root / "flac" / "b.flac")],
            )

    def test_repeated_textgrid_basenames_remain_distinct(self) -> None:
        patches = _patched_loaders()
        for active_patch in patches:
            active_patch.start()
        self.addCleanup(lambda: [active_patch.stop() for active_patch in reversed(patches)])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for song in ("song_a", "song_b"):
                _touch(root / song / "0000.TextGrid")
                _touch(root / song / "0000.txt")
                _touch(root / song / "0000_ph.txt")
                _touch(root / song / "0000_wf0.wav")

            examples = load_dataset(Dataset.POPCS, root)
            self.assertEqual(len(examples), 2)
            self.assertEqual({example.utterance_id for example in examples}, {"song_a/0000", "song_b/0000"})
            self.assertEqual(len({example.audio_path for example in examples}), 2)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for singer in ("singer_a", "singer_b"):
                leaf = root / singer / "style" / "song" / "group"
                _touch(leaf / "0000.TextGrid")
                _touch(leaf / "0000.json")
                _touch(leaf / "0000.wav")

            examples = load_dataset(Dataset.GTSINGER_CHINESE, root)
            self.assertEqual(len(examples), 2)
            self.assertEqual(len({example.utterance_id for example in examples}), 2)
            self.assertEqual(len({example.audio_path for example in examples}), 2)

    @unittest.skipUnless(DATA_ROOT.exists(), f"dataset root not found: {DATA_ROOT}")
    @unittest.skipUnless(
        RUN_MOUNTED_DATASET_TESTS,
        "set SVS_RUN_MOUNTED_DATASET_TESTS=1 to scan the mounted datasets",
    )
    def test_loader_counts_match_mounted_dataset_counts(self) -> None:
        for dataset, expected_count in MOUNTED_EXPECTED_COUNTS.items():
            with self.subTest(dataset=dataset.value):
                examples = load_dataset(dataset, _dataset_root(dataset))
                utterance_ids = [example.utterance_id for example in examples]

                self.assertEqual(len(examples), expected_count)
                self.assertEqual(len(set(utterance_ids)), len(utterance_ids))
                self.assertTrue(all(example.audio_path for example in examples))
