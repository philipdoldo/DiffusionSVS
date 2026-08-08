# svs-datasets

Utilities for loading most singing voice dataset labels from [this list](https://github.com/openvpi/MakeDiffSinger/wiki/Public-datasets)
into a coherent format. In particular, normalizes each into a shared phoneset for each language
based loosely around the [SynthV phonesets](https://manual.synthv.info/phonemes/). These phonesets
can be found at `preprocessing/phonesets/phonesets.py`.

**DOES NOT load audio!!!** Use or write a binarizer for that.

## Public API

```python
from svs_datasets import CanonicalExample, Dataset, Interval, NoteInterval, load_dataset
```

`load_dataset` is the main entrypoint:

```python
examples = load_dataset(Dataset.M4SINGER, "/data/singing/zh/m4singer")
first = examples[0]

print(first.utterance_id)
print(first.audio_path)
print(first.phone_intervals[:5])
```

Signature:

```python
load_dataset(
    dataset: Dataset | str,
    dataset_root: str | Path,
    *,
    include_audio_metadata: bool = False,
    **dataset_options,
) -> tuple[CanonicalExample, ...]
```

`dataset` may be a `Dataset` enum member, enum value string, enum name, or a
small alias such as `"m4"` or `"open_cpop"`. `dataset_root` is the directory for
that specific dataset, not the language parent directory. Progress bars are shown
with `tqdm`.

Usually the root is just the root after the dataset zip/tarball has been
extracted, with the exception of datasets that contain multiple languages such as
TIGER and GTSinger, in which case the root must point to a particular language split.
Also, for Ritsu, it expects one folder up (e.g. `ritsu/「波音リツ」歌声データベースVer2.0.2`,
this is to be compatible with multiple Ritsu voicebanks), and we only use the sung split of
Sung and Spoken (duh).

Set `include_audio_metadata=True` when metadata like sample rate/sample count are
necessary and leave it disabled otherwise.

Dataset-specific options currently include:

- `audio_variant=...` for `Dataset.NO7SINGING`
- `variant=...` for `Dataset.RITSU`
- `transcriptions_path=...` for `Dataset.OPEN_CPOP`

## CanonicalExample

Every loader returns `CanonicalExample` instances. The required fields are:

- `audio_path: str` - path to the waveform for this sample
- `utterance_id: str` - stable dataset-local ID
- `source_dataset: str` - original dataset ID
- `raw_format: str` - adapter/parser family that produced the example

Optional fields:

- `speaker_id: str | None`
- `audio_sampling_rate: int | None`
- `audio_num_samples: int | None`
- `lyrics_text: str | None`
- `phone_sequence: tuple[str, ...] | None`
- `phone_intervals: tuple[Interval, ...]`
- `word_intervals: tuple[Interval, ...] | None`
- `note_intervals: tuple[NoteInterval, ...] | None`
- `line_start_sec: float | None`
- `line_end_sec: float | None`
- `source_paths: dict[str, str]`
- `metadata: dict[str, object]`

`phone_intervals` is really the only relevant field for singing examples; this lists
phonemes as well as the time interval for which each occurs. `audio_path` will also
be useful for binarizers.

`Interval` is a labeled time span:

```python
Interval(label: str, start_sec: float, end_sec: float)
```

`NoteInterval` is a note-aligned span, when MIDI pitch information is available:

```python
NoteInterval(
    start_sec: float,
    end_sec: float,
    pitch_midi: int | None = None,
    lyric: str | None = None,
    is_slur: bool | None = None,
)
```

All three dataclasses provide `to_dict()` / `from_dict()` helpers.

## Supported Datasets

`Dataset` contains the dataset IDs understood by `load_dataset`:

- `ALEX_FLOAREA_AI_SVS` -> `"AlexFloarea-AI-SVS"`
- `ALEX_FLOAREA_EN_PUBLIC` -> `"Alex_Floarea_EN_Public_Corpus"`
- `PROJECT_AIDOL_PUBLIC_ENGLISH` -> `"Project-AIdol-Public-English-Dataset"`
- `NGYY` -> `"NGYY_ENG_Dataset"`
- `OPEN_CPOP` -> `"opencpop"`
- `M4SINGER` -> `"m4singer"`
- `NO7SINGING` -> `"no7singing"`
- `RITSU` -> `"ritsu"`
- `POPCS` -> `"popcs"`
- `SUNG_AND_SPOKEN` -> `"sungandspoken"`
- `GTSINGER_CHINESE` -> `"GTSinger_Chinese"`
- `GTSINGER_ENGLISH` -> `"GTSinger_English"`
- `GTSINGER_JAPANESE` -> `"GTSinger_Japanese"`
- `AMABOSHI_CIPHERDB` -> `"Amaboshi_CipherDB"`
- `KUROTAKE_KOUGA_AI_SONG` -> `"Kurotake_Kouga_AI_Song"`
- `OFUTON_P_UTAGOE_DB` -> `"OFUTON_P_UTAGOE_DB"`
- `ONIKU_KURUMI_UTAGOE_DB` -> `"ONIKU_KURUMI_UTAGOE_DB"`
- `PJS_CORPUS` -> `"PJS_corpus_ver1.1"`
- `ENUNU_KODOKU` -> `"enunu_kodoku_database_20220807-2"`
- `ITAKO_SINGING` -> `"itako_singing"`
- `NIT070_DB` -> `"nit070_db"`
- `TIGER_EN` -> `"tiger_en"`
- `TIGER_JP` -> `"tiger_jp"`

### Types of silence

"AP" and "SP" are used to describe breath sounds ("aspirated pause")
and silence ("silent pause") respectively. However, the way these are
treated differs greatly across datasets, and are sometimes not even
consistent within a dataset.

For instance, AP (or its equivalents such as pau or br) is sometimes
used to label a stretch of a few seconds that only has the breath at the end,
which should really be a long SP interval followed by a short AP. Or a song
might end with an AP annotation, which doesn't make sense - if there is trailing
silence it should be SP.

This needs to be addressed somehow somewhere, probably in preprocessing.
One possibility is to normalize silence handling across all datasets in a
preprocessing step: group all intervals marked as any form of silence
and use a heuristic such as energy to determine which spans are truly silent
and which are breath sounds. This has the added benefit of being helpful downstream
for forced alignment, where AP/SP annotations are sparse, because the same
heuristic can be used to determine these from the audio.
