"""
First, we create splits and a phoneme vocabulary using the `build_phoneme_vocab` and `build_splits` functions.
This script parses all of the TextGrid files found under `raw_data_dirs` (recursively, at any depth) to build
a phoneme vocabulary which includes special tokens. This vocabulary is saved in `<save_dir>/vocab.json`.
Additionally, this builds train, val, and test splits where the val and test splits are defined in the config.
The splits are stored in `<save_dir>/splits.json`.

    Directory traversal
    --------------------
    Each entry in `raw_data_dirs` is walked recursively via `os.walk`, so datasets don't need to keep all of
    their TextGrid files exactly one level deep like PopCS does (`popcs/popcs-Bad/0000.TextGrid`). Datasets
    whose audio lives several subdirectories down are handled the same way.

    Any directory whose *name* appears in `config["data"]["exclude_dirs"]` is pruned from the walk entirely
    (we never descend into it), which is useful for e.g. skipping a `Paired_Speech_Group` subdirectory that
    contains spoken, not sung, audio.

    Item identity vs. item metadata
    ---------------------------------
    Each utterance's h5 group key is an opaque, zero-padded sequential id ('000000', '000001', ...) assigned
    at binarization time -- NOT a string built out of directory/file names. The original relative path is
    still recorded, but as an HDF5 attribute on the group (`grp.attrs["source"]`) and in
    `splits/{split}_items.json`, not as part of the key itself.

    We deliberately avoid building identifiers by joining path components with a delimiter (e.g.
    "popcs|popcs-Bad|0000"). That approach ties correctness to an assumption about which characters never
    appear in a dataset's directory/file names, which is exactly the kind of thing that's fine for months
    and then silently isn't. Sequential ids can't collide by construction, so there's nothing to validate.

    We walk each raw_data_dir exactly once (`collect_items`) and carry the resolved wav/TextGrid paths
    alongside each record from that point on, so nothing ever needs to be reconstructed from a key.

    val_songs / test_songs config format
    --------------------------------------
    With multiple raw_data_dirs, a bare song directory name (e.g. 'popcs-说散就散') is ambiguous -- two
    different datasets could have a same-named song subdirectory. So val_songs/test_songs entries must be
    prefixed with the raw_data_dir name, e.g. 'popcs/popcs-说散就散', matching each item's `song_name` field
    (see `collect_items`).

    When building mel2ph we sometimes encounter TextGrid intervals with empty-string text fields, e.g. interval 53
    in the example below:

            intervals [52]:
				xmin = 8.14
				xmax = 8.61
				text = "ou"
			intervals [53]:
				xmin = 8.61
				xmax = 9.59
				text = ""
			intervals [54]:
				xmin = 9.59
				xmax = 9.86
				text = "uo"

    We handle these cases by replacing the empty string with the silence token SP if the interval isn't the first or last interval. If it is,
    then we replace the empty string with <BOS> if it is the first interval and <EOS> if it is the last interval.
"""
import numpy as np
import librosa
import parselmouth
import argparse
import textgrid
import toml
import json
import h5py
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

PAD_TOKEN = "<PAD>"
SILENCE_TOKEN = "<SP>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"


# ----------------------------------------------------------------------------
# Directory traversal / item collection
# ----------------------------------------------------------------------------

def iter_textgrid_files(raw_data_dir: Path, exclude_dir_names: set[str]):
    """
    Recursively walk `raw_data_dir`, yielding paths to every `.TextGrid` file found, at any depth.

    Any directory whose name is in `exclude_dir_names` is pruned from `dirnames` in place, so `os.walk`
    never descends into it at all (not just filtered out afterward).

    Both directories and files are sorted at each level so results are deterministic across runs/machines.
    """
    for dirpath, dirnames, filenames in os.walk(raw_data_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in exclude_dir_names)
        for fname in sorted(filenames):
            if fname.endswith(".TextGrid"):
                yield Path(dirpath) / fname


def find_wav_for_textgrid(tg_path: Path) -> Path:
    """
    Find the audio file that corresponds to a TextGrid file. We look in the TextGrid's own directory for
    any file matching `{stem}*.wav`, which covers both PopCS's `{stem}_wf0.wav` convention and a plain
    `{stem}.wav`. Raises if we can't find exactly one unambiguous match.
    """
    stem = tg_path.stem
    candidates = sorted(tg_path.parent.glob(f"{stem}*.wav"))

    if len(candidates) == 0:
        raise FileNotFoundError(f"no .wav file found for {tg_path} (looked for '{stem}*.wav' in {tg_path.parent})")

    if len(candidates) > 1:
        print(f"Surely this never happens? {tg_path=}, {candidates=}")
        exact = tg_path.parent / f"{stem}.wav"
        if exact in candidates:
            return exact
        raise BinarizationError(f"ambiguous wav match for {tg_path}: {candidates}")

    return candidates[0]


def collect_items(raw_data_dirs: list[Path], exclude_dirs: list[str] = ()) -> list[dict]:
    """
    Walk every directory in `raw_data_dirs` recursively (pruning any directory named in `exclude_dirs`) and
    return a list of dicts, one per TextGrid file found:

        {
            "source":    str,   # human-readable relative path, e.g. 'popcs/popcs-Bad/0000' -- metadata
                                 # only, never used as a dict/h5 key, so it's fine for this to contain '/'
                                 # or even (rare, harmless) duplicates across raw_data_dirs
            "tg_path":   Path,
            "wav_path":  Path,
            "song_name": str,   # raw_data_dir name + path down to the TextGrid's parent dir, e.g.
                                 # 'popcs/popcs-Bad' -- used for val/test song matching. Prefixed with the
                                 # raw_data_dir name (not just the immediate parent dir name) so that two
                                 # different datasets with a same-named song subdirectory don't collide;
                                 # config val_songs/test_songs entries should use this same
                                 # '<raw_data_dir name>/<song dir>' form, e.g. 'popcs/popcs-说散就散'.
        }

    We do this walk exactly once and reuse the resulting list for vocab building, split building, and
    binarization, rather than re-walking the tree three separate times.

    TextGrids without a resolvable wav file are skipped with a printed warning rather than raising, since
    this is a traversal/bookkeeping step, not a per-utterance data-quality step (that's what
    `process_utterance` / `BinarizationError` are for).
    """
    exclude_dir_names = set(exclude_dirs)
    items = []
    for raw_data_dir in raw_data_dirs:
        raw_data_dir = Path(raw_data_dir)
        for tg_path in iter_textgrid_files(raw_data_dir, exclude_dir_names):
            try:
                wav_path = find_wav_for_textgrid(tg_path)
            except (FileNotFoundError, BinarizationError) as e:
                print(f"skipping {tg_path}: {e}")
                continue

            rel = tg_path.relative_to(raw_data_dir).with_suffix("").as_posix()

            rel_parent = tg_path.parent.relative_to(raw_data_dir).as_posix()  # '.' if TextGrid sits directly in raw_data_dir
            song_name = raw_data_dir.name if rel_parent == "." else f"{raw_data_dir.name}/{rel_parent}"

            items.append({
                "source":    f"{raw_data_dir.name}/{rel}",
                "tg_path":   tg_path,
                "wav_path":  wav_path,
                "song_name": song_name,
            })

    print(f"collected {len(items)} items from {len(raw_data_dirs)} raw_data_dir(s)")
    return items


# ----------------------------------------------------------------------------
# Vocab / splits
# ----------------------------------------------------------------------------

def build_phoneme_vocab(items: list[dict], save_dir: str) -> dict[str, int]:
    """
    `items` is the list of records returned by `collect_items`. This iterates through every TextGrid file
    referenced there and collects the phonemes from Tier 2 of the TextGrid.

    Output is a dictionary with string token keys (corresponding to phonemes or special tokens) and integer
    values. A .json file of the collected vocabulary is saved in `save_dir`
    """
    special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SILENCE_TOKEN]
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    vocab = set()  # store phonemes
    for item in items:
        text_grid = textgrid.TextGrid.fromFile(str(item["tg_path"]))
        tier = text_grid.tiers[1] # we use index 1 because we want the second tier because that is what other people used
        for interval in tier:
            phoneme = interval.mark.strip()
            if len(phoneme) > 0:
                vocab.add(phoneme)

    vocab = special_tokens + sorted(vocab)
    vocab = {token: i for i, token in enumerate(vocab)}
    save_path = save_dir / "vocab.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    print(f"Phoneme vocabulary: {len(vocab)} tokens (including special tokens: {special_tokens})")
    print(f"Tokens: {vocab}")
    print(f"Saved to {save_path}")

    return vocab


def build_splits(items: list[dict], save_dir: str, val_songs: list[str], test_songs: list[str]) -> dict[str, list[dict]]:
    """
    Given lists of song names for the val and test sets. With multiple raw_data_dirs, entries must be
    prefixed with the raw_data_dir name to disambiguate, e.g.
        test_songs = ["popcs/popcs-说散就散", "popcs/popcs-隐形的翅膀"]
    (DiffSinger itself uses bare names like "popcs-说散就散" since it only ever dealt with a single
    raw_data_dir; see https://github.com/MoonInTheRiver/DiffSinger/blob/ce7789f1427ddcdec647b3ab2bf2d1b12134e51e/data_gen/tts/base_binarizer.py#L65 -- also note DiffSinger made val == test there, which looks unintentional)

    `items` is the list of records returned by `collect_items`; each item's `song_name` (already prefixed
    with its raw_data_dir name -- see `collect_items`) is matched against `val_songs`/`test_songs`. This
    works regardless of how deeply that directory is nested under its raw_data_dir.

    Returns a dict of split -> list of item dicts (the same records passed in, just partitioned) so that
    `binarize_split` can use them directly, with no name-based lookup step in between.

    A human-readable record of the assignment (each item's `source` string) is also saved to
    `<save_dir>/splits.json`, for inspection only -- nothing in this script reads it back.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    test_songs = set(test_songs)
    val_songs = set(val_songs)

    splits = {"test": [], "val": [], "train": []}

    for item in items:
        if item["song_name"] in test_songs:
            splits["test"].append(item)
        elif item["song_name"] in val_songs:
            splits["val"].append(item)
        else:
            splits["train"].append(item)

    save_path = save_dir / "splits.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump({split: [item["source"] for item in split_items] for split, split_items in splits.items()}, f, ensure_ascii=False, indent=2)

    for split, split_items in splits.items():
        print(f"{split:>5}: {len(split_items)} utterances saved in {save_path}")

    return splits


class BinarizationError(Exception):
    pass


def _is_nondecreasing(arr):
    return np.all(np.diff(arr) >= 0)


def process_utterance(
    wav_path:       str,
    tg_path:        str,
    phoneme_to_idx: dict[str, int],
    config:         dict,
) -> dict:
    """
    Extract all features for a single utterance. Raises BinarizationError if
    the utterance should be skipped.

    Returns a dict with keys: mel, f0, uv, mel2ph, txt_tokens
    """
    audio_config = config["audio"]
    f0_config    = config["f0"]

    # ------------------------------------------------------------------
    # 1. load and resample
    # ------------------------------------------------------------------
    audio, sr = librosa.load(wav_path, sr=None, mono=True)
    if sr != audio_config["sample_rate"]:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=audio_config["sample_rate"], res_type="kaiser_best")
    audio = audio / (np.abs(audio).max() + 1e-8)

    # ------------------------------------------------------------------
    # 2. mel spectrogram
    # ------------------------------------------------------------------
    x_stft = librosa.stft(
        audio,
        n_fft=audio_config["window_size"],
        hop_length=audio_config["hop_size"],
        win_length=audio_config["window_size"],
        window="hann",
        pad_mode="constant",
    )
    spc = np.abs(x_stft)  # amplitude spectrogram (n_bins, T)

    mel_basis = librosa.filters.mel(
        sr=audio_config["sample_rate"],
        n_fft=audio_config["window_size"],
        n_mels=audio_config["n_mels"],
        fmin=audio_config["min_freq"],
        fmax=audio_config["max_freq"],
    )
    mel = mel_basis @ spc  # (n_mels, T)
    log_mel = np.log10(np.maximum(audio_config["mel_log_clip"], mel)).astype(np.float32)  # (n_mels, T)

    T = log_mel.shape[1]

    # ------------------------------------------------------------------
    # 3. f0
    # ------------------------------------------------------------------
    frame_times = np.arange(T) * audio_config["hop_size"] / audio_config["sample_rate"]

    sound = parselmouth.Sound(audio, sampling_frequency=audio_config["sample_rate"])
    pitch = sound.to_pitch_ac(
        time_step=audio_config["hop_size"] / audio_config["sample_rate"],
        pitch_floor=f0_config["f0_min"],
        pitch_ceiling=f0_config["f0_max"],
    )
    f0_raw = np.array([pitch.get_value_at_time(t) or 0.0 for t in frame_times], dtype=np.float32)
    uv = np.logical_or(f0_raw == 0.0, np.isnan(f0_raw))  # True = unvoiced
    if np.all(uv):
        raise ValueError(f"All frames of f0 are unvoiced, {wav_path=}, {tg_path=}")

    f0_raw = np.where(np.logical_not(uv), np.clip(f0_raw, f0_config["f0_min"], f0_config["f0_max"]), f0_raw) # this might not be needed since it might already be clipped, but doing it just in case
    f0_mel = np.where(np.logical_not(uv), 1127 * np.log(1 + f0_raw / 700), 0.0)

    # linearly interpolate log f0 across unvoiced gaps
    voiced_indices = np.where(np.logical_not(uv))[0]
    f0 = np.interp(x=np.arange(T), xp=voiced_indices, fp=f0_mel[voiced_indices]).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. mel2ph and txt_tokens from TextGrid
    # ------------------------------------------------------------------
    tg = textgrid.TextGrid.fromFile(tg_path)
    tier = tg.tiers[1] # we use the second tier

    # If an interval has `text = ""` (i.e. no phoneme), then we map it to the silence phoneme SP
    # unless it is the first or last interval in which case we map it to <BOS> or <EOS>, respectively.
    tier_list = list(tier) # wrap in list so length can be computed
    intervals = [
        (
            interval.minTime,
            interval.maxTime,
            BOS_TOKEN if i == 0 and not interval.mark.strip()
            else EOS_TOKEN if i == len(tier_list) - 1 and not interval.mark.strip()
            else interval.mark.strip() or SILENCE_TOKEN
        )
        for i, interval in enumerate(tier_list)
    ]

    txt_tokens = np.array([phoneme_to_idx[ph] for _, _, ph in intervals], dtype=np.int32)

    starts = [int(np.round(start * audio_config["sample_rate"] / audio_config["hop_size"])) for start, _, _ in intervals]
    ends = starts[1:] + [T]

    assert starts[0] == 0, f"{starts=}"

    mel2ph = np.zeros(T, dtype=np.int32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        mel2ph[s:e] = i

    if (not _is_nondecreasing(starts)) or (not _is_nondecreasing(ends)) or (not _is_nondecreasing(mel2ph)) or (mel2ph.max() != len(txt_tokens) - 1):
        raise BinarizationError # skip if the data isn't formatted properly, e.g. `popcs/popcs-爱你十分泪七分/0015.TextGrid` has final xmax of 11.819999999999993 but `/popcs/popcs-爱你十分泪七分/0015_wf0.wav` is only 11.42328798185941 seconds

    max_mel_frames = config["data"].get("max_mel_frames") # truncate to max_mel_frames
    if max_mel_frames is not None and T > max_mel_frames:
        log_mel    = log_mel[:, :max_mel_frames]
        f0         = f0[:max_mel_frames]
        uv         = uv[:max_mel_frames]
        mel2ph     = mel2ph[:max_mel_frames]
        txt_tokens = txt_tokens[:mel2ph[-1] + 1]

    return {
        "mel":        log_mel,    # float32 (n_mels, T)
        "f0":         f0,         # float32 (T,)
        "uv":         uv,         # bool    (T,)
        "mel2ph":     mel2ph,     # int32   (T,)
        "txt_tokens": txt_tokens, # int32   (P,)
    }


def binarize_split(
    split:          str,
    split_items:    list[dict],
    save_dir:       Path,
    phoneme_to_idx: dict[str, int],
    config:         dict,
):
    """
    `split_items` is a list of the per-item dicts produced by `collect_items` (already filtered down to
    just this split). Each utterance is written under an opaque, zero-padded sequential id (e.g. '000000')
    as its h5 group key; the id can't collide with anything else by construction, since it's just this
    loop's position. The original relative path is kept as an attribute on the group for traceability,
    not as part of the key.
    """
    successful = [] # list of {"id": ..., "source": ...}, for splits/{split}_items.json
    skipped    = [] # list of source strings

    id_width = max(1, len(str(len(split_items))))  # cosmetic padding width only; ids never overflow or collide regardless

    with h5py.File(save_dir / f"{split}.h5", "w") as f:
        for i, item in enumerate(tqdm(split_items, desc=split)):
            item_id = f"{i:0{id_width}d}"
            source = item["source"]
            try:
                result = process_utterance(
                    wav_path=str(item["wav_path"]),
                    tg_path=str(item["tg_path"]),
                    phoneme_to_idx=phoneme_to_idx,
                    config=config,
                )
            except BinarizationError as e:
                print(f"skipping {source}: {e}")
                skipped.append(source)
                continue

            grp = f.create_group(item_id)
            grp.attrs["source"] = source  # e.g. h5py.File(...)['000000'].attrs['source'] -> 'popcs/popcs-Bad/0000'
            grp.create_dataset("mel",        data=result["mel"])
            grp.create_dataset("f0",         data=result["f0"])
            grp.create_dataset("uv",         data=result["uv"])
            grp.create_dataset("mel2ph",     data=result["mel2ph"])
            grp.create_dataset("txt_tokens", data=result["txt_tokens"])
            successful.append({"id": item_id, "source": source})

    splits_dir = save_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    # lengths = []
    # with h5py.File(save_dir / f"{split}.h5", "r") as f:
    #     for item_name in successful:
    #         lengths.append(f[item_name]["mel"].shape[-1]) # save number of mel frames, maybe useful for length batching but I don't think I need this for now
    # np.save(splits_dir / f"{split}_lengths.npy", np.array(lengths, dtype=np.int32))

    with open(splits_dir / f"{split}_items.json", "w", encoding="utf-8") as f:
        json.dump(successful, f, ensure_ascii=False, indent=2)

    with open(splits_dir / f"{split}_skipped.json", "w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)

    print(f"{split}: {len(successful)} saved, {len(skipped)} skipped")
    if skipped:
        print(f"  skipped: {skipped}")

def compute_and_save_stats(split: str, save_dir: Path):
    with h5py.File(save_dir / f"{split}.h5", "r") as f:
        item_ids = list(f.keys())
        if not item_ids:
            raise ValueError(f"{split}.h5 has no items -- can't compute stats")

        n_mels = f[item_ids[0]]["mel"].shape[0] # typically 80, we'll want mean, std, min, max for all 80 values across all mel frames
        mel_sum, mel_sumsq = np.zeros(n_mels), np.zeros(n_mels)
        mel_min, mel_max   = np.full(n_mels, np.inf), np.full(n_mels, -np.inf)
        mel_count = 0
        f0_sum = f0_sumsq = 0.0
        f0_min, f0_max = np.inf, -np.inf
        f0_count = 0

        for item_id in tqdm(item_ids, desc=f"{split} stats"):
            mel = f[item_id]["mel"][:]  # (n_mels, T)
            f0  = f[item_id]["f0"][:]   # (T,)

            mel_sum   += mel.sum(axis=1)
            mel_sumsq += np.square(mel, dtype=np.float64).sum(axis=1)
            mel_min = np.minimum(mel_min, mel.min(axis=1))
            mel_max = np.maximum(mel_max, mel.max(axis=1))
            mel_count += mel.shape[1]

            f0_sum   += f0.sum()
            f0_sumsq += np.square(f0, dtype=np.float64).sum()
            f0_min = min(f0_min, f0.min())
            f0_max = max(f0_max, f0.max())
            f0_count += f0.shape[0]

    mel_mean = mel_sum / mel_count
    mel_std  = np.sqrt(mel_sumsq / mel_count - mel_mean ** 2)
    f0_mean  = f0_sum / f0_count
    f0_std   = np.sqrt(f0_sumsq / f0_count - f0_mean ** 2)

    np.savez(
        save_dir / f"{split}_stats.npz",
        mel_mean = mel_mean.astype(np.float32),
        mel_std  = mel_std.astype(np.float32),
        mel_min  = mel_min.astype(np.float32),
        mel_max  = mel_max.astype(np.float32),
        f0_mean  = np.float32(f0_mean),
        f0_std   = np.float32(f0_std),
        f0_min   = np.float32(f0_min),
        f0_max   = np.float32(f0_max),
    )
    print(f"{split}: stats saved to {save_dir / f'{split}_stats.npz'}")


def create_data_dir(parent_dir, config_name):
    timestamp = datetime.now().strftime("%m-%d-%Y-%Hh%Mm%Ss")
    data_dir = os.path.join(parent_dir, config_name, f"{timestamp}")
    os.makedirs(data_dir, exist_ok=True)
    return Path(data_dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help=".toml file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = toml.load(f)

    save_dir = create_data_dir(parent_dir=config["data"]["save_dir"], config_name=Path(args.config).stem)
    if type(config["data"]["raw_data_dirs"]) != list:
        raise ValueError(f"Expected list of strings, got {type(config['data']['raw_data_dirs'])=}, {config['data']['raw_data_dirs']=}")
    raw_data_dirs = [Path(s) for s in config["data"]["raw_data_dirs"]]

    exclude_dirs = config["data"].get("exclude_dirs", []) # e.g. exclude_dirs = ["Paired_Speech_Group"] in the config's [data] table to skip any subdirectory with that name, at any depth, in any raw_data_dir.

    with open(os.path.join(save_dir, "config.toml"), "w") as f:
        toml.dump(config, f) # save copy of config in binarized data directory

    items = collect_items(raw_data_dirs=raw_data_dirs, exclude_dirs=exclude_dirs)

    vocab = build_phoneme_vocab(items=items, save_dir=save_dir)

    splits = build_splits(
        items=items,
        save_dir=save_dir,
        val_songs=config["data"]["val_songs"],
        test_songs=config["data"]["test_songs"],
    )

    for split, split_items in splits.items():
        binarize_split(
            split=split,
            split_items=split_items,
            save_dir=save_dir,
            phoneme_to_idx=vocab,
            config=config,
        )

    for split in splits.keys():
        compute_and_save_stats(split=split, save_dir=save_dir)
    print("done")