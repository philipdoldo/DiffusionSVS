import sys
sys.path.insert(0, "third_party/svs_datasets/code")
from third_party.svs_datasets.code import  CanonicalExample, Dataset, Interval, NoteInterval, load_dataset
import numpy as np
import librosa
import parselmouth
import argparse
import toml
import json
import h5py
import os
import pprint
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

PAD_TOKEN = "<PAD>"
SILENCE_TOKEN = "SP" # we explicitly do NOT want the brackets, that is we want "SP" and NOT "<SP>" when using `svs_datasets`
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"



def get_dataset_paths(config, log_file):
    """
    Reads a config for the paths to every dataset handled by `svs_datasets`, returns a dict
    If a dataset path is None, prints a warning.
    """
    dataset_paths = {
        Dataset.ALEX_FLOAREA_AI_SVS : config.get('ALEX_FLOAREA_AI_SVS_path'),
        Dataset.ALEX_FLOAREA_EN_PUBLIC : config.get('ALEX_FLOAREA_EN_PUBLIC_path'),
        Dataset.PROJECT_AIDOL_PUBLIC_ENGLISH : config.get('PROJECT_AIDOL_PUBLIC_ENGLISH_path'),
        Dataset.NGYY : config.get('NGYY_path'),
        Dataset.OPEN_CPOP : config.get('OPEN_CPOP_path'),
        Dataset.M4SINGER : config.get('M4SINGER_path'),
        Dataset.NO7SINGING : config.get('NO7SINGING_path'),
        Dataset.RITSU : config.get('RITSU_path'),
        Dataset.POPCS : config.get('POPCS_path'),
        Dataset.SUNG_AND_SPOKEN : config.get('SUNG_AND_SPOKEN_path'),
        Dataset.GTSINGER_ENGLISH : config.get('GTSINGER_ENGLISH_path'),
        Dataset.GTSINGER_CHINESE : config.get('GTSINGER_CHINESE_path'),
        Dataset.GTSINGER_JAPANESE : config.get('GTSINGER_JAPANESE_path'),
        Dataset.AMABOSHI_CIPHERDB :  config.get('AMABOSHI_CIPHERDB_path'),
        Dataset.KUROTAKE_KOUGA_AI_SONG : config.get('KUROTAKE_KOUGA_AI_SONG_path'),
        Dataset.OFUTON_P_UTAGOE_DB : config.get('OFUTON_P_UTAGOE_DB_path'),
        Dataset.ONIKU_KURUMI_UTAGOE_DB : config.get('ONIKU_KURUMI_UTAGOE_DB_path'),
        Dataset.PJS_CORPUS : config.get('PJS_CORPUS_path'),
        Dataset.ENUNU_KODOKU : config.get('ENUNU_KODOKU_path'),
        Dataset.ITAKO_SINGING : config.get('ITAKO_SINGING_path'),
        Dataset.NIT070_DB : config.get('NIT070_DB_path'),
        Dataset.TIGER_EN : config.get('TIGER_EN_path'),
        Dataset.TIGER_JP : config.get('TIGER_JP_path'),
    }

    seen_paths = set() # quickly check for duplicate paths, just as a sanity check -- we don't want the same path to correspond to different datasets, we'd likely catch this error later anyway when checking if `load_dataset` gives us an empty list
    for dataset, path in dataset_paths.items():
        if path in seen_paths:
            raise ValueError(f"The same path corresponds to multiple different datasets, got {path=}, check your config!")
        if path is None:
            log_and_print(f"WARNING: {dataset=} has {path=}, check your config!", log_file=log_file)
        else:
            seen_paths.add(path)
    return dataset_paths
    
def collect_examples(dataset_paths: dict) -> list[CanonicalExample]:
    examples = []
    for dataset, path in dataset_paths.items():
        if path is None:
            continue
        dataset_examples = load_dataset(dataset, path)
        if len(dataset_examples) == 0:
            raise ValueError(f"Failed to properly load dataset with {dataset=}, {path=}, check your config!")
        examples.extend(dataset_examples)
    return examples

def build_phoneme_vocab(examples: list[CanonicalExample], save_dir: Path, log_file) -> dict[str, int]:
    """
    Collects every distinct phoneme label across all examples and assigns stable integer ids.
    Special tokens are prepended first so their ids never shift. Saved to <save_dir>/vocab.json.
    """
    special_tokens = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SILENCE_TOKEN]
    vocab = set()
    special_counts = {} # sanity check, stores number of times each special token shows up in the data 

    for ex in examples:
        for interval in ex.phone_intervals:
            phoneme = interval.label.strip()
            assert len(phoneme) > 0, "sanity check"
            if phoneme in special_tokens:
                special_counts[phoneme] = special_counts.get(phoneme, 0) + 1
            if phoneme and phoneme not in special_tokens:
                vocab.add(phoneme)

    vocab = special_tokens + sorted(vocab)
    vocab = {token: i for i, token in enumerate(vocab)}
    assert vocab[PAD_TOKEN] == 0, f"{vocab[PAD_TOKEN]=}, {vocab=}"

    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "vocab.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    log_and_print(f"phoneme vocabulary: {len(vocab)} tokens (including special tokens: {special_tokens})", log_file=log_file)
    log_and_print(f"  Sanity Check: {special_counts=}", log_file=log_file)
    log_and_print(f"saved to {save_path}", log_file=log_file)
    return vocab

def build_splits(examples: list[CanonicalExample], save_dir: Path, val_ids: list[tuple[str, str]], test_ids: list[tuple[str, str]], log_file) -> dict[str, list[CanonicalExample]]:
    """
    val_ids/test_ids are lists of (source_dataset, utterance_id) pairs, e.g. ('popcs', 'popcs-Bad/0000').
    """
    val_ids = set(val_ids)
    test_ids = set(test_ids)
    splits = {"train": [], "val": [], "test": []}

    for ex in examples:
        key = (ex.source_dataset, ex.utterance_id)
        if key in test_ids:
            splits["test"].append(ex)
        elif key in val_ids:
            splits["val"].append(ex)
        else:
            splits["train"].append(ex)

    save_dir.mkdir(parents=True, exist_ok=True)
    with open(save_dir / "splits.json", "w", encoding="utf-8") as f:
        json.dump({split: [[ex.source_dataset, ex.utterance_id] for ex in split_examples]
                   for split, split_examples in splits.items()}, f, ensure_ascii=False, indent=2)

    for split, split_examples in splits.items():
        log_and_print(f"{split:>5}: {len(split_examples)} utterances", log_file=log_file)

    return splits


def process_utterance(
    example:        CanonicalExample,
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
    audio, sr = librosa.load(example.audio_path, sr=None, mono=True)
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
        raise ValueError(f"All frames of f0 are unvoiced, {example.audio_path=}, {example.utterance_id=}")

    f0_raw = np.where(np.logical_not(uv), np.clip(f0_raw, f0_config["f0_min"], f0_config["f0_max"]), f0_raw) # this might not be needed since it might already be clipped, but doing it just in case
    f0_mel = np.where(np.logical_not(uv), 1127 * np.log(1 + f0_raw / 700), 0.0)

    # linearly interpolate log f0 across unvoiced gaps
    voiced_indices = np.where(np.logical_not(uv))[0]
    f0 = np.interp(x=np.arange(T), xp=voiced_indices, fp=f0_mel[voiced_indices]).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. mel2ph and txt_tokens from svs_datasets' phone_intervals
    # ------------------------------------------------------------------
    # svs_datasets already normalizes silence/BOS/EOS itself (never emits empty-string labels --
    # verified via the assert in build_phoneme_vocab), so no BOS/EOS/SP synthesis needed here.
    intervals = example.phone_intervals

    txt_tokens = np.array([phoneme_to_idx[iv.label.strip()] for iv in intervals], dtype=np.int32)

    starts = [int(np.round(iv.start_sec * audio_config["sample_rate"] / audio_config["hop_size"])) for iv in intervals]
    ends = starts[1:] + [T]

    if starts[0] != 0:
        raise BinarizationError(f"first phoneme interval doesn't start at 0 ({starts[0]=}, {example.phone_intervals[0].start_sec=})")

    mel2ph = np.zeros(T, dtype=np.int32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        mel2ph[s:e] = i

    starts_ok = _is_nondecreasing(starts)
    ends_ok = _is_nondecreasing(ends)
    mel2ph_ok = _is_nondecreasing(mel2ph)
    final_ok = bool(mel2ph.max() == len(txt_tokens) - 1)

    if not (starts_ok and ends_ok and mel2ph_ok and final_ok):
        # skip if the data isn't formatted properly, e.g. `popcs/popcs-爱你十分泪七分/0015.TextGrid` has final xmax of 11.819999999999993 but `/popcs/popcs-爱你十分泪七分/0015_wf0.wav` is only 11.42328798185941 seconds
        raise BinarizationError(f"{starts_ok=}, {ends_ok=}, {mel2ph_ok=},  {final_ok=}, {mel2ph.max()=}, {len(txt_tokens)=}, {T=}, {starts[:3]=}, {starts[-3:]=}, {ends[:3]=}, {ends[-3:]=}, {example.phone_intervals[0]=}, {example.phone_intervals[-1]=}")

    # max_mel_frames = config["data"].get("max_mel_frames") # truncate to max_mel_frames
    # if max_mel_frames is not None and T > max_mel_frames:
    #     log_mel    = log_mel[:, :max_mel_frames]
    #     f0         = f0[:max_mel_frames]
    #     uv         = uv[:max_mel_frames]
    #     mel2ph     = mel2ph[:max_mel_frames]
    #     txt_tokens = txt_tokens[:mel2ph[-1] + 1]

    return {
        "mel":        log_mel,    # float32 (n_mels, T)
        "f0":         f0,         # float32 (T,)
        "uv":         uv,         # bool    (T,)
        "mel2ph":     mel2ph,     # int32   (T,)
        "txt_tokens": txt_tokens, # int32   (P,)
    }


def compute_segment_boundaries(phone_intervals, min_sec, max_sec):
    """
    Greedily extends each segment with as many subsequent phoneme intervals as possible while
    staying <= max_sec, then starts a new segment at the next interval. Cuts always land exactly
    on phoneme boundaries.

    Returns (start_idx, end_idx) pairs -- some may still be under min_sec or (only if a single
    interval alone exceeds max_sec) over max_sec. This function only proposes cuts; call
    validate_segments() afterward to separate good segments from bad ones.
    """
    n = len(phone_intervals)
    boundaries = []
    seg_start = 0
    while seg_start < n:
        j = seg_start + 1
        while j < n:
            candidate_dur = phone_intervals[j].end_sec - phone_intervals[seg_start].start_sec
            if candidate_dur <= max_sec:
                j += 1
            else:
                break
        boundaries.append((seg_start, j))
        seg_start = j

    # try to rescue an undersized final segment by merging it into the previous one
    if len(boundaries) >= 2:
        s_last, e_last = boundaries[-1]
        dur_last = phone_intervals[e_last - 1].end_sec - phone_intervals[s_last].start_sec
        if dur_last < min_sec:
            s_prev, e_prev = boundaries[-2]
            merged_dur = phone_intervals[e_last - 1].end_sec - phone_intervals[s_prev].start_sec
            if merged_dur <= max_sec:
                boundaries[-2] = (s_prev, e_last)
                boundaries.pop()

    return boundaries


def validate_segments(boundaries, phone_intervals, min_sec, max_sec):
    """
    Splits proposed boundaries into (valid, invalid). A segment is invalid if it's over max_sec
    (only possible when a single interval alone is that long -- unfixable by any choice of cuts)
    or under min_sec (can happen when a short interval is followed by one long enough that
    merging would push it over max_sec). Invalid segments are dropped individually -- they do
    NOT affect any other segment from the same utterance.
    """
    valid, invalid = [], []
    for s, e in boundaries:
        dur = phone_intervals[e - 1].end_sec - phone_intervals[s].start_sec
        if dur > max_sec:
            invalid.append((s, e, "over max_sec", dur))
        elif dur < min_sec:
            invalid.append((s, e, "under min_sec", dur))
        else:
            valid.append((s, e))
    return valid, invalid




def split_features(features: dict, phone_intervals, boundaries: list[tuple[int, int]]) -> list[dict]:
    """
    Slices the full-utterance features returned by process_utterance into per-segment features,
    using mel2ph (already frame-aligned) to find each segment's frame range -- never re-loads or
    re-computes anything from audio.
    """
    mel2ph = features["mel2ph"]
    segments = []
    for seg_idx, (s, e) in enumerate(boundaries):
        f0_frame = int(np.searchsorted(mel2ph, s, side="left"))
        f1_frame = int(np.searchsorted(mel2ph, e - 1, side="right"))
        segments.append({
            "mel":           features["mel"][:, f0_frame:f1_frame],
            "f0":            features["f0"][f0_frame:f1_frame],
            "uv":            features["uv"][f0_frame:f1_frame],
            "mel2ph":        features["mel2ph"][f0_frame:f1_frame] - s,  # reindex to segment-local phoneme ids
            "txt_tokens":    features["txt_tokens"][s:e],
            "segment_index": seg_idx,
            "start_sec":     phone_intervals[s].start_sec,
            "end_sec":       phone_intervals[e - 1].end_sec,
        })
    return segments

def binarize_split(
    split:          str,
    split_examples: list[CanonicalExample],
    save_dir:       Path,
    phoneme_to_idx: dict[str, int],
    config:         dict,
    log_file
):
    """
    `split_examples` is the list of CanonicalExamples for just this split (already filtered by
    build_splits). Each utterance is first fully processed once via process_utterance, then cut
    into segments per config's min_sec/max_sec. Each SEGMENT (not utterance) gets an opaque,
    zero-padded sequential id ('000000', '000001', ...) as its h5 group key -- ids are sequential
    over segments, since one utterance can produce multiple segments. The original dataset name,
    utterance id, and segment provenance are kept as attributes on the group for traceability, not
    as part of the key.
    """
    min_sec = config["data"]["min_sec"]
    max_sec = config["data"]["max_sec"]

    # segment boundaries only depend on phone_intervals (start/end times), not on audio/mel -- so
    # we can precompute every utterance's boundaries up front, purely to get an exact total segment
    # count for id_width, before doing any of the actual (slow) audio processing below. Utterances
    # that fail here (BinarizationError) are recorded as skipped now, same as process_utterance
    # failures below -- one bad utterance must never crash the whole split.
    boundaries_by_example = []       # list of VALID (start_idx, end_idx) lists, one per example
    invalid_segments_log  = []       # list of dicts, for inspection -- segments dropped individually
    skipped = []                     # utterance_ids skipped entirely (only from process_utterance failures now)

    for ex in split_examples:
        proposed = compute_segment_boundaries(ex.phone_intervals, min_sec, max_sec)
        valid, invalid = validate_segments(proposed, ex.phone_intervals, min_sec, max_sec)
        boundaries_by_example.append(valid)
        for s, e, reason, dur in invalid:
            log_and_print(f"dropping segment [{s}:{e}] ({dur:.1f}s, {reason}) from {ex.source_dataset}/{ex.utterance_id}", log_file=log_file)
            invalid_segments_log.append({
                "source_dataset": ex.source_dataset, "utterance_id": ex.utterance_id,
                "start_idx": s, "end_idx": e, "reason": reason, "duration_sec": dur,
            })

    total_segments = sum(len(b) for b in boundaries_by_example)
    id_width = max(1, len(str(total_segments - 1))) if total_segments > 0 else 1

    successful = []

    with h5py.File(save_dir / f"{split}.h5", "w") as f:
        counter = 0
        for example, boundaries in zip(tqdm(split_examples, desc=split), boundaries_by_example):
            if not boundaries:
                continue  # every proposed segment for this utterance was invalid

            try:
                result = process_utterance(
                    example=example,
                    phoneme_to_idx=phoneme_to_idx,
                    config=config,
                )
            except BinarizationError as e:
                log_and_print(f"skipping {example.source_dataset}/{example.utterance_id}: {e}", log_file=log_file)
                log_and_print(example, log_file=log_file, pretty=True)
                skipped.append(example.utterance_id)
                continue

            segments = split_features(result, example.phone_intervals, boundaries)

            for seg in segments:
                item_id = f"{counter:0{id_width}d}"
                counter += 1

                grp = f.create_group(item_id)
                grp.attrs["source_dataset"] = example.source_dataset
                grp.attrs["utterance_id"]   = example.utterance_id
                grp.attrs["segment_index"]  = seg["segment_index"]
                grp.attrs["start_sec"]      = seg["start_sec"]
                grp.attrs["end_sec"]        = seg["end_sec"]
                grp.create_dataset("mel",        data=seg["mel"])
                grp.create_dataset("f0",         data=seg["f0"])
                grp.create_dataset("uv",         data=seg["uv"])
                grp.create_dataset("mel2ph",     data=seg["mel2ph"])
                grp.create_dataset("txt_tokens", data=seg["txt_tokens"])
                successful.append({
                    "id": item_id, "source_dataset": example.source_dataset,
                    "utterance_id": example.utterance_id, "segment_index": seg["segment_index"],
                    "start_sec": seg["start_sec"], "end_sec": seg["end_sec"],
                })

    splits_dir = save_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    with open(splits_dir / f"{split}_items.json", "w", encoding="utf-8") as f:
        json.dump(successful, f, ensure_ascii=False, indent=2)

    with open(splits_dir / f"{split}_skipped.json", "w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)

    with open(splits_dir / f"{split}_invalid_segments.json", "w", encoding="utf-8") as f:
        json.dump(invalid_segments_log, f, ensure_ascii=False, indent=2)

    log_and_print(f"\n\n{split}: {len(successful)} segments saved (from {len(split_examples) - len(skipped)} utterances), {len(skipped)} utterances fully skipped, {len(invalid_segments_log)} individual segments dropped\n\n", log_file=log_file)

class BinarizationError(Exception):
    pass


def _is_nondecreasing(arr):
    return bool(np.all(np.diff(arr) >= 0))


def compute_and_save_stats(split: str, save_dir: Path, log_file):
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
    log_and_print(f"{split}: stats saved to {save_dir / f'{split}_stats.npz'}", log_file=log_file)


def create_data_dir(parent_dir, config_name):
    timestamp = datetime.now().strftime("%m-%d-%Y-%Hh%Mm%Ss")
    data_dir = os.path.join(parent_dir, config_name, f"{timestamp}")
    os.makedirs(data_dir, exist_ok=True)
    return Path(data_dir)


def is_paired_speech(example: CanonicalExample) -> bool:
    """
    GTSinger datasets include a 'Paired_Speech_Group' subset that's spoken, not sung, audio --
    exclude it from an SVS pipeline. Checked by dataset name, not just utterance_id, since other
    datasets could coincidentally contain that substring without meaning the same thing.
    """
    gtsinger_datasets = {"GTSinger_English", "GTSinger_Chinese", "GTSinger_Japanese"}
    return example.source_dataset in gtsinger_datasets and "Paired_Speech_Group" in example.utterance_id


def log_and_print(text, log_file, pretty=False):
    """`text` is a string to print and write to the log file"""
    if pretty:
        text = pprint.pformat(text)
    print(text)
    with open(log_file, 'a') as f:
        f.write(text + "\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help=".toml file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = toml.load(f)

    save_dir = create_data_dir(parent_dir=config["data"]["save_dir"], config_name=Path(args.config).stem)

    with open(os.path.join(save_dir, "config.toml"), "w") as f:
        toml.dump(config, f) # save copy of config in binarized data directory

    log_file = os.path.join(save_dir, "log.txt")
    with open(log_file, "w") as f:
        f.write("") # initialize log file

    dataset_paths = get_dataset_paths(config['data'], log_file=log_file)
    val_ids = [tuple(pair) for pair in config['data']["val_ids"]]
    test_ids = [tuple(pair) for pair in config['data']["test_ids"]]

    examples = collect_examples(dataset_paths)
    n_before = len(examples)
    examples = [ex for ex in examples if not is_paired_speech(ex)]
    log_and_print(f"filtered out {n_before - len(examples)} paired-speech examples, {len(examples)} remain\n\n", log_file=log_file)

    vocab = build_phoneme_vocab(examples=examples, save_dir=save_dir, log_file=log_file)

    splits = build_splits(
        examples=examples,
        save_dir=save_dir,
        val_ids=val_ids,
        test_ids=test_ids,
        log_file=log_file
    )

    for split, split_examples in splits.items():
        binarize_split(
            split=split,
            split_examples=split_examples,
            save_dir=save_dir,
            phoneme_to_idx=vocab,
            config=config,
            log_file=log_file
        )

    for split in splits.keys():
        compute_and_save_stats(split=split, save_dir=save_dir, log_file=log_file)
    print("done")