
"""
    When building mel2ph we sometimes encounter TextGrid intervals with emptry-string text fields, e.g. interval 53 in the example below:

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
import tomllib
import json
import h5py
from pathlib import Path
from tqdm import tqdm

SILENCE_TOKEN = "SP"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"

class BinarizationError(Exception): # TODO delete this
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
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=audio_config["sample_rate"],
        n_fft=audio_config["window_size"],
        hop_length=audio_config["hop_size"],
        win_length=audio_config["window_size"],
        n_mels=audio_config["n_mels"],
        fmin=audio_config["min_freq"],
        fmax=audio_config["max_freq"],
        center=audio_config["mel_center"],
    )
    log_mel = np.log(np.clip(mel, a_min=audio_config["mel_log_clip"], a_max=None)) # (n_mels, T)
    log_mel = log_mel.T.astype(np.float32)  # (T, n_mels)
    T = log_mel.shape[0]

    # ------------------------------------------------------------------
    # 3. f0
    # ------------------------------------------------------------------
    frame_times = np.arange(T) * audio_config["hop_size"] / audio_config["sample_rate"]

    sound   = parselmouth.Sound(audio, sampling_frequency=audio_config["sample_rate"])
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
    f0_log = np.where(np.logical_not(uv), np.log(f0_raw + 1e-8), 0.0)

    # linearly interpolate log f0 across unvoiced gaps
    voiced_indices = np.where(np.logical_not(uv))[0]
    f0 = np.interp(x=np.arange(T), xp=voiced_indices, fp=f0_log[voiced_indices]).astype(np.float32)

    # ------------------------------------------------------------------
    # 4. mel2ph and txt_tokens from TextGrid
    # ------------------------------------------------------------------
    tg   = textgrid.TextGrid.fromFile(tg_path)
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

    #print(f"{len(txt_tokens)=}, {len(starts)=}, {len(ends)=}, {T=}, {wav_path=}, {tg_path=}")
    assert starts[0] == 0, f"{starts=}"

    mel2ph = np.zeros(T, dtype=np.int32)
    for i, (s, e) in enumerate(zip(starts, ends)):
        mel2ph[s:e] = i # TODO check this #i + 1  # 1-based; 0 is the padding sentinel # TODO this seems stupid, fix this??

    #assert mel2ph.min() >= 1, "mel2ph contains zeros after gap fill" # TODO bad check
    #assert mel2ph.max() == len(txt_tokens) - 1, f"mel2ph.max() {mel2ph.max()} != len(txt_tokens) {len(txt_tokens)} -- {len(txt_tokens)=}, {len(starts)=}, {len(ends)=}, {T=}, {wav_path=}, {tg_path=} -- {mel2ph=}, {txt_tokens=}, {starts=}, {ends=}" # TODO bad check, I added - 1

    if (not _is_nondecreasing(starts)) or (not _is_nondecreasing(ends)) or (not _is_nondecreasing(mel2ph)) or (mel2ph.max() != len(txt_tokens) - 1):
        raise BinarizationError # skip if the data isn't formatted properly, e.g. `popcs/popcs-爱你十分泪七分/0015.TextGrid` has final xmax of 11.819999999999993 but `/popcs/popcs-爱你十分泪七分/0015_wf0.wav` is only 11.42328798185941 seconds

    return {
        "mel":        log_mel,    # float32 (T, n_mels)
        "f0":         f0,         # float32 (T,)
        "uv":         uv,         # bool    (T,)
        "mel2ph":     mel2ph,     # int32   (T,)
        "txt_tokens": txt_tokens, # int32   (P,)
    }

def item_name_to_paths(item_name: str, raw_data_dir: Path) -> tuple[Path, Path]:
    # item_name e.g. "popcs-Bad-0000"
    song_name = item_name.rsplit("-", 1)[0]   # "popcs-Bad"
    utt_index = item_name.rsplit("-", 1)[1]   # "0000"
    song_dir  = raw_data_dir / song_name
    wav_path  = song_dir / f"{utt_index}_wf0.wav"
    tg_path   = song_dir / f"{utt_index}.TextGrid"
    return wav_path, tg_path


def binarize_split(
    split:          str,
    item_names:     list[str],
    raw_data_dir:   Path,
    save_dir: Path,
    phoneme_to_idx: dict[str, int],
    config:         dict,
):
    successful = []
    skipped    = []

    with h5py.File(save_dir / f"{split}.h5", "w") as f:
        for item_name in tqdm(item_names, desc=split):
            wav_path, tg_path = item_name_to_paths(item_name, raw_data_dir)

            try:
                result = process_utterance(wav_path=str(wav_path), tg_path=str(tg_path), phoneme_to_idx=phoneme_to_idx, config=config)
            except BinarizationError as e:
                print(f"skipping {item_name}: {e}")
                skipped.append(item_name)
                continue

            grp = f.create_group(item_name)
            grp.create_dataset("mel",        data=result["mel"])
            grp.create_dataset("f0",         data=result["f0"])
            grp.create_dataset("uv",         data=result["uv"])
            grp.create_dataset("mel2ph",     data=result["mel2ph"])
            grp.create_dataset("txt_tokens", data=result["txt_tokens"])
            successful.append(item_name)

    splits_dir = save_dir / "splits"
    splits_dir.mkdir(exist_ok=True)

    lengths = []
    with h5py.File(save_dir / f"{split}.h5", "r") as f:
        for item_name in successful:
            lengths.append(f[item_name]["mel"].shape[0])

    np.save(splits_dir / f"{split}_lengths.npy", np.array(lengths, dtype=np.int32))
    #json.dump(successful, open(splits_dir / f"{split}_items.json", "w", encoding="utf-8"))
    with open(splits_dir / f"{split}_items.json", "w", encoding="utf-8") as f:
        json.dump(successful, f, ensure_ascii=False, indent=2)
    
    with open(splits_dir / f"{split}_skipped.json", "w", encoding="utf-8") as f:
        json.dump(skipped, f, ensure_ascii=False, indent=2)

    print(f"{split}: {len(successful)} saved, {len(skipped)} skipped")
    if skipped:
        print(f"  skipped: {skipped}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help=".toml file")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)

    raw_data_dir = Path(config["data"]["raw_data_dir"])
    save_dir = Path(config["data"]["save_dir"])

    phoneme_to_idx = json.load(open(save_dir / "vocab.json", encoding="utf-8"))
    splits = json.load(open(save_dir / "splits.json", encoding="utf-8"))

    for split, item_names in splits.items():
        binarize_split(
            split=split, 
            item_names=item_names, 
            raw_data_dir=raw_data_dir, 
            save_dir=save_dir, 
            phoneme_to_idx=phoneme_to_idx, 
            config=config
            )

