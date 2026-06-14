import sys
sys.path.insert(0, "third_party/DiffSinger-vocoder/code")

import yaml
import torch
import numpy as np
import h5py
import soundfile as sf
from modules.hifigan.hifigan import HifiGanGenerator

CKPT_PATH   = "third_party/DiffSinger-vocoder/model_ckpt_steps_280000.ckpt" # checkpoint from the .zip file downloaded
CONFIG_PATH = "third_party/DiffSinger-vocoder/config.yaml" # config from the .zip file downloaded
TEST_H5     = "/home/phil/DiffusionSVS/binarized_data/binarize-PopCS/06-14-2026-11h08m51s/test.h5"##"/home/phil/DiffusionSVS/binarized_data/binarize-PopCS/06-13-2026-19h51m02s/test.h5"
OUTPUT_PATH = "_output/output.wav"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)

ckpt = torch.load(CKPT_PATH, map_location=device)
model = HifiGanGenerator(config)
model.load_state_dict(ckpt["state_dict"]["model_gen"], strict=True)
model.remove_weight_norm()
model.eval().to(device)

with h5py.File(TEST_H5, "r") as f:
    key = list(f.keys())[0]
    mel = f[key]["mel"][:]  # (n_mels, T), log mel
    f0  = f[key]["f0"][:]   # (T,), mel-scale log f0

train_stats_path = "/home/phil/DiffusionSVS/binarized_data/binarize-PopCS/06-14-2026-11h08m51s/train_stats.npz"##"/home/phil/DiffusionSVS/binarized_data/binarize-PopCS/06-13-2026-19h51m02s/train_stats.npz"
stats = np.load(train_stats_path)
# mel_min = stats['mel_min'][:, None] # (M, 1)
# mel_max = stats['mel_max'][:, None] # (M, 1)
# mel = 2 * (mel - mel_min) / (mel_max - mel_min) - 1
# mel_mean = stats['mel_mean'][:, None] # (M, 1)
# mel_std = stats['mel_std'][:, None] # (M, 1)
# mel = (mel - mel_mean) / mel_std

# convert stored mel-scale f0 back to Hz for the NSF vocoder
f0_hz = 700.0 * (np.exp(f0 / 1127.0) - 1.0)

mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).to(device)    # (1, n_mels, T)
f0_tensor  = torch.from_numpy(f0_hz).float().unsqueeze(0).to(device)  # (1, T)

with torch.no_grad():
    print(f"{mel.min()=}, {mel.max()=}, {mel.mean()=}") 
    # mel values should be in [-6, 1.5] because in config.yaml they define `mel_vmin: -6` and `mel_vmax: 1.5`
    audio = model(mel_tensor, f0_tensor).view(-1)

audio = audio.cpu().numpy()
sf.write(OUTPUT_PATH, audio, samplerate=config["audio_sample_rate"])
print(f"wrote {OUTPUT_PATH}")