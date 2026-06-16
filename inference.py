import sys
sys.path.insert(0, "third_party/DiffSinger-vocoder/code")

import yaml
import torch
import numpy as np
import h5py
import toml
import soundfile as sf
from modules.hifigan.hifigan import HifiGanGenerator
from data.dataloader import NaiveDataLoader
from model import WaveNetDenoiser
from train import DiffusionProcess

if __name__ == "__main__":

    CKPT_PATH   = "third_party/DiffSinger-vocoder/model_ckpt_steps_280000.ckpt" # checkpoint from the .zip file downloaded
    CONFIG_PATH = "third_party/DiffSinger-vocoder/config.yaml" # config from the .zip file downloaded
    TEST_H5     = "/home/phil/DiffusionSVS/binarized_data/binarize-PopCS/06-14-2026-11h08m51s/test.h5"##"/home/phil/DiffusionSVS/binarized_data/binarize-PopCS/06-13-2026-19h51m02s/test.h5"
    OUTPUT_PATH = "_output/output.wav"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    ckpt = torch.load(CKPT_PATH, map_location=device)
    vocoder = HifiGanGenerator(config)
    vocoder.load_state_dict(ckpt["state_dict"]["model_gen"], strict=True)
    vocoder.remove_weight_norm()
    vocoder.eval().to(device)

    denoiser_checkpoint_path = "/mnt/data_r60_1/adv_robust_project/DiffusionSVS/experiments/_train-wavenet-denoiser/06-15-2026-16h33m52s/checkpoints/checkpoint_step159999.pt"
    denoiser_checkpoint = torch.load(denoiser_checkpoint_path, map_location=device)
    denoiser_config_path = "/mnt/data_r60_1/adv_robust_project/DiffusionSVS/experiments/_train-wavenet-denoiser/06-15-2026-16h33m52s/config.toml"
    with open(denoiser_config_path, "r") as f:
        denoiser_config = toml.load(f)
    denoiser = WaveNetDenoiser(denoiser_config['model'])

    train_stats_path = "/mnt/data_r60_1/adv_robust_project/DiffusionSVS/binarized_data/_binarize-PopCS/06-15-2026-02h12m44s/train_stats.npz"
    stats = np.load(train_stats_path)
    test_loader = NaiveDataLoader(data_path=TEST_H5, batch_size=1, padding_value=0, rng_seed=21, diffusion_k=100, stats_path=train_stats_path)

    batch = test_loader.next_batch() # maybe create dummy .h5 of just a single example

    beta_min = 1e-4
    beta_max = 0.06
    diffusion = DiffusionProcess(
            beta_min=beta_min, 
            beta_max=beta_max, 
            T=100,
            device=device
            )
    
    mel = diffusion.sample(
        model=denoiser, 
        txt_tokens=batch['txt_tokens'], 
        mel2ph=batch['mel2ph'], 
        f0=batch['f0'], 
        uv=batch['uv'], 
        ph_padding_mask=batch['ph_padding_mask'], 
        mel_padding_mask=batch['mel_padding_mask'],
        M=80
        )
    
    # normalize mel-spectrograms to [-1, 1]
    mel_min = stats['mel_min'][None, :, None] # (1, M, 1)
    mel_max = stats['mel_max'][None, :, None] # (1, M, 1)

    mel = ((mel + 1)/2) * (mel_max - mel_min) + mel_min # undo the normalization done inside the dataloader, which was: mel = 2 * (mel - mel_min) / (mel_max - mel_min) - 1

    f0 = batch['f0'] * stats['f0_std'] + stats['f0_mean'] # undo the standardization to mean 0 std 1 done in the dataloader, which was: f0 = (f0 - stats['f0_mean']) / stats['f0_std'] # scalar mean and std should broadcast properly

    # convert stored mel-scale f0 back to Hz for the NSF vocoder
    f0_hz = 700.0 * (np.exp(f0 / 1127.0) - 1.0)

    mel_tensor = torch.from_numpy(mel).float().unsqueeze(0).to(device)    # (1, n_mels, T)
    f0_tensor  = torch.from_numpy(f0_hz).float().unsqueeze(0).to(device)  # (1, T)

    with torch.no_grad():
        print(f"{mel.min()=}, {mel.max()=}, {mel.mean()=}") 
        # mel values should be in [-6, 1.5] because in config.yaml they define `mel_vmin: -6` and `mel_vmax: 1.5`
        audio = vocoder(mel_tensor, f0_tensor).view(-1)

    audio = audio.cpu().numpy()
    sf.write(OUTPUT_PATH, audio, samplerate=config["audio_sample_rate"])
    print(f"wrote {OUTPUT_PATH}")