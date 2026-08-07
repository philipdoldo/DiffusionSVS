import sys
sys.path.insert(0, "third_party/DiffSinger-vocoder/code")

import yaml
import torch
import numpy as np
import h5py
import toml
import argparse
import soundfile as sf
from pathlib import Path
from modules.hifigan.hifigan import HifiGanGenerator
from dataloader import NaiveDataLoader
from model import WaveNetDenoiser, DiT
from train import DiffSingerDiffusion, SimpleFlow
from exponential_moving_average import ExponentialMovingAverage

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot training/validation loss from a CSV log.")
    parser.add_argument("checkpoint", help="Path to denoiser checkpoint")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True, help="Whether or not to use EMA weights")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num_iter", type=int, default=100)
    parser.add_argument("--cfg_scale", type=float, default=None)

    args = parser.parse_args()

    VOCODER_CHECKPOINT_PATH = "third_party/DiffSinger-vocoder/model_ckpt_steps_280000.ckpt" # checkpoint from the .zip file downloaded
    VOCODER_CONFIG_PATH = "third_party/DiffSinger-vocoder/config.yaml" # config from the .zip file downloaded
    ### TODO infer both test h5 and train stats from same data dir input, maybe create subdir to store audio output in with a file showing what dataset what used for generating the inference output
    ###TEST_H5 = "/mnt/data_r60_1/adv_robust_project/DiffusionSVS/binarized_data/_binarize-popcs-m4singer-gtsinger/07-13-2026-05h16m12s/test.h5"###"/mnt/data_r60_1/adv_robust_project/DiffusionSVS/binarized_data/_binarize-PopCS/06-15-2026-02h12m44s/test.h5"#"/home/phil/DiffusionSVS/binarized_data/binarize-PopCS/06-16-2026-16h12m26s/test.h5"#"/home/phil/DiffusionSVS/binarized_data/binarize-PopCS/06-14-2026-11h08m51s/test.h5"##"/home/phil/DiffusionSVS/binarized_data/binarize-PopCS/06-13-2026-19h51m02s/test.h5"
    # TODO, need to change the training stats path to be based on the config since it'll be different for different binarized datasets!
    ###TRAIN_STATS_PATH = "/mnt/data_r60_1/adv_robust_project/DiffusionSVS/binarized_data/_binarize-popcs-m4singer-gtsinger/07-13-2026-05h16m12s/train_stats.npz"###"/mnt/data_r60_1/adv_robust_project/DiffusionSVS/binarized_data/_binarize-PopCS/06-15-2026-02h12m44s/train_stats.npz"#"/home/phil/DiffusionSVS/binarized_data/binarize-PopCS/06-16-2026-16h12m26s/train_stats.npz"
    #OUTPUT_PATH = "_output/inference_output0-ok10-160k-dit-ema.wav"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(VOCODER_CONFIG_PATH) as f:
        vocoder_config = yaml.safe_load(f)

    vocoder_checkpoint = torch.load(VOCODER_CHECKPOINT_PATH, map_location=device)
    vocoder = HifiGanGenerator(vocoder_config)
    vocoder.load_state_dict(vocoder_checkpoint["state_dict"]["model_gen"], strict=True)
    vocoder.remove_weight_norm()
    vocoder.eval().to(device)

    denoiser_checkpoint_path = Path(args.checkpoint)
    output_dir = denoiser_checkpoint_path.parent
    if args.config is None:
        denoiser_config_path = output_dir / "config.toml"
    else:
        denoiser_config_path = args.config
    print(f"  LOADING DENOISER CHECKPOINT: {denoiser_checkpoint_path}")
    print(f"  USING DENOISER CONFIG: {denoiser_config_path}")

    denoiser_checkpoint = torch.load(denoiser_checkpoint_path, map_location=device)
    with open(denoiser_config_path, "r") as f:
        denoiser_config = toml.load(f)
    denoiser_data_path = Path(denoiser_config['training']['data_dir'])
    TEST_H5 = denoiser_data_path / "test.h5"
    TRAIN_STATS_PATH = denoiser_data_path / "train_stats.npz"

    model_type = denoiser_config['model']['model_type'] 
    if model_type == "DiT":
        model_class = DiT
    elif model_type == "WaveNetDenoiser":
        model_class = WaveNetDenoiser
    else:
        raise ValueError(f"{model_type=}")
    denoiser = model_class(denoiser_config['model'])
    denoiser.load_state_dict(denoiser_checkpoint['model'])
    denoiser.eval().to(device)

    if args.ema:
        ema = ExponentialMovingAverage(params=denoiser.parameters(), decay=denoiser_config["training"]["ema_decay"])
        ema.load_state_dict(denoiser_checkpoint["ema"], device=device)
        ema.store(denoiser.parameters()) # store copy of the actual model weights
        ema.copy_to(denoiser.parameters()) # copy EMA weights into the model

    #"/home/phil/DiffusionSVS/binarized_data/binarize-PopCS/06-14-2026-11h08m51s/train_stats.npz" #"/mnt/data_r60_1/adv_robust_project/DiffusionSVS/binarized_data/_binarize-PopCS/06-15-2026-02h12m44s/train_stats.npz"
    stats = np.load(TRAIN_STATS_PATH)

    diffusion_type = denoiser_config['training']['diffusion'].get('diffusion_type', 'DiffSingerDiffusion')
    diffusion_k = denoiser_config['training']['diffusion'].get('k')
    mel_pad_multiple = denoiser_config['training'].get('mel_pad_multiple', 1)
    print(f"\n     ----- USING {diffusion_type=} -----\n")
    print(f"{diffusion_k=}")
    print(f"{mel_pad_multiple=}")
    test_loader = NaiveDataLoader(data_path=TEST_H5, batch_size=1, padding_value=0, rng_seed=21, diffusion_k=diffusion_k, stats_path=TRAIN_STATS_PATH, diffusion_type=diffusion_type, mel_pad_multiple=mel_pad_multiple)

    batch = test_loader.next_batch(device) # maybe create dummy .h5 of just a single example
    
    use_num_iter = False
    if diffusion_type == "DiffSingerDiffusion":
        beta_min = denoiser_config['training']['diffusion']['beta_min']#1e-4
        beta_max = denoiser_config['training']['diffusion']['beta_max']#0.06
        diffusion = DiffSingerDiffusion(
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
            M=80,
            )
    elif diffusion_type == "SimpleFlow":
        use_num_iter = True
        print(f" --- using {args.num_iter=} sampling iterations")
        diffusion = SimpleFlow(device=device)
        mel = diffusion.sample(
            model=denoiser, 
            txt_tokens=batch['txt_tokens'], 
            mel2ph=batch['mel2ph'], 
            f0=batch['f0'], 
            uv=batch['uv'], 
            ph_padding_mask=batch['ph_padding_mask'], 
            mel_padding_mask=batch['mel_padding_mask'],
            M=80,
            num_iter=args.num_iter,
            cfg_scale=args.cfg_scale
            )
        
    
    print(f"[0] {mel.shape=}")
    
    # normalize mel-spectrograms to [-1, 1]
    mel_min = stats['mel_min'][None, :, None] # (1, M, 1)
    mel_max = stats['mel_max'][None, :, None] # (1, M, 1)

    mel = mel.cpu().numpy()

    mel = ((mel + 1)/2) * (mel_max - mel_min) + mel_min # undo the normalization done inside the dataloader, which was: mel = 2 * (mel - mel_min) / (mel_max - mel_min) - 1

    f0 = batch['f0'].cpu().numpy() * stats['f0_std'] + stats['f0_mean'] # undo the standardization to mean 0 std 1 done in the dataloader, which was: f0 = (f0 - stats['f0_mean']) / stats['f0_std'] # scalar mean and std should broadcast properly

    # convert stored mel-scale f0 back to Hz for the NSF vocoder
    f0_hz = 700.0 * (np.exp(f0 / 1127.0) - 1.0)

    mel_tensor = torch.from_numpy(mel).float().to(device)    # (1, n_mels, T)
    f0_tensor  = torch.from_numpy(f0_hz).float().to(device)  # (1, T)

    print(f"{mel_tensor.shape=}")
    print(f"{f0_tensor.shape=}")
    #mel_tensor = torch.clamp(mel_tensor, min=-6, max=1.5)

    with torch.no_grad():
        print(f"    {mel.min()=}, {mel.max()=}, {mel.mean()=}") 
        # mel values should be in [-6, 1.5] because in config.yaml they define `mel_vmin: -6` and `mel_vmax: 1.5`
        audio = vocoder(mel_tensor, f0_tensor).view(-1)

    audio = audio.cpu().numpy()
    output_name = f"OUTPUT-{denoiser_checkpoint_path.stem}-{model_type}"
    if args.ema:
        output_name += f"-EMA"
    if use_num_iter:
        output_name += f"-N={args.num_iter}"
    if args.cfg_scale is not None:
        output_name += f"-CFG={args.cfg_scale}"
    output_name += ".wav"
    output_path = output_dir / output_name
    sf.write(output_path, audio, samplerate=vocoder_config["audio_sample_rate"])
    print(f"wrote {output_path}")