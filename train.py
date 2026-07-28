import argparse
import toml
import json
import time
import os
from datetime import datetime
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from pathlib import Path
from model import MusicScoreEncoder, AuxiliaryDecoder, WaveNet, EncoderDecoder, WaveNetDenoiser, DiT
from exponential_moving_average import ExponentialMovingAverage
from data.dataloader import NaiveDataLoader
from loss_functions import get_loss_function
from muon import Muon
from functools import partial
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

class SimpleFlow:
    def __init__(self, device):
        self.device = device

    def get_interpolant(self, mel, epsilon, t):
        """
        `mel` has shape (B, M, T) and is the ground-truth (normalized) mel-spectrogram (i.e., pure data)
        `epsilon` has shape (B, M, T) and is isotropic gaussian noise
        `t` has shape (B,) and is a batch of times in [0,1]
        """
        if not torch.all( torch.logical_and(t >= 0, t <= 1)):
            raise ValueError(f"times should be in [0,1] but got {t.min()=}, {t.max()=}, {t=}")
        t = t[:, None, None] # (B, 1, 1)
        interpolant = t * mel + (1-t) * epsilon # at t=0, we have pure noise, at t=1 pure data
        return interpolant # this is the "noisy" mel-spectrogram that we input into the denoiser

    def sample(self, model, txt_tokens, mel2ph, f0, uv, ph_padding_mask, mel_padding_mask, M=80, num_iter=100, cfg_scale=None):
        """
        `model` is the denoiser, e.g. a WaveNetDenoiser class object
        
        `txt_tokens` has shape (B, P)  --  where B is batch size and P is the number of phonemes in the sequence
            contains sequences of phoneme token ids (corresponding to the phonemes used in an audio file)
        `mel2ph` has shape (B, T)  --  where T is the number of mel frames (when constructing the mel-spectrogram, time was discretized into T mel frames)
            `mel2ph` on a given mel frame contains the `txt_token` index corresponding to the phoneme used on in the audio, it is important to note that
            this is not the token id but the index into `txt_tokens`, this will allow for the same token id to potentially receive different positional
            information if the same token is used multiple times in `txt_tokens`
        `f0` has shape (B, T)
            contains the fundamental frequency during each mel frame (interpolation was used to smooth across unvoiced segments, see data preprocessing/binarization code)
        `uv` has shape (B, T) -- unused
            boolean mask which is True when the audio was unvoiced, corresponds to where the preinterpolated f0 was zero. False otherwise.
        `ph_padding_mask` has shape (B, P) and is True for padding values, False otherwise
        `mel_padding_mask` has shape (B, T) and is True if the mel frame index (for a given batch index) corresponds to a padding value, False otherwise
        `M` is an integer which represents in the number of mel bins of the mel-spectrogram that we want to generate

        `num_iter` is the number of forward euler steps we do
        """
        if num_iter < 1 or not isinstance(num_iter, int):
            raise ValueError(f"`num_iter` should be a positive integer, but got {num_iter=}, {type(num_iter)=}")
        with torch.no_grad():
            model.eval()
            B, T = mel_padding_mask.shape
            M_t = torch.randn((B, M, T), device=self.device) # inital noisy mel-spec at t=0
            t = torch.zeros(B, dtype=torch.float32, device=self.device) # shape (B,) -- batch of times from t=0 (pure noise)
            step_size = 1 / num_iter
            for i in range(num_iter):

                model_output_conditioned = model(
                    txt_tokens=txt_tokens, 
                    mel2ph=mel2ph,
                    f0=f0,
                    uv=uv, 
                    ph_padding_mask=ph_padding_mask, 
                    mel_padding_mask=mel_padding_mask,
                    mel=M_t, 
                    t=t
                    )

                if cfg_scale is None:
                    v = model_output_conditioned
                else:
                    model_output_unconditioned = model(
                        txt_tokens=txt_tokens, 
                        mel2ph=mel2ph,
                        f0=f0,
                        uv=uv, 
                        ph_padding_mask=ph_padding_mask, 
                        mel_padding_mask=mel_padding_mask,
                        mel=M_t, 
                        t=t,
                        null_mask=torch.tensor([1], dtype=torch.bool)
                        )
                    v = (1-cfg_scale) * model_output_unconditioned + cfg_scale * model_output_conditioned

                    
                M_t = M_t + step_size * model_output # forward euler update, model_output is the vector field F of an ODE dM_t/dt = F(t, M_t) with initial condition M_0 ~ N(0, I)
                t += step_size
        return M_t # generated mel-spectrogram


class DiffSingerDiffusion:
    def __init__(self, beta_min: float=1e-4, beta_max: float=0.06, T: int=100, device=None):
        """
        See Implementation Details in Section 4.1 https://arxiv.org/abs/2105.02446 which motivates the default values
        Note: with defaults we have (1 - diffusion.alpha_bar(100))**(0.5) = 0.9764 for the full noise weight in the interpolant
        """
        self.beta_min = beta_min # beta_1 -- t=1 should be (approximately) pure data
        self.beta_max = beta_max # beta_T -- t=T shouldbe pure noise (not sure why they use 0.06, it makes alpha_bar close enough I guess?)
        self.T = T # the maximum diffusion time step
        self.device = device
        self.precompute_alpha_bars() # precompute all of the alpha_bar values, see docstrings in `alpha_bar` and `precompute_alpha_bars` methods

    def beta(self, t):
        """
        `t` has shape (B,) where B is the batch size
        Compute beta at timestep t. We compute this for a batch of timesteps t.
        
        "We set T to 100 and β to constants increasing linearly from β_1 = 10^{−4} to β_T = 0.06" -- Section 4.1 of https://arxiv.org/abs/2105.02446

        Each timestep must be in {1, ..., T} and we will have beta increase linearly from t=1 to t=T
        """
        if torch.any(t < 1) or torch.any(t > self.T):
            raise ValueError(f"t must be in [1, {self.T}], got min={t.min().item()}, max={t.max().item()}")
        slope = (self.beta_max - self.beta_min) / (self.T - 1) # scalar
        beta = slope * (t - 1) + self.beta_min # shape (B,) --  have line pass through the point (t=1, beta=beta_min)
        return beta
    
    def alpha(self, t):
        if torch.any(t < 1) or torch.any(t > self.T):
            raise ValueError(f"t must be in [1, {self.T}], got min={t.min().item()}, max={t.max().item()}")
        return 1 - self.beta(t)

    def alpha_bar(self, t):
        """
        `t` has shape (B,) where B is the batch size
        alpha_bar(t) := alpha(1) * alpha(2) * ... * alpha(t)

        I'm just going to precompute all possible values of alpha_bar so we don't recompute the products all the time. 
        Not sure if this even matters much in terms of computational cost, but surely it doesn't hurt.
        """
        if torch.any(t < 1) or torch.any(t > self.T):
            raise ValueError(f"t must be in [1, {self.T}], got min={t.min().item()}, max={t.max().item()}")
        return self.alpha_bars[t-1] # shape (B,) -- we subtract 1 from t because index 0 in self.alpha_bars corresponds to timestep 1, etc.

    def precompute_alpha_bars(self):
        """actually precomputing all of the alpha_bar values for all possible timesteps, to be indexed by `alpha_bar` method"""
        timesteps = torch.arange(1, self.T+1) # shape (T,) -- timesteps {1, ..., T}
        alphas = self.alpha(timesteps) # shape (T,)
        self.alpha_bars = torch.cumprod(alphas, dim=0) # shape (T,)
        self.alpha_bars = self.alpha_bars.to(self.device)

    def get_interpolant(self, mel, epsilon, t):
        """
        `mel` has shape (B, M, T) and is the ground-truth (normalized) mel-spectrogram (i.e., pure data)
        `epsilon` has shape (B, M, T) and is isotropic gaussian noise
        `t` has shape (B,) and is a batch of timesteps
        This method computes the interpolant between pure data and pure noise, which gets fed as input into the denoiser,
        see Algorithm 1 in the DiffSinger paper https://arxiv.org/abs/2105.02446
        """
        alpha_bar = self.alpha_bar(t)[:, None, None] # shape (B, 1, 1) so it broadcasts on the next line
        interpolant = torch.sqrt(alpha_bar) * mel + torch.sqrt(1 - alpha_bar) * epsilon # shape (B, M, T)
        return interpolant # this is the "noisy" mel-spectrogram that we input into the denoiser
    
    def sigma(self, t):
        """
        `t` has shape (B,) where B is the batch size

        Right before Section 3 of the DiffSinger paper, they define 
            \tilde{beta}_t := [(1 - alpha_bar(t-1))/(1 - alpha_bar(t))] * beta(t) 
        and they set sigma_t^2 equal to it. This function computes sigma_t, which is the square root of \tilde{beta}_t. 
        """
        beta_tilde_t = ((1 - self.alpha_bar(t-1)) / (1 - self.alpha_bar(t))) * self.beta(t)
        return torch.sqrt(beta_tilde_t) # (B,)
    
    def sample(self, model, txt_tokens, mel2ph, f0, uv, ph_padding_mask, mel_padding_mask, M=80):
        """
        `model` is the denoiser, e.g. a WaveNetDenoiser class object
        
        `txt_tokens` has shape (B, P)  --  where B is batch size and P is the number of phonemes in the sequence
            contains sequences of phoneme token ids (corresponding to the phonemes used in an audio file)
        `mel2ph` has shape (B, T)  --  where T is the number of mel frames (when constructing the mel-spectrogram, time was discretized into T mel frames)
            `mel2ph` on a given mel frame contains the `txt_token` index corresponding to the phoneme used on in the audio, it is important to note that
            this is not the token id but the index into `txt_tokens`, this will allow for the same token id to potentially receive different positional
            information if the same token is used multiple times in `txt_tokens`
        `f0` has shape (B, T)
            contains the fundamental frequency during each mel frame (interpolation was used to smooth across unvoiced segments, see data preprocessing/binarization code)
        `uv` has shape (B, T) -- unused
            boolean mask which is True when the audio was unvoiced, corresponds to where the preinterpolated f0 was zero. False otherwise.
        `ph_padding_mask` has shape (B, P) and is True for padding values, False otherwise
        `mel_padding_mask` has shape (B, T) and is True if the mel frame index (for a given batch index) corresponds to a padding value, False otherwise
        `M` is an integer which represents in the number of mel bins of the mel-spectrogram that we want to generate

        This function does naive sampling used in DiffSinger (i.e. NOT the shallow diffusion approach that they also use in their paper)

        # TODO be careful with padding masks, just try batch size of 1 for now
        """
        with torch.no_grad():
            model.eval()
            B, T = mel_padding_mask.shape # do not confuse T (number of mel frames) with self.T (total diffusion time steps)
            M_t = torch.randn((B, M, T), device=self.device)
            for i in range(self.T, 1, -1): # TODO
                t = torch.ones(B, dtype=torch.long, device=self.device) * i # shape (B,)

                model_output = model(
                    txt_tokens=txt_tokens, 
                    mel2ph=mel2ph,
                    f0=f0,
                    uv=uv, 
                    ph_padding_mask=ph_padding_mask, 
                    mel_padding_mask=mel_padding_mask,
                    mel=M_t, 
                    t=t
                    )

                z = torch.randn((B, M, T), device=self.device)
                alpha_t = self.alpha(t)
                M_t = (1/torch.sqrt(alpha_t)) * (M_t - ((1 - alpha_t)/torch.sqrt(1 - self.alpha_bar(t)))*model_output ) + self.sigma(t) * z
        return M_t # generated mel-spectrogram
    

def print0(s="", **kwargs):
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        print(s, **kwargs)

def write0(s, log_file):
    """
    `s` is the string to write to the log file
    `log_file` is the path to the log.txt file
    """
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        with open(log_file, 'a') as f:
            f.write(s)

def create_log_dir(parent_dir, config_name):
    rank = int(os.environ.get('RANK', 0))
    log_dir = None # initialize as None to avoid errors on nonzero ranks
    if rank == 0:
        timestamp = datetime.now().strftime("%m-%d-%Y-%Hh%Mm%Ss")
        log_dir = os.path.join(parent_dir, config_name, f"{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_dir = os.path.join(log_dir, "checkpoints") # store checkpoints here
        os.makedirs(checkpoint_dir, exist_ok=True)
        sample_dir = os.path.join(log_dir, "samples") # store image samples made during training here
        os.makedirs(sample_dir, exist_ok=True)

    # if using DDP, define the directory on all ranks to allow any rank to write to the log file if desired
    if dist.is_available() and dist.is_initialized():
        obj = [log_dir]
        dist.broadcast_object_list(obj, src=0)
        log_dir = obj[0]
    return log_dir

def freeze(model):
    """freeze all weights of a model"""
    for param in model.parameters():
        param.requires_grad = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help=".toml file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = toml.load(f)
    
    effective_batch_size = config["training"]["effective_batch_size"]
    batch_size = config["training"]["batch_size"]
    grad_accum_steps = config["training"]["grad_accum_steps"]
    training_steps = config["training"]["training_steps"]
    clip_grad_norm_threshold = config['training'].get('clip_grad_norm_threshold', 1.0)

    val_loss_interval = config["training"]["val_loss_interval"]
    checkpoint_interval = config["training"]["checkpoint_interval"]

    save_dir = config["training"]["save_dir"]
    data_dir = config["training"]["data_dir"]
    train_data_path = os.path.join(data_dir, "train.h5")
    val_data_path = os.path.join(data_dir, "val.h5")
    train_data_stats_path = os.path.join(data_dir, "train_stats.npz")
    phoneme_vocab_path = os.path.join(data_dir, "vocab.json") 
    vocab =  json.load(open(phoneme_vocab_path, encoding="utf-8")) # load phoneme vocabulary to sanity check padding token matches config
    if config["model"]["pad_token_id"] != vocab["<PAD>"]:
        raise ValueError(f"Padding token id in {phoneme_vocab_path} does not match the one in {args.config}. {config['model']['pad_token_id']=}, {vocab['<PAD>']=}")
    if config["model"]["vocab_size"] != len(vocab):
        raise ValueError(f"Dataset vocab size ({len(vocab)=}) does not match model vocab size ({config['model']['vocab_size']=}), used vocab from: {phoneme_vocab_path}")

    # Learning Rate Schedule (Cosine Decay -- warmup + constant if you let min_lrm = 1 and cosine_decay_steps=0)
    warmup_steps = config["training"]["warmup_steps"]
    cosine_decay_steps = config["training"].get("cosine_decay_steps", training_steps - warmup_steps)
    min_lrm = config["training"].get("min_lrm", 1/10)
    if min_lrm > 1:
        raise ValueError(f"Expected to have `min_lrm` <= 1, got: {min_lrm=}")
    def cosine_lrm(step: int, warmup_steps: int, cosine_decay_steps: int, min_lrm: float):
        """
        learning rate multiplier schedule to simplify using different learning rates for different param groups with muon,
        also allows backwards compatibility with how lr was set previously -- I effectively normalized everything by `max_lr`
        EDIT: nevermind, I'm replacing min_lr with `min_lrm` and `max_lr` is replaced by `adamw_base_lr`, so it won't be 
        backwards compatible and I'm not changing the rest of this docstring, I think the main idea is pretty clear

        `step` is the iteration of training we're on, which is assumed to start at 0
        `warmup_steps` is the number of steps to do a linear warmup from `1/(warmup_steps+1)` to `warmup_steps/(warmup_steps+1)`
        `min_lrm` is the minimum learning rate multiplier, set to be `min_lr/max_lr` e.g. 1/10 if we want to decrease lr by a factor of 10 after cosine decay
        `cosine_decay_steps` is the number of steps that the cosine decay portion of the schedule lasts for
        
        The cosine decay ends after `warmup_steps + cosine_decay_steps` steps where the first `warmup_steps` steps are a linear 
        increase and the next `cosine_decay_steps` use a cosine decay and then any steps afterwards use `min_lrm`

        For a constant lr schedule, let min_lrm=1 and cosine_decay_steps=0
        """
        if step < warmup_steps: # linear warmup for warmup_steps steps
            return (step + 1) / (warmup_steps + 1) # linearly increase to 1 (never quite reach 1 during warmup stage)
        elif cosine_decay_steps == 0:
            return 1
        elif step >= warmup_steps and step < warmup_steps + cosine_decay_steps: # in between, use cosine decay down from 1 to min_lrm (start at 1, but never quite reach min_lrm during cosine decay stage)
            decay_ratio = (step - warmup_steps) / cosine_decay_steps # increases from 0 to 1
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # `coeff` goes from 1 down to 0
            return coeff + (1 - coeff) * min_lrm # convex combination between 1 and min_lrm where `coeff` is in [0,1]
        elif step >= warmup_steps + cosine_decay_steps: # after the cosine decay is done, return the minimum learning rate multiplier
            return min_lrm

    get_lrm = partial(cosine_lrm, warmup_steps=warmup_steps, cosine_decay_steps=cosine_decay_steps, min_lrm=min_lrm) 

    model_config = config["model"]
    diffusion_type = config['training']['diffusion'].get('diffusion_type')
    diffusion_k = config['training']['diffusion'].get('k')
    if model_config["model_type"] == "EncoderDecoder": # Phase 1 of training
        model = EncoderDecoder(model_config)
    elif model_config["model_type"] == "WaveNetDenoiser": # Phase 2 of training
        model = WaveNetDenoiser(model_config)
        encoder_checkpoint_path = config['model'].get('encoder_checkpoint')
        if encoder_checkpoint_path is not None:
            encoder_checkpoint = torch.load(encoder_checkpoint_path, map_location="cpu") # actually checkpoint for EncoderDecoder model...
            encoder_decoder = EncoderDecoder(model_config)
            encoder_decoder.load_state_dict(encoder_checkpoint["model"])
            model.encoder = encoder_decoder.encoder
            print0(f"ENCODER LOADED WITH CHECKPOINT {encoder_checkpoint_path}\n")
        if model_config.get('freeze_encoder', True):
            freeze(model.encoder)
            model.encoder.eval() # if the encoder is frozen, we don't want dropout active inside it
            print0("ENCODER PARAMETERS ARE FROZEN")
    elif model_config["model_type"] == "DiT":
        model = DiT(model_config)     
    else:
        raise ValueError(f"{model_config['model_type']=}")

    if config["training"].get("resume_training", False):
        checkpoint = torch.load(config["training"]["checkpoint_path"], map_location="cpu")
        # If we resume training, we change the rng seed in the config as a lazy way making sure we don't get the same random times and such
        # I didn't bother storing rng state of each rank because I might resume with a different number of gpus anyway and it is simpler this way
        # The checkpoint stores a set of all rng seeds used across all training runs to be sure we never repeat any of them (the gpus I'm using can have a lot of issues)
        if config["training"]["rng_seed"] in checkpoint["rng_seeds"]:
            raise ValueError(f"Change the rng seed in the config before you resume training! {checkpoint['rng_seeds']=}, {config['training']['rng_seed']=}")

        model.load_state_dict(checkpoint["model"])
        print0(f"MODEL LOADED WITH CHECKPOINT {config['training']['checkpoint_path']}\n")
    prior_rng_seeds = checkpoint["rng_seeds"] if config["training"].get("resume_training", False) else set() # to be stored in checkpoint to be sure we don't accidentally resume training with a previously used rng seed
    prior_rng_seeds.add(config["training"]["rng_seed"])

    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = dist.get_world_size()

        assert rank == dist.get_rank(), f"{rank=}, {dist.get_rank()=}"

        device = f'cuda:{local_rank}'
        device_type = 'cuda'
        torch.cuda.set_device(device)
        print(f"{rank=}, {local_rank=}, {world_size=}, {device=}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device_type = device
        print(f"{rank=}, {local_rank=}, {world_size=}, {device=}")

    # sanity check inputs
    if world_size * batch_size * grad_accum_steps != effective_batch_size:
        raise ValueError(f"{effective_batch_size=}, {world_size=}, {batch_size=}, {grad_accum_steps=}, {world_size*batch_size*grad_accum_steps=}")

    # Initialize log file
    log_dir = create_log_dir(save_dir, config_name=Path(args.config).stem) # we wait until after DDP to define this and it gets shared across ranks
    log_file = f"{log_dir}/log.txt"
    log_csv = f"{log_dir}/log.csv"
    if rank == 0:
        with open(log_file, 'w') as f:
            f.write("") # initialize log file
        with open(log_csv, 'w') as f:
            f.write(",".join(["step", "train_loss", "val_loss", "lrm", "grad_norm"]) + "\n") # initialize header for csv

        with open(os.path.join(log_dir, "config.toml"), "w") as f:
            toml.dump(config, f) # save copy of config in log directory

    # Write basic info at start of log file (config, GPU info, etc.)
    write0("Config:\n", log_file=log_file)
    for key, value in config.items():
        if isinstance(value, dict):
            write0(f"  {key}:\n", log_file=log_file)
            for subkey, subvalue in value.items():
                write0(f"    {subkey}: {subvalue}\n", log_file=log_file)
        else:
            write0(f"  {key}: {value}\n", log_file=log_file)
    write0(f"Using {world_size} GPU(s)\n", log_file=log_file)
    write0(f"GPU Type: {torch.cuda.get_device_name()}\n", log_file=log_file)

    model = model.to(device)
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    orig_model = model.module if ddp else model
    
    if config['training'].get('use_torch_compile', False):
        dynamic = config['training'].get('torch_compile_dynamic', False)
        if type(dynamic) is not bool:
            if dynamic == "none":
                dynamic = None
            else:
                raise ValueError(f"`dynamic` should be either `true`, `false`, or `'none'` in the config, got {dynamic=}")
        model = torch.compile(model, dynamic=dynamic)
        write0(f"Using torch.compile with {dynamic=}\n", log_file=log_file)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write0(f"Model Parameters: {num_params:,}\nTrainable Model Parameters: {num_trainable_params:,}\n", log_file=log_file)

    mel_pad_multiple = config['training'].get('mel_pad_multiple', 1)
    use_cfg_null_embedding = config['model'].get('use_cfg_null_embedding', False)
    null_embedding_probability = config['training'].get('null_embedding_probability')
    train_loader = NaiveDataLoader(data_path=train_data_path, batch_size=batch_size, padding_value=config["model"]["pad_token_id"], rng_seed=config["training"]["rng_seed"], diffusion_k=diffusion_k, stats_path=train_data_stats_path, diffusion_type=diffusion_type, mel_pad_multiple=mel_pad_multiple, use_cfg_null_embedding=use_cfg_null_embedding, null_embedding_probability=null_embedding_probability)

    # Setup optimizer(s)
    if config['training'].get('use_muon', False):
        if not hasattr(orig_model, "setup_optimizers"):
            raise ValueError(f"If using Muon, it's expected that a model with a `setup_optimizers` method is being used, {type(orig_model)=}")
        write0(f"Using Muon (and AdamW) optimizers\n", log_file=log_file)
        optimizers = orig_model.setup_optimizers(
            adamw_base_lr=config['training']['AdamW_base_lr'],
            adamw_weight_decay=config["training"]["AdamW_weight_decay"], 
            adamw_betas=config["training"]["AdamW_betas"], 
            adamw_epsilon=config["training"]["AdamW_epsilon"], 
            muon_base_lr=config['training']['Muon_base_lr'], 
            muon_weight_decay=config['training']['Muon_weight_decay'], 
            muon_momentum=config['training']['Muon_momentum'],
            null_embedding_base_lr=config['training'].get('null_embedding_base_lr')
        )
    else:
        write0(f"Only using AdamW optimizer\n", log_file=log_file)
        adamw_optimizer = torch.optim.AdamW(
            model.parameters(),
            weight_decay=config["training"]["AdamW_weight_decay"],
            betas=config["training"]["AdamW_betas"],
            eps=config["training"]["AdamW_epsilon"],
            fused=True
            )

        optimizers = {'AdamW' : adamw_optimizer}
        for name, opt in optimizers.items():
            for param_group in opt.param_groups:
                param_group["base_lr"] = config['training']['AdamW_base_lr'] # set base_lr which will be multiplied by our learning rate multiplier `lrm` during training

    ema = ExponentialMovingAverage(params=model.parameters(), decay=config["training"]["ema_decay"])

    loss_function = get_loss_function(config['training']['loss'])

    # Define dtype for optional mixed precision training, if fp16 we will use GradScaler -- partially borrowed scaler logic from https://github.com/karpathy/nanochat/blob/master/scripts/base_train.py
    compute_dtype = config['training'].get('compute_dtype', 'fp32')
    amp_dtypes = ['fp16', 'bf16']
    valid_dtypes = ['fp32'] + amp_dtypes
    amp_dtype = None
    if compute_dtype == "fp16":
        amp_dtype = torch.float16
    elif compute_dtype == "bf16":
        amp_dtype = torch.bfloat16
    amp_enabled = True if compute_dtype in amp_dtypes else False

    if compute_dtype not in valid_dtypes:
        raise ValueError(f"Expected one of {valid_dtypes=}, but got {compute_dtype=}")
    write0(f"Training with dtype {compute_dtype} ({amp_dtype=}, {amp_enabled=})\n", log_file=log_file)
    # GradScaler for fp16 training (bf16/fp32 don't need it -- bf16 has the same exponent range as fp32)
    scaler = torch.amp.GradScaler() if compute_dtype == 'fp16' else None
    if scaler is not None:
        print0(f"GradScaler is enabled for fp16 training ({amp_dtype=}, {amp_enabled=})")
        write0(f"GradScaler is enabled for fp16 training ({amp_dtype=}, {amp_enabled=})\n", log_file=log_file)

    if config["training"].get("resume_training", False):
        for name, opt in optimizers.items():
            opt.load_state_dict(checkpoint["optimizer"][name])
        ema.load_state_dict(checkpoint["ema"], device=device)

        if rank == 0:
            dataloader_state = checkpoint["dataloader"]
        else:
            dataloader_state = [None] * world_size # placeholder; rank 0 will broadcast
        train_loader.load_state_dict(dataloader_state) # Must be called on ALL ranks (does broadcast internally)

        initial_step = checkpoint["step"]

        if scaler is not None and checkpoint.get('scaler') is not None:
            scaler.load_state_dict(checkpoint['scaler'])
            write0(f"Restored GradScaler state dict from checkpoint\n", log_file=log_file)

        write0(f"RESUMING TRAINING WITH CHECKPOINT {config['training']['checkpoint_path']} AT STEP {initial_step}\n", log_file=log_file)
    else:
        initial_step = 0 # if not resuming training, have training loop start at step 0

    write0(f"USING DIFFUSION TYPE: {diffusion_type}\n", log_file=log_file)
    torch.manual_seed(config["training"]["rng_seed"] + rank) # (I guess this affects the categorical sampling, random times are in the dataloader)
    if config['model']['model_type'] in ["WaveNetDenoiser", "DiT"]: # just hardcoding some stuff like this for now...
        assert diffusion_type is not None
    if diffusion_type == "DiffSingerDiffusion":
        diffusion = DiffSingerDiffusion(
            beta_min=config['training']['diffusion']['beta_min'], 
            beta_max=config['training']['diffusion']['beta_max'], 
            T=config['training']['diffusion']['T'],
            device=device
            )
    elif diffusion_type == "SimpleFlow":
        diffusion = SimpleFlow(device=device)
    elif diffusion_type is not None:
        raise ValueError(f"Got {diffusion_type=}")
        
    # TRAINING LOOP
    for step in range(initial_step, training_steps):
        computed_val_loss_this_iteration = False

        # SAVE CHECKPOINTS
        if (step % checkpoint_interval == 0 or step == training_steps - 1):
            dataloader_state_dict = train_loader.get_state_dict() # need to call this on all ranks, then save all-gathered result on rank 0
            optimizer_state_dicts = {}
            for name, opt in optimizers.items():
                optimizer_state_dicts[name] = opt.state_dict() # need to call this on all ranks for the Muon class
            if rank == 0:
                torch.cuda.synchronize()
                t0 = time.time()
                
                checkpoint = {
                    'step' : step,
                    'model' : model.module.state_dict() if ddp else model.state_dict(),
                    'optimizer' : optimizer_state_dicts,
                    'dataloader' : dataloader_state_dict,
                    'ema' : {'ema_params' : ema.ema_params, 'decay' : ema.decay},
                    'rng_seeds' : prior_rng_seeds,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                }
                checkpoint_path = os.path.join(log_dir, f'checkpoints/checkpoint_step{step}.pt')
                torch.save(checkpoint, checkpoint_path)
                torch.cuda.synchronize()
                t1 = time.time()
                write0(f" --- Checkpoint saved to {checkpoint_path} in {t1-t0:.4f}s\n", log_file=log_file)

        # VALIDATION LOSS
        if rank == 0 and (step % val_loss_interval == 0 or step == training_steps - 1): # TODO
            with torch.no_grad():
                
                t0 = time.time()
                model.eval()
                rng_state = torch.get_rng_state() # val might change rng state on rank 0, so save and restore it just in case, probably not very important
                val_loader = NaiveDataLoader(data_path=val_data_path, batch_size=min(batch_size, config['training'].get('val_dataset_length', 24)), padding_value=config["model"]["pad_token_id"], diffusion_k=diffusion_k, stats_path=train_data_stats_path, diffusion_type=diffusion_type, mel_pad_multiple=mel_pad_multiple, use_cfg_null_embedding=use_cfg_null_embedding, null_embedding_probability=null_embedding_probability) # Should be reinitialized with same rng seed every time. Also notice how I intentionally use the default rng seed for val loader so that it never changes even when I resume training with a new rng seed in my config
                val_loader.reset() # should be unnecessary

                ema.store(model.parameters()) # store copy of the actual model weights
                ema.copy_to(model.parameters()) # copy EMA weights into the model
                val_losses = []
                for val_step in range(config["training"].get("val_steps", 1)):
                    val_batch = val_loader.next_batch(device=device)

                    # whatever, I'm just hardcoding this for now...
                    val_ground_truth_mel = val_batch['mel']
                    with torch.amp.autocast(device_type, dtype=amp_dtype, enabled=amp_enabled): # just put forward pass and loss computation inside amp
                        if config['model']['model_type'] in ["WaveNetDenoiser", "DiT"]: # just hardcoding some stuff like this for now...

                            val_interpolant = diffusion.get_interpolant(mel=val_ground_truth_mel, epsilon=val_batch['epsilon'], t=val_batch['t'])
                            if type(diffusion) == DiffSingerDiffusion:
                                val_target = val_batch['epsilon']
                            elif type(diffusion) == SimpleFlow:
                                val_target = val_ground_truth_mel - val_batch['epsilon']

                            val_model_output = model(
                                txt_tokens=val_batch['txt_tokens'], 
                                mel2ph=val_batch['mel2ph'],
                                f0=val_batch['f0'],
                                uv=val_batch['uv'], 
                                ph_padding_mask=val_batch['ph_padding_mask'], 
                                mel_padding_mask=val_batch['mel_padding_mask'],
                                mel=val_interpolant, 
                                t=val_batch['t'],
                                null_mask=val_batch.get('null_embedding_mask')
                                )
                            val_loss = loss_function(target=val_target, pred=val_model_output, mel_padding_mask=val_batch['mel_padding_mask'])
                        elif config['model']['model_type'] == "EncoderDecoder":
                            val_model_output = model(
                                txt_tokens=val_batch['txt_tokens'],
                                mel2ph=val_batch['mel2ph'],
                                f0=val_batch['f0'],
                                uv=val_batch['uv'],
                                ph_padding_mask=val_batch['ph_padding_mask'],
                                mel_padding_mask=val_batch['mel_padding_mask']
                                )
                            val_loss = loss_function(target=val_ground_truth_mel, pred=val_model_output, mel_padding_mask=val_batch['mel_padding_mask'])
                        else:
                            raise ValueError(f"{config['model']['model_type']=}")

                    val_losses.append(val_loss)
                val_loss = sum(val_losses) / len(val_losses)
                ema.restore(model.parameters()) # copy stored model weights back into the model
                torch.set_rng_state(rng_state) # restore rng state on rank 0
                model.train()
                if model_config.get("freeze_encoder", True) and model_config["model_type"] == "WaveNetDenoiser":
                    model.module.encoder.eval() if ddp else model.encoder.eval() # keep the frozen encoder in eval to ignore dropout
                t1 = time.time()
                computed_val_loss_this_iteration = True
                write0(f"val loss: {val_loss}{' '*(8 - len(str(step)))}{(t1-t0)*1000:.0f}ms\n", log_file=log_file)

        # TRAINING
        torch.cuda.synchronize()
        t0 = time.time()
        train_loss = 0.0 # for logging
        for micro_step in range(grad_accum_steps):

            batch = train_loader.next_batch(device=device)

            if ddp: # only sync gradients on the last micro step
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            # Effective batch size is per_gpu_batch_size * num_gpus * grad_accum_steps. Before syncing gradients, the local gradient has
            # per_gpu_batch_size in the denominator (since the loss is averaged over the local batch) and then when we sync gradients with
            # the .backward() call (with require_backward_grad_sync True), they are averaged over all ranks, so the resulting gradient has
            # per_gpu_batch_size * num_gpus in the denominator. Dividing the loss by grad_accum_steps gives us the correct final denominator.
            ground_truth_mel = batch['mel'] # TODO need to normalize? probably do in collator and rename collator to pad and norm or something
            

            with torch.amp.autocast(device_type, dtype=amp_dtype, enabled=amp_enabled): # just put forward pass and loss computation inside amp

                if config['model']['model_type'] in ["WaveNetDenoiser", "DiT"]: # just hardcoding some stuff like this for now...

                    interpolant = diffusion.get_interpolant(mel=ground_truth_mel, epsilon=batch['epsilon'], t=batch['t'])
                    if type(diffusion) == DiffSingerDiffusion:
                        target = batch['epsilon']
                    elif type(diffusion) == SimpleFlow:
                        target = ground_truth_mel - batch['epsilon']

                    model_output = model(
                        txt_tokens=batch['txt_tokens'], 
                        mel2ph=batch['mel2ph'],
                        f0=batch['f0'],
                        uv=batch['uv'], 
                        ph_padding_mask=batch['ph_padding_mask'], 
                        mel_padding_mask=batch['mel_padding_mask'],
                        mel=interpolant, 
                        t=batch['t'],
                        null_mask=batch.get('null_embedding_mask')
                        )
                    loss = loss_function(target=target, pred=model_output, mel_padding_mask=batch['mel_padding_mask']) / grad_accum_steps
                elif config['model']['model_type'] == "EncoderDecoder":
                    model_output = model(
                        txt_tokens=batch['txt_tokens'],
                        mel2ph=batch['mel2ph'],
                        f0=batch['f0'],
                        uv=batch['uv'],
                        ph_padding_mask=batch['ph_padding_mask'],
                        mel_padding_mask=batch['mel_padding_mask']
                        )
                    loss = loss_function(target=ground_truth_mel, pred=model_output, mel_padding_mask=batch['mel_padding_mask']) / grad_accum_steps
                else:
                    raise ValueError(f"{config['model']['model_type']=}")

            train_loss += loss.detach() # for logging

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if dist.is_initialized():
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # for logging
        
        lrm = get_lrm(step)
        for name, opt in optimizers.items():
            for param_group in opt.param_groups:
                param_group['lr'] = lrm * param_group['base_lr'] # base_lr was set for each param group when we defined the optimizers

        if scaler is not None:
            for name, opt in optimizers.items():
                scaler.unscale_(opt) # important that this is done before clip_grad_norm
            # In distributed training, all ranks must agree on whether to skip the step.
            # Each rank may independently encounter inf/nan gradients, so we all-reduce
            # the found_inf flag (MAX = if any rank found inf, all ranks skip).
            if dist.is_available() and dist.is_initialized():
                for name, opt in optimizers.items():
                    for v in scaler._found_inf_per_device(opt).values():
                        dist.all_reduce(v, op=dist.ReduceOp.MAX)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_threshold) # we need to do this after unscaling, otherwise it uses the scaled up values to determine if the clipping threshold has been reached
            for name, opt in optimizers.items():
                scaler.step(opt)
            scaler.update()
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_threshold)
            for name, opt in optimizers.items():
                opt.step()

        ema.update(model.parameters())
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t1 = time.time()

        write0(f"Step {step}:{' '*(8 - len(str(step)))}{(t1-t0)*1000:.0f}ms    train loss: {train_loss.item():.6f}    lrm: {lrm:.10f}    grad norm: {norm.item():.6f} \n", log_file=log_file)
        csv_val_loss = None if not computed_val_loss_this_iteration else val_loss.item()
        write0(",".join(map(str, [step, train_loss.item(), csv_val_loss, lrm, norm.item()])) + "\n", log_file=log_csv)
        # with open(log_file, 'a') as f:
        #     f.write(f"    {step=}    {rank=}   {batch['ph_padding_mask'].shape=}    {torch.sum(batch['ph_padding_mask'])=}    {batch['mel_padding_mask'].shape=}    {torch.sum(batch['mel_padding_mask'])=}\n")

    if ddp:
        dist.destroy_process_group()
