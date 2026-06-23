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
import torch._dynamo
torch._dynamo.config.cache_size_limit = 64

"""
TODO:
    - get stats to normalize mel-spectrograms (don't need to unnormalize outputs I think, vocoder should accept normalized inputs) -- modify binarize.py
        - save config when generating dataset and put in a nicer directory maybe
        - get stats (min, max, mean, median, std) for f0 as well since in Section 4.1 they claim to standardize f0
    # get loss functions defined for both phases of training and handle using them correctly (feeding correct inputs to models, both for train and val, etc.)
        # need to load models properly depending on which phase of training is being done, just hardcode for now
    - get vocoder set up to test inference
"""

class DiffusionProcess:
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

    # Learning Rate Schedule (Cosine Decay -- warmup + constant if you let min_lr = max_lr and cosine_decay_steps=0)
    warmup_steps = config["training"].get("warmup_steps") # should only be None if `use_step_decay` is True
    max_lr = config["training"].get("max_lr") # should only be None if `use_step_decay` is True
    min_lr = config["training"].get("min_lr", None if max_lr is None else max_lr/10)
    lr_decay_steps = config["training"].get("cosine_decay_steps", None if warmup_steps is None else training_steps - warmup_steps)
    use_step_decay = config["training"].get("use_step_decay", False)
    def get_lr(it):
        if not use_step_decay:
            # 1) linear warmup for warmup_steps steps
            if it < warmup_steps:
                return max_lr * (it + 1) / (warmup_steps + 1)
            # 2) if it > lr_decay_steps, return min learning rate
            if it > lr_decay_steps:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - warmup_steps) / (lr_decay_steps - warmup_steps)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (max_lr - min_lr)
        else:
            # hardcoding what DiffSinger uses as default for training their denoiser, just for replication purposes. I am basing this off of the lr plot from training DiffSinger with their code
            if it < 50000:
                return 1e-3
            elif it < 100000:
                return 5e-4
            elif it < 150000:
                return 2.5e-4
            else:
                return 1.25e-4

    model_config = config["model"]
    if model_config["model_type"] == "EncoderDecoder": # Phase 1 of training
        model = EncoderDecoder(model_config)
        diffusion_k = None
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
        diffusion_k = config['training']['diffusion']['k']
    elif model_config["model_type"] == "DiT":
        model = DiT(model_config)
        diffusion_k = config['training']['diffusion']['k']
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
            f.write(",".join(["step", "train_loss", "val_loss", "lr", "grad_norm"]) + "\n") # initialize header for csv

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

    train_loader = NaiveDataLoader(data_path=train_data_path, batch_size=batch_size, padding_value=config["model"]["pad_token_id"], rng_seed=config["training"]["rng_seed"], diffusion_k=diffusion_k, stats_path=train_data_stats_path)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=config["training"]["AdamW_weight_decay"],
        betas=config["training"]["AdamW_betas"],
        eps=config["training"]["AdamW_epsilon"],
        fused=True
        )
    
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
        optimizer.load_state_dict(checkpoint["optimizer"])
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

    torch.manual_seed(config["training"]["rng_seed"] + rank) # (I guess this affects the categorical sampling, random times are in the dataloader)
    if config['model']['model_type'] in ["WaveNetDenoiser", "DiT"]: # just hardcoding some stuff like this for now...
        diffusion = DiffusionProcess(
            beta_min=config['training']['diffusion']['beta_min'], 
            beta_max=config['training']['diffusion']['beta_max'], 
            T=config['training']['diffusion']['T'],
            device=device
            )
        
    # TRAINING LOOP
    for step in range(initial_step, training_steps):
        computed_val_loss_this_iteration = False

        # SAVE CHECKPOINTS
        if (step % checkpoint_interval == 0 or step == training_steps - 1):
            dataloader_state_dict = train_loader.get_state_dict() # need to call this on all ranks, then save all-gathered result on rank 0
            if rank == 0:
                torch.cuda.synchronize()
                t0 = time.time()
                checkpoint = {
                    'step' : step,
                    'model' : model.module.state_dict() if ddp else model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
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
                val_loader = NaiveDataLoader(data_path=val_data_path, batch_size=min(batch_size, config['training'].get('val_dataset_length', 24)), padding_value=config["model"]["pad_token_id"], diffusion_k=diffusion_k, stats_path=train_data_stats_path) # Should be reinitialized with same rng seed every time. Also notice how I intentionally use the default rng seed for val loader so that it never changes even when I resume training with a new rng seed in my config
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

                            val_model_output = model(
                                txt_tokens=val_batch['txt_tokens'], 
                                mel2ph=val_batch['mel2ph'],
                                f0=val_batch['f0'], # TODO need to standardize? probably do in collator?
                                uv=val_batch['uv'], 
                                ph_padding_mask=val_batch['ph_padding_mask'], 
                                mel_padding_mask=val_batch['mel_padding_mask'],
                                mel=val_interpolant, 
                                t=val_batch['t']
                                )
                            val_loss = loss_function(ground_truth_mel=val_batch['epsilon'], output_mel=val_model_output, mel_padding_mask=val_batch['mel_padding_mask'])
                        elif config['model']['model_type'] == "EncoderDecoder":
                            val_model_output = model(
                                txt_tokens=val_batch['txt_tokens'],
                                mel2ph=val_batch['mel2ph'],
                                f0=val_batch['f0'],
                                uv=val_batch['uv'],
                                ph_padding_mask=val_batch['ph_padding_mask'],
                                mel_padding_mask=val_batch['mel_padding_mask']
                                )
                            val_loss = loss_function(ground_truth_mel=val_ground_truth_mel, output_mel=val_model_output, mel_padding_mask=val_batch['mel_padding_mask'])
                        else:
                            raise ValueError(f"{config['model']['model_type']=}")

                    #val_loss = loss_function(ground_truth_mel=val_ground_truth_mel, output_mel=val_model_output, mel_padding_mask=val_batch['mel_padding_mask'])
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

                    model_output = model(
                        txt_tokens=batch['txt_tokens'], 
                        mel2ph=batch['mel2ph'],
                        f0=batch['f0'], # TODO need to standardize? probably do in collator?
                        uv=batch['uv'], 
                        ph_padding_mask=batch['ph_padding_mask'], 
                        mel_padding_mask=batch['mel_padding_mask'],
                        mel=interpolant, 
                        t=batch['t']
                        )
                    loss = loss_function(ground_truth_mel=batch['epsilon'], output_mel=model_output, mel_padding_mask=batch['mel_padding_mask']) / grad_accum_steps
                elif config['model']['model_type'] == "EncoderDecoder":
                    model_output = model(
                        txt_tokens=batch['txt_tokens'],
                        mel2ph=batch['mel2ph'],
                        f0=batch['f0'],
                        uv=batch['uv'],
                        ph_padding_mask=batch['ph_padding_mask'],
                        mel_padding_mask=batch['mel_padding_mask']
                        )
                    loss = loss_function(ground_truth_mel=ground_truth_mel, output_mel=model_output, mel_padding_mask=batch['mel_padding_mask']) / grad_accum_steps
                else:
                    raise ValueError(f"{config['model']['model_type']=}")

            #loss = loss_function(ground_truth_mel=ground_truth_mel, output_mel=model_output, mel_padding_mask=batch['mel_padding_mask']) / grad_accum_steps # uhh, surely this was a typo?? I need the target to be different during diffusion
            train_loss += loss.detach() # for logging

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if dist.is_initialized():
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # for logging
        
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        if scaler is not None:
            scaler.unscale_(optimizer) # important that this is done before clip_grad_norm
            # In distributed training, all ranks must agree on whether to skip the step.
            # Each rank may independently encounter inf/nan gradients, so we all-reduce
            # the found_inf flag (MAX = if any rank found inf, all ranks skip).
            if dist.is_available() and dist.is_initialized():
                for v in scaler._found_inf_per_device(optimizer).values():
                    dist.all_reduce(v, op=dist.ReduceOp.MAX)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_threshold) # we need to do this after unscaling, otherwise it uses the scaled up values to determine if the clipping threshold has been reached
            scaler.step(optimizer)
            scaler.update()
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm_threshold)
            optimizer.step()

        ema.update(model.parameters())
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t1 = time.time()

        write0(f"Step {step}:{' '*(8 - len(str(step)))}{(t1-t0)*1000:.0f}ms    train loss: {train_loss.item():.6f}    lr: {lr:.10f}    grad norm: {norm.item():.6f} \n", log_file=log_file)
        csv_val_loss = None if not computed_val_loss_this_iteration else val_loss.item()
        write0(",".join(map(str, [step, train_loss.item(), csv_val_loss, lr, norm.item()])) + "\n", log_file=log_csv)
        # with open(log_file, 'a') as f:
        #     f.write(f"    {step=}    {rank=}   {batch['ph_padding_mask'].shape=}    {torch.sum(batch['ph_padding_mask'])=}    {batch['mel_padding_mask'].shape=}    {torch.sum(batch['mel_padding_mask'])=}\n")

    if ddp:
        dist.destroy_process_group()
