import argparse
import tomllib
import toml
import time
import os
import datetime
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.model import MusicScoreEncoder, AuxiliaryDecoder, WaveNet, EncoderDecoder, WaveNetDenoiser
from DiffSinger.training.exponential_moving_average import ExponentialMovingAverage
from DiffSinger.data.dataloader import NaiveDataLoader

"""
TODO:
    - get stats to normalize mel-spectrograms (don't need to unnormalize outputs I think, vocoder should accept normalized inputs) -- modify binarize.py
    - get loss functions defined for both phases of training and handle using them correctly (feeding correct inputs to models, both for train and val, etc.)
        - need to load models properly depending on which phase of training is being done, just hardcode for now
    - get vocoder set up to test inference
"""

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
    return log_dir

def freeze(model):
    """freeze all weights of a model"""
    for param in model.parameters():
        param.requires_grad = False

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help=".toml file")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        config = tomllib.load(f)
    
    effective_batch_size = config["effective_batch_size"]
    batch_size = config["batch_size"]
    grad_accum_steps = config["grad_accum_steps"]
    training_steps = config["training_steps"]

    val_loss_interval = config["val_loss_interval"]
    checkpoint_interval = config["checkpoint_interval"]
    text_sample_interval = config["text_sample_interval"]

    save_dir = config["save_dir"]

    log_dir = create_log_dir(save_dir, config_name=os.path.basename(args.config))

    # Learning Rate Schedule (Cosine Decay -- warmup + constant if you let min_lr = max_lr and cosine_decay_steps=0)
    warmup_steps = config["warmup_steps"]
    max_lr = config["max_lr"]
    min_lr = config.get("min_lr", max_lr/10)
    lr_decay_steps = config.get("cosine_decay_steps", training_steps - warmup_steps)
    def get_lr(it):
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

    model_config = config["model"]
    if model_config["model_type"] == "EncoderDecoder": # Phase 1 of training
        model = EncoderDecoder(model_config)
    elif model_config["model_type"] == "WaveNetDenoiser": # Phase 2 of training
        model = WaveNetDenoiser(model_config)
        freeze(model.encoder)
    else:
        raise ValueError(f"{model_config['model_type']=}")

    if config.get("resume_training", False):
        checkpoint = torch.load(config["checkpoint_path"], map_location="cpu")
        # If we resume training, we change the rng seed in the config as a lazy way making sure we don't get the same random times and such
        # I didn't bother storing rng state of each rank because I might resume with a different number of gpus anyway and it is simpler this way
        # The checkpoint stores a set of all rng seeds used across all training runs to be sure we never repeat any of them (the gpus I'm using can have a lot of issues)
        if config["rng_seed"] in checkpoint["rng_seeds"]:
            raise ValueError(f"Change the rng seed in the config before you resume training! {checkpoint['rng_seeds']=}, {config['rng_seed']=}")

        model.load_state_dict(checkpoint["model"])
        print0(f"MODEL LOADED WITH CHECKPOINT {config['checkpoint_path']}\n")
    prior_rng_seeds = checkpoint["rng_seeds"] if config.get("resume_training", False) else set() # to be stored in checkpoint to be sure we don't accidentally resume training with a previously used rng seed
    prior_rng_seeds.add(config["rng_seed"])

    ddp = int(os.environ.get('RANK', -1)) != -1

    if ddp:
        dist.init_process_group(backend='nccl')
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = dist.get_world_size()

        assert rank == dist.get_rank(), f"{rank=}, {dist.get_rank()=}"

        device = f'cuda:{local_rank}'
        torch.cuda.set_device(device)
        print(f"{rank=}, {local_rank=}, {world_size=}, {device=}")
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"{rank=}, {local_rank=}, {world_size=}, {device=}")

    # sanity check inputs
    if world_size * batch_size * grad_accum_steps != effective_batch_size:
        raise ValueError(f"{effective_batch_size=}, {world_size=}, {batch_size=}, {grad_accum_steps=}, {world_size*batch_size*grad_accum_steps=}")

    # Initialize log file
    log_file = f"{log_dir}/log.txt"
    if rank == 0:
        with open(log_file, 'w') as f:
            f.write("") # initialize log file

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

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    write0(f"Model Parameters: {num_params:,}\nTrainable Model Parameters: {num_trainable_params:,}\n", log_file=log_file)

    train_loader = NaiveDataLoader(data_path=config["train_data_path"], batch_size=batch_size, padding_value=config["model"]["pad_token_id"], rng_seed=config["rng_seed"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        weight_decay=config["AdamW_weight_decay"],
        betas=config["AdamW_betas"],
        eps=config["AdamW_epsilon"],
        fused=True
        )
    
    ema = ExponentialMovingAverage(params=model.parameters(), decay=config["ema_decay"])

    if config.get("resume_training", False):
        optimizer.load_state_dict(checkpoint["optimizer"])
        ema.load_state_dict(checkpoint["ema"], device=device)

        if rank == 0:
            dataloader_state = checkpoint["dataloader"]
        else:
            dataloader_state = [None] * world_size # placeholder; rank 0 will broadcast
        train_loader.load_state_dict(dataloader_state) # Must be called on ALL ranks (does broadcast internally)

        initial_step = checkpoint["step"]

        write0(f"RESUMING TRAINING WITH CHECKPOINT {config['checkpoint_path']} AT STEP {initial_step}\n", log_file=log_file)
    else:
        initial_step = 0 # if not resuming training, have training loop start at step 0

    torch.manual_seed(config["rng_seed"] + rank) # (I guess this affects the categorical sampling, random times are in the dataloader)
    ctmc = UniformCTMC(config) # TODO fix this and figure out training losses and such

    for step in range(initial_step, training_steps):

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
                }
                checkpoint_path = os.path.join(log_dir, f'checkpoints/checkpoint_step{step}.pt')
                torch.save(checkpoint, checkpoint_path)
                torch.cuda.synchronize()
                t1 = time.time()
                write0(f" --- Checkpoint saved to {checkpoint_path} in {t1-t0:.4f}s\n", log_file=log_file)

        if rank == 0 and (step % val_loss_interval == 0 or step == training_steps - 1): # TODO
            with torch.no_grad():
                
                t0 = time.time()
                rng_state = torch.get_rng_state() # val might change rng state on rank 0, so save and restore it just in case, probably not very important
                val_loader = NaiveDataLoader(data_path=config["val_data_path"], batch_size=batch_size, padding_value=config["model"]["pad_token_id"]) # Should be reinitialized with same rng seed every time. Also notice how I intentionally use the default rng seed for val loader so that it never changes even when I resume training with a new rng seed in my config
                val_loader.reset() # should be unnecessary

                ema.store(model.parameters()) # store copy of the actual model weights
                ema.copy_to(model.parameters()) # copy EMA weights into the model
                val_losses = []
                for val_step in range(config.get("val_steps", 98)):
                    val_x0, val_t = val_loader.next_batch()
                    val_x0 = val_x0.to(device)
                    val_t = val_t.to(device)
                    val_loss = ctmc.loss_DWDSE(log_score_model=model, x_0=val_x0, t=val_t)
                    val_losses.append(val_loss)
                val_loss = sum(val_losses) / len(val_losses)
                ema.restore(model.parameters()) # copy stored model weights back into the model
                torch.set_rng_state(rng_state) # restore rng state on rank 0
                t1 = time.time()
                write0(f"val loss: {val_loss}{' '*(8 - len(str(step)))}{(t1-t0)*1000:.0f}ms\n", log_file=log_file)

        torch.cuda.synchronize()
        t0 = time.time()

        train_loss = 0.0 # for logging
        for micro_step in range(grad_accum_steps):

            x0, t = train_loader.next_batch()
            x0 = x0.to(device)
            t = t.to(device)

            if ddp: # only sync gradients on the last micro step
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)

            # Effective batch size is per_gpu_batch_size * num_gpus * grad_accum_steps. Before syncing gradients, the local gradient has
            # per_gpu_batch_size in the denominator (since the loss is averaged over the local batch) and then when we sync gradients with
            # the .backward() call (with require_backward_grad_sync True), they are averaged over all ranks, so the resulting gradient has
            # per_gpu_batch_size * num_gpus in the denominator. Dividing the loss by grad_accum_steps gives us the correct final denominator.
            loss = ctmc.loss_DWDSE(log_score_model=model, x_0=x0, t=t) / grad_accum_steps
            train_loss += loss.detach() # for logging
            loss.backward()

        if dist.is_initialized():
            dist.all_reduce(train_loss, op=dist.ReduceOp.AVG) # for logging
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = get_lr(step)
        optimizer.step()
        ema.update(model.parameters())
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t1 = time.time()

        write0(f"Step {step}:{' '*(8 - len(str(step)))}{(t1-t0)*1000:.0f}ms    train loss: {train_loss.item():.6f}    grad norm: {norm.item():.6f}\n", log_file=log_file)
    
    if ddp:
        dist.destroy_process_group()
