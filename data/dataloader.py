import torch
import torch.distributed as dist
import numpy as np
import threading
import queue
import time

import h5py
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import math
from functools import partial

def pad_and_norm_collate_fn(batch, padding_value, stats=None, mel_pad_multiple=1):
    """`batch` is a dictionary of lists of tensors"""
    # (T,) tensors -- pad along T to (B, T_max)
    f0         = pad_sequence(batch["f0"],         batch_first=True, padding_value=padding_value) # (B, T_max)
    mel2ph     = pad_sequence(batch["mel2ph"],     batch_first=True, padding_value=padding_value) # (B, T_max)
    uv         = pad_sequence(batch["uv"],         batch_first=True, padding_value=padding_value) # (B, T_max)

    # (M, T) tensors -- transpose to (T, M), pad to (B, T_max, M), transpose back -- pad_sequence always pads along dimension 0
    mel        = pad_sequence([m.T for m in batch["mel"]], batch_first=True, padding_value=padding_value).permute(0, 2, 1) # (B, M, T_max)
    if "epsilon" in batch:
        epsilon    = pad_sequence([m.T for m in batch["epsilon"]], batch_first=True, padding_value=padding_value).permute(0, 2, 1) # (B, M, T_max)

    # (P,) tensors -- pad along P to (B, P_max)
    txt_tokens = pad_sequence(batch["txt_tokens"], batch_first=True, padding_value=padding_value) # (B, P_max)

    # Pad T further to the nearest multiple of `mel_pad_multiple` (rounding up) -- useful if using patchify in DiT, for example
    T_max = mel.shape[-1]
    T_padded = math.ceil(T_max / mel_pad_multiple) * mel_pad_multiple
    extra = T_padded - T_max
    if extra > 0:
        f0     = F.pad(f0,     (0, extra), value=padding_value)
        mel2ph = F.pad(mel2ph, (0, extra), value=padding_value)
        uv     = F.pad(uv,     (0, extra), value=padding_value)
        mel    = F.pad(mel,    (0, extra), value=padding_value)
        if "epsilon" in batch:
            epsilon = F.pad(epsilon, (0, extra), value=padding_value)

    # Compute masks: True if padding, False otherwise
    mel_lengths = torch.tensor([x.shape[-1] for x in batch["mel"]]) # (B,)
    txt_lengths = torch.tensor([x.shape[0] for x in batch["txt_tokens"]]) # (B,)
    P_max = txt_tokens.shape[-1]
    mel_mask = torch.arange(T_padded)[None, :] >= mel_lengths[:, None] # (B, T_padded)
    txt_mask = torch.arange(P_max)[None, :] >= txt_lengths[:, None] # (B, P_max)
    
    if stats is not None:
        # normalize mel-spectrograms to [-1, 1]
        mel_min = stats['mel_min'][None, :, None] # (1, M, 1)
        mel_max = stats['mel_max'][None, :, None] # (1, M, 1)
        mel = 2 * (mel - mel_min) / (mel_max - mel_min) - 1

        # standardize f0 to be mean 0 and standard deviation 1
        f0 = (f0 - stats['f0_mean']) / stats['f0_std'] # scalar mean and std should broadcast properly

    collated_batch = {
        'f0' : f0,
        'mel' : mel,
        'mel2ph' : mel2ph,
        'uv' : uv,
        'txt_tokens' : txt_tokens,
        'mel_padding_mask' : mel_mask,
        'ph_padding_mask' : txt_mask,
    }
    if "epsilon" in batch:
        collated_batch['epsilon'] = epsilon
    for k, v in collated_batch.items(): # do this in case there are other keys that we don't want to process in the collate function that we don't want to lose
        batch[k] = v
    return batch


class NaiveDataLoader:
    """
    Supports DDP -- each rank maintains its own position in the token stream, offset by rank so ranks read non-overlapping data.

    Supports optional background prefetching via a daemon thread and a bounded queue. Set `prefetch_batches=0` to disable prefetching entirely (for benchmarking/debugging).

    State can be saved and restored via `get_state_dict()` and `load_state_dict()` for resuming training runs. Note: if prefetching is enabled, the queue is drained 
    before saving state so that no batches are skipped or repeated on resume.

    `diffusion_k` is the integer value from Algorithm 1 of DiffSinger such that we sample random timesteps from the set {1, ..., k} during training if doing diffusion. If
    `diffusion_k` is None, then we do not do diffusion (as in DiffSinger).
    
    `diffusion_k` should only not be None if using setting `diffusion_type` to be `"DiffSingerDiffusion"`, other noise types change the data we load. For example, setting
    `diffusion_type` to `SimpleFlow` will cause the dataloader to load batches of times `t` in [0,1] rather than integer times in {1, ..., diffusion_k} as in DiffSingerDiffusion
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        padding_value: int,
        rng_seed: int = 21,
        prefetch_batches: int = 2,
        collate_fn: callable = pad_and_norm_collate_fn,
        diffusion_k: int = None,
        stats_path: str = None, # .npz files
        diffusion_type: str = None,
        mel_pad_multiple: int = 1,
    ):
        self.data_path = data_path
        with h5py.File(data_path, "r") as f:
            self.utterances = list(f.keys())
        self.batch_size = batch_size
        self.padding_value = padding_value
        self.prefetch_batches = prefetch_batches
        self.collate_fn = partial(collate_fn, mel_pad_multiple=mel_pad_multiple)
        self.diffusion_k = diffusion_k
        self.diffusion_type = diffusion_type

        ##### Validate that the inputted diffusion_type is consistent with other input arguments
        self.DIFF_SINGER_DIFFUSION = "DiffSingerDiffusion"
        self.SIMPLE_FLOW = "SimpleFlow"
        self.VALID_diffusion_typeS = {self.DIFF_SINGER_DIFFUSION, self.SIMPLE_FLOW, None}
        if self.diffusion_type not in self.VALID_diffusion_typeS:
            raise ValueError(f"`diffusion_type` must be in {self.VALID_diffusion_typeS}, but got {self.diffusion_type=}")
        if self.diffusion_type == "DiffSingerDiffusion":
            if self.diffusion_k is None:
                raise ValueError(f"if using {self.diffusion_type=}, then must set `diffusion_k` to an approriate integer, e.g. 100")
        else:
            if self.diffusion_k is not None:
                raise ValueError(f"if using {self.diffusion_type=}, then must set `diffusion_k` to be `None`, but got {self.diffusion_k=}")
        #####

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        torch.manual_seed(rng_seed + self.rank)

        self.reset()

        self._state_dict_batches = [] # when prefetching data, we save the prefetched batches to the state dict and use them first when we resume training since the memmap positions were already updated when prefetching them

        if self.prefetch_batches > 0:
            self._queue = queue.Queue(maxsize=self.prefetch_batches)
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self._thread.start()
        
        # load .npz file containing stats on f0 and mel-spectrograms to be used for normalization inside of the collate function
        # e.g. mel_mean has shape (mel_bins,) which is typically (80,), f0_mean is a scalar. 
        self.stats = None
        if stats_path is not None:
            self.stats = np.load(stats_path) # self.stats.files = ['mel_mean', 'mel_std', 'mel_min', 'mel_max', 'mel_median', 'f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_median']

    def reset(self):
        # TODO optionally shuffle self.utterances because we truncate the dataset and shuffling allows us to in principle see the whole dataset
        # maybe don't shuffle though because then harder to track stuff 
        self.current_position = self.batch_size * self.rank

    def _compute_next_batch(self) -> tuple[torch.Tensor, list[list[int]], torch.Tensor]:

        utterances = self.utterances[self.current_position : self.current_position + self.batch_size]

        batch = {
            'f0' : [],
            'mel' : [],
            'mel2ph' : [],
            'uv' : [],
            'txt_tokens' : [],
        }

        if self.diffusion_type is not None:
            batch['epsilon'] = []

        with h5py.File(self.data_path, "r") as f:
            for utterance in utterances:
                _f0         = torch.from_numpy(f[utterance]["f0"][:]).float()         # (T,)
                _mel        = torch.from_numpy(f[utterance]["mel"][:]).float()        # (80, T)
                _mel2ph     = torch.from_numpy(f[utterance]["mel2ph"][:]).long()      # (T,)
                _uv         = torch.from_numpy(f[utterance]["uv"][:]).float()         # (T,)
                _txt_tokens = torch.from_numpy(f[utterance]["txt_tokens"][:]).long()  # (P,)

                batch['f0'].append(_f0)
                batch['mel'].append(_mel)
                batch['mel2ph'].append(_mel2ph)
                batch['uv'].append(_uv)
                batch['txt_tokens'].append(_txt_tokens)
                if self.diffusion_type is not None:
                    batch['epsilon'].append(torch.randn_like(_mel))

        self.current_position += self.batch_size * self.world_size
        if self.current_position + self.batch_size * self.world_size >= len(self.utterances):
            self.reset()

        if self.diffusion_k is not None:
            assert self.diffusion_type == self.DIFF_SINGER_DIFFUSION, f"{self.diffusion_type=}"
            t = torch.randint(1, self.diffusion_k+1, size=(self.batch_size,)) # batch of random integers in {1, ..., k}
            batch['t'] = t
        elif self.diffusion_type == self.SIMPLE_FLOW:
            t = torch.rand(size=(self.batch_size,)) # batch of random times in [0, 1]
            batch['t'] = t

        collated_batch = self.collate_fn(batch, padding_value=self.padding_value, stats=self.stats)

        return collated_batch

    def _prefetch_worker(self):
        while not self._stop_event.is_set():
            batch = self._compute_next_batch()
            self._queue.put(batch) # blocks if queue is full

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, list[list[int]], torch.Tensor]:
        if self._state_dict_batches:
            batch = self._state_dict_batches.pop(0)
        elif self.prefetch_batches > 0:
            batch = self._queue.get()
        else:
            batch = self._compute_next_batch()

        # TODO, is there a cleaner way to handle this?
        if isinstance(device, str):
            is_cuda = device.startswith("cuda")
        else:
            is_cuda = device.type == "cuda"

        if is_cuda: 
            batch = {k : v.pin_memory().to(device, non_blocking=True) for k, v in batch.items()}
        else:
            print(f"{device=}")
            batch = {k : v.to(device) for k, v in batch.items()}

        return batch
    
    def get_state_dict(self) -> list | None:
        """
        Call `get_state_dict` on all ranks in your training script, but only save the checkpoint on rank 0 (the information from
        the other ranks get all-gathered to rank 0). Then, to load the state doing something like this:
            ```
            if rank == 0:
                checkpoint = torch.load("checkpoint_step1000.pt", weights_only=False)
                dataloader_state = checkpoint["dataloader"]
            else:
                dataloader_state = [None] * world_size # placeholder; rank 0 will broadcast
            dataloader.load_state_dict(dataloader_state) # Must be called on ALL ranks (does broadcast internally)
            ```
        """
        queued_batches = []
        if self.prefetch_batches > 0:
            while not self._queue.empty():
                queued_batches.append(self._queue.get())

        state = {
            "current_position": self.current_position,
            "queued_batches": queued_batches,
        }
        if self.world_size > 1:
            all_states = [None] * self.world_size
            dist.all_gather_object(all_states, state)
            return all_states if self.rank == 0 else None
        else:
            return [state]  # single-GPU: just wrap in a list

    def load_state_dict(self, all_states: list):
        if self.world_size > 1:
            dist.broadcast_object_list(all_states, src=0)
        my_state = all_states[self.rank]
        self.current_position = my_state["current_position"]
        self._state_dict_batches = my_state["queued_batches"]

    def __del__(self):
        if hasattr(self, "_stop_event"):
            self._stop_event.set()

