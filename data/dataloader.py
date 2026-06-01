import torch
import torch.distributed as dist
import numpy as np
import threading
import queue
import time

import h5py
from torch.nn.utils.rnn import pad_sequence


def pad_collate_fn(batch, padding_value):
    """`batch` is a dictionary of lists of tensors"""
    # (T,) tensors -- pad along T to (B, T_max)
    f0         = pad_sequence(batch["f0"],         batch_first=True, padding_value=padding_value)
    mel2ph     = pad_sequence(batch["mel2ph"],     batch_first=True, padding_value=padding_value)
    uv         = pad_sequence(batch["uv"],         batch_first=True, padding_value=padding_value)

    # (M, T) tensors -- transpose to (T, M), pad to (B, T_max, M), transpose back -- pad_sequence always pads along dimension 0
    mel        = pad_sequence([m.T for m in batch["mel"]], batch_first=True, padding_value=padding_value).permute(0, 2, 1)

    # (P,) tensors -- pad along P to (B, P_max)
    txt_tokens = pad_sequence(batch["txt_tokens"], batch_first=True, padding_value=padding_value)

    padded_batch = {
        'f0' : f0,
        'mel' : mel,
        'mel2ph' : mel2ph,
        'uv' : uv,
        'txt_tokens' : txt_tokens
    }
    return padded_batch


class NaiveDataLoader:
    """
    Dataloader for UniRef50 protein sequences stored as numpy memmaps.

    `tokens_path` is a path to a uint8 numpy memmap which is a memmap containing tokenized protein sequences concatenated together (without any delimiters between sequences)
    `lengths_path` is a path to a uint16 numpy memmap which is a memmap containing the token lengths of each protein sequence in the tokens memmap, this is
    used because we don't want attention to attend across sequences, so we'll use a block-diagonal attention mask and we need the lengths of the sequences
    in order to properly build this mask

    The main idea is that we don't want attention to attend across protein sequence boundaries, we want each sequence to get its own noise time sample, and we implement data
    prefetching. Because of this, instead of `batch_size` being the number of sequences of length `seq_len` (e.g. seq_len=1024), we'll use `batch_size` to refer to the number
    of *tokens* in the batch rather than the number of sequences in the batch. This is due to the protein sequence lengths varying and I wanted to avoid having a lot of padding
    tokens. Technically, there are other ways around this like using a dynamic batch size to group sequences of similar lengths together to reduce the amount of padding tokens
    in a given batch -- this is what EvoDiff does and it might initially sound better in that it sounds like sequences don't get truncated, but actually EvoDiff truncates to
    2048 residues (not necessarily the first 2048 though, I think it randomly selects a substring). My way avoids dealing with padding tokens and more tokens are included in the
    training set. I don't think the choice will make a big difference, I'm doing it this way because I think it is simpler and more tokens are seen. 

    Each call to `next_batch()` returns:
        x:        (batch_size,) int64 tensor of token ids
        seq_lens: list of ints -- seq_lens is a flat list of all sequence lengths across the batch. 
                  Sequences that span two batches are split across the seq_lens of those batches.
                  These can be passed directly to xformers' BlockDiagonalMask.from_seqlens().
        t:        (batch_size,) float32 tensor of uniform random times in [0,1] for flow matching -- tokens in the batch corresponding to the same protein sequence get the same time

    Supports DDP -- each rank maintains its own position in the token stream, offset by rank so ranks read non-overlapping data.

    Supports optional background prefetching via a daemon thread and a bounded queue. Set `prefetch_batches=0` to disable prefetching entirely (for benchmarking/debugging).

    State can be saved and restored via `get_state_dict()` and `load_state_dict()` for resuming training runs. Note: if prefetching is enabled, the queue is drained 
    before saving state so that no batches are skipped or repeated on resume.
    """

    def __init__(
        self,
        data_path: str,
        batch_size: int,
        padding_value: int,
        rng_seed: int = 21,
        prefetch_batches: int = 2,
    ):
        self.data_path = data_path
        with h5py.File(data_path, "r") as f:
            self.utterances = list(f.keys())
        self.batch_size = batch_size
        self.padding_value = padding_value
        self.prefetch_batches = prefetch_batches

        if dist.is_available() and dist.is_initialized():
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        torch.manual_seed(rng_seed + self.rank)

        self.reset()

        self._state_dict_batches = [] # when prefetching data, we save the prefetched batches to the state dict and use them first when we resume training since the memmap positions were already updated when prefetching them

        if self.prefetch_batches > 0: # TODO
            self._queue = queue.Queue(maxsize=self.prefetch_batches)
            self._stop_event = threading.Event()
            self._thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            self._thread.start()

    def reset(self):
        # TODO optionally shuffle self.utterances because we truncate the dataset and shuffling allows us to in principle see the whole dataset
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

        with h5py.File(self.h5_path, "r") as f:
            for utterance in utterances:
                _f0         = torch.from_numpy(f[utterance]["f0"][:]).float()         # (T,)
                _mel        = torch.from_numpy(f[utterance]["mel"][:]).float()        # (80, T)
                _mel2ph     = torch.from_numpy(f[utterance]["mel2ph"][:]).long()      # (T,)
                _uv         = torch.from_numpy(f[utterance]["uv"][:]).float()         # (T,)
                _txt_tokens = torch.from_numpy(f[utterance]["txt_tokens"][:]).long()  # (P,)

                batch['f0'].append(_f0)
                batch['mel_'].append(_mel)
                batch['mel2ph'].append(_mel2ph)
                batch['uv'].append(_uv)
                batch['txt_tokens'].append(_txt_tokens)

        self.current_position += self.batch_size * self.world_size
        if self.current_position + self.batch_size * self.world_size >= len(self.utterances):
            self.reset()

        padded_batch = pad_collate_fn(batch, padding_value=self.padding_value)

        # TODO maybe generate and return diffsion time steps as well
        return padded_batch

    # TODO check/fix everything below here!!!!!
    def _prefetch_worker(self):
        while not self._stop_event.is_set():
            batch = self._compute_next_batch()
            self._queue.put(batch)  # blocks if queue is full

    def next_batch(self, device: torch.device) -> tuple[torch.Tensor, list[list[int]], torch.Tensor]:
        if self._state_dict_batches:
            x, seq_lens, t = self._state_dict_batches.pop(0)
        elif self.prefetch_batches > 0:
            x, seq_lens, t = self._queue.get()
        else:
            x, seq_lens, t = self._compute_next_batch()

        if device.type == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            t = t.pin_memory().to(device, non_blocking=True)
        else:
            print(f"{device=}, {device.type=}")
            x = x.to(device)
            t = t.to(device)

        return x, seq_lens, t
    
    def get_state_dict(self) -> list | None:
        queued_batches = []
        if self.prefetch_batches > 0:
            while not self._queue.empty():
                queued_batches.append(self._queue.get())

        state = {
            "current_seq_idx": self.currentj_seq_idx,
            "current_seq_offset": self.current_seq_offset,
            "queued_batches": queued_batches,
        }
        all_states = [None] * self.world_size
        dist.all_gather_object(all_states, state)
        if self.rank == 0:
            return all_states
        return None

    def load_state_dict(self, all_states: list):
        dist.broadcast_object_list(all_states, src=0)
        my_state = all_states[self.rank]
        self.current_seq_idx = my_state["current_seq_idx"]
        self.current_seq_offset = my_state["current_seq_offset"]
        self.current_position = int(self.seq_start_positions[self.current_seq_idx]) + self.current_seq_offset
        self._state_dict_batches = my_state["queued_batches"]

    def __del__(self):
        if hasattr(self, "_stop_event"):
            self._stop_event.set()


# --------------------------------------------------------------------------------------
# Benchmark script
#
# Simulates a real training loop to measure whether the dataloader is actually a
# bottleneck. The key metrics are:
#   - dataloader wait time: how long next_batch() blocks per step
#   - GPU compute time: how long the fake forward/backward takes per step
#
# If dataloader wait time << GPU compute time, the dataloader is not a bottleneck
# and prefetching is working. If they are comparable, you need faster loading.
#
# --gpu_matmul_size controls the size of the fake matmul used to simulate GPU compute.
# Tune this to match the compute time of your actual model's forward/backward pass.
# --------------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens_path",     type=str, required=True)
    parser.add_argument("--lengths_path",    type=str, required=True)
    parser.add_argument("--batch_size",      type=int, default=2**19)
    parser.add_argument("--n_batches",       type=int, default=100,  help="number of batches to benchmark")
    parser.add_argument("--prefetch",        type=int, default=2,    help="prefetch_batches, set 0 to disable")
    parser.add_argument("--gpu_matmul_size", type=int, default=8192, help="size of fake matmul to simulate GPU compute; increase to simulate a larger model")
    parser.add_argument("--device",          type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda":
        print("WARNING: running on CPU, prefetching benchmark is not meaningful without a GPU")

    print(f"device={device} | batch_size={args.batch_size} | prefetch={args.prefetch} | gpu_matmul_size={args.gpu_matmul_size}")

    loader = UniRef50DataLoader(
        tokens_path=args.tokens_path,
        lengths_path=args.lengths_path,
        batch_size=args.batch_size,
        prefetch_batches=args.prefetch,
    )

    # warmup
    print("Warming up...")
    for _ in range(3):
        x, seq_lens, t = loader.next_batch(device)
        if device.type == "cuda":
            _ = torch.randn(args.gpu_matmul_size, args.gpu_matmul_size, device=device) @ torch.randn(args.gpu_matmul_size, args.gpu_matmul_size, device=device)
            torch.cuda.synchronize()

    # benchmark
    print(f"Benchmarking {args.n_batches} batches...")
    dataloader_wait_times = []
    gpu_compute_times = []

    for _ in range(args.n_batches):
        # measure how long we block waiting for the next batch
        t0 = time.perf_counter()
        x, seq_lens, t = loader.next_batch(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        dataloader_wait_times.append(time.perf_counter() - t0)

        # measure fake GPU compute (simulates forward/backward)
        t0 = time.perf_counter()
        if device.type == "cuda":
            _ = torch.randn(args.gpu_matmul_size, args.gpu_matmul_size, device=device) @ torch.randn(args.gpu_matmul_size, args.gpu_matmul_size, device=device)
            torch.cuda.synchronize()
        gpu_compute_times.append(time.perf_counter() - t0)

    dl_mean  = sum(dataloader_wait_times) / len(dataloader_wait_times)
    gpu_mean = sum(gpu_compute_times) / len(gpu_compute_times)
    tokens_per_batch = args.batch_size

    print(f"\nResults ({args.n_batches} batches):")
    print(f"  dataloader wait:   {dl_mean * 1000:.2f}ms/batch  (mean)")
    print(f"  GPU compute:       {gpu_mean * 1000:.2f}ms/batch  (mean)")
    print(f"  dataloader is {'NOT ' if dl_mean > gpu_mean * 0.05 else ''}the bottleneck")
    print(f"  tokens/sec:        {tokens_per_batch / (dl_mean + gpu_mean):,.0f}")
    print(f"\nSanity checks:")
    print(f"  x.shape:           {x.shape}")
    print(f"  x.device:          {x.device}")
    print(f"  sum(seq_lens)      {sum(seq_lens)} should equal {tokens_per_batch}")