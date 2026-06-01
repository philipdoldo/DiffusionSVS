import h5py
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MelDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            self.keys = list(f.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        with h5py.File(self.h5_path, "r") as f:
            f0         = torch.from_numpy(f[key]["f0"][:]).float()         # (T,)
            mel        = torch.from_numpy(f[key]["mel"][:]).float()        # (80, T)
            mel2ph     = torch.from_numpy(f[key]["mel2ph"][:]).long()      # (T,)
            uv         = torch.from_numpy(f[key]["uv"][:]).float()         # (T,)
            txt_tokens = torch.from_numpy(f[key]["txt_tokens"][:]).long()  # (P,)
        return f0, mel, mel2ph, uv, txt_tokens


def pad_collate_fn(batch, padding_value):
    f0_list, mel_list, mel2ph_list, uv_list, txt_list = zip(*batch)

    # (T,) tensors — pad along T to (B, T_max)
    f0     = pad_sequence(f0_list,     batch_first=True, padding_value=padding_value)
    mel2ph = pad_sequence(mel2ph_list, batch_first=True, padding_value=padding_value)
    uv     = pad_sequence(uv_list,     batch_first=True, padding_value=padding_value)

    # (80, T) tensors — transpose to (T, 80), pad to (B, T_max, 80), transpose back
    mel = pad_sequence([m.T for m in mel_list], batch_first=True, padding_value=padding_value).permute(0, 2, 1)  # (B, 80, T_max)

    # (P,) tensors — pad along P to (B, P_max)
    txt_tokens = pad_sequence(txt_list, batch_first=True, padding_value=padding_value)

    return f0, mel, mel2ph, uv, txt_tokens


if __name__ == "__main__":
    # uv run torchrun --nproc_per_node=1 dataset.py
    import os
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler


    def setup():
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


    def cleanup():
        dist.destroy_process_group()


    setup()

    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device(f"cuda:{local_rank}")

    dataset = MelDataset("/home/phil/DiffSinger/binarized_data/train.h5")
    sampler = DistributedSampler(dataset, shuffle=True)

    from functools import partial
    padding_value = 0
    collate_fn = partial(pad_collate_fn, padding_value=padding_value)

    loader = DataLoader(
        dataset,
        batch_size=8,
        sampler=sampler,       # replaces shuffle=True when using DDP
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )

    step = 0 
    epoch = 0
    while step < 10000:
        sampler.set_epoch(epoch)  # must be called so shuffling differs per epoch

        for _step, (f0, mel, mel2ph, uv, txt_tokens) in enumerate(loader):
            f0         = f0.to(device)          # (B, T_max)
            mel        = mel.to(device)         # (B, 80, T_max)
            mel2ph     = mel2ph.to(device)      # (B, T_max)
            uv         = uv.to(device)          # (B, T_max)
            txt_tokens = txt_tokens.to(device)  # (B, P_max)

            # your training step here

            if dist.get_rank() == 0 and _step == 0:
                print(
                    f"Epoch {epoch} | "
                    f"f0: {tuple(f0.shape)}, mel: {tuple(mel.shape)}, "
                    f"mel2ph: {tuple(mel2ph.shape)}, uv: {tuple(uv.shape)}, "
                    f"txt_tokens: {tuple(txt_tokens.shape)}"
                    f"step: {step}"
                )
            step += 1
            print(f"{step=}, {f0.shape=}")
        epoch += 1

    cleanup()
