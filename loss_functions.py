import torch

def get_loss_function(config):
    weights = {
        'l1': config.get('l1_weight', 0),
        'ssim': config.get('ssim_weight', 0),
    }
    loss_fns = {
        'l1': L1_loss,
        'ssim': ssim_loss,
    }

    def loss(ground_truth_mel, output_mel, mel_padding_mask):
        at_least_one_nonzero_loss_weight = False
        total = 0.0
        for name, w in weights.items():
            if w != 0.0:
                total = total + w * loss_fns[name](ground_truth_mel, output_mel, mel_padding_mask)
                at_least_one_nonzero_loss_weight = True
        if not at_least_one_nonzero_loss_weight:
            raise ValueError(f"{config=}")
        return total

    return loss


def L1_loss(ground_truth_mel, output_mel, mel_padding_mask):
    """
    `ground_truth_mel` is the ground-truth mel-spectrogram and `output_mel` is the mel-spectrogram output from the model.
        Both have shape (B, M, T)
    `mel_padding_mask` is True when mel-frames correspond to padding.
        shape (B, T)
    
    In practice, we need to ignore padding terms because the mel-frame (time) dimension is padded.

    In DiffSinger, they use F.l1_loss with reduction=None, but I'm going to choose to take the mean anyway.
    See: https://github.com/MoonInTheRiver/DiffSinger/blob/ce7789f1427ddcdec647b3ab2bf2d1b12134e51e/tasks/tts/fs2.py#L161
    """
    mask = torch.logical_not(mel_padding_mask) # (B, T)
    counts = mask.sum(dim=1) # (B,)
    mask = mask[:, None, :] # (B, 1, T)
    B, T, M = ground_truth_mel.shape
    target = ground_truth_mel * mask # (B, M, T)
    pred = output_mel * mask # (B, M, T)
    diff = target - pred # (B, M, T)
    loss = torch.abs(diff).sum(dim=(1,2)) / (counts * M) # shape (B,) -- normalize by T * M * B to take the average over all terms, but we use `counts` instead of `T` because we don't want to average over padding terms
    loss = torch.sum(loss) / B # average over all batches
    return loss


# TODO, define ssim loss because they use linear combination of l1 and ssim, see https://github.com/MoonInTheRiver/DiffSinger/blob/ce7789f1427ddcdec647b3ab2bf2d1b12134e51e/modules/commons/ssim.py#L354

def ssim_loss(ground_truth_mel, output_mel, mel_padding_mask):
    return None # TODO -- be sure to account for padding properly