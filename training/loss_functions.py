import torch



def L1_loss(ground_truth_mel, output_mel, mel_padding_mask):
    """
    `ground_truth_mel` is the ground-truth mel-spectrogram and `output_mel` is the mel-spectrogram output from the model.
        Both have shape (B, T, M)
    `mel_padding_mask` is True when mel-frames correspond to padding.
        shape (B, T)
    
    In practice, we need to ignore padding terms because the mel-frame (time) dimension is padded.

    In DiffSinger, they use F.l1_loss with reduction=None, but I'm going to choose to take the mean anyway.
    See: https://github.com/MoonInTheRiver/DiffSinger/blob/ce7789f1427ddcdec647b3ab2bf2d1b12134e51e/tasks/tts/fs2.py#L161
    """
    mask = torch.logical_not(mel_padding_mask) # (B, T)
    counts = mask.sum(dim=1) # (B,)
    mask = mask[..., None] # (B, T, 1)
    B, T, M = ground_truth_mel.shape
    target = ground_truth_mel * mask # (B, T, M)
    pred = output_mel * mask # (B, T, M)
    diff = target - pred # (B, T, M)
    loss = torch.abs(diff).sum(dim=(1,2)) / (counts * M) # normalize by T * M to take the average over all terms, but we use `counts` instead of `T` because we don't want to average over padding terms
    return loss


# TODO, define ssim loss because they use linear combination of l1 and ssim, see https://github.com/MoonInTheRiver/DiffSinger/blob/ce7789f1427ddcdec647b3ab2bf2d1b12134e51e/modules/commons/ssim.py#L354