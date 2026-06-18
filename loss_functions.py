import torch
import torch.nn.functional as F

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
    B, M, T = ground_truth_mel.shape
    target = ground_truth_mel * mask # (B, M, T)
    pred = output_mel * mask # (B, M, T)
    diff = target - pred # (B, M, T)
    loss = torch.abs(diff).sum(dim=(1,2)) / (counts * M) # shape (B,) -- normalize by T * M * B to take the average over all terms, but we use `counts` instead of `T` because we don't want to average over padding terms
    loss = torch.sum(loss) / B # average over all batches
    return loss


# TODO, define ssim loss because they use linear combination of l1 and ssim, see 

def ssim_loss(ground_truth_mel, output_mel, mel_padding_mask, bias=6.0):
    """
    `ground_truth_mel` is the ground-truth mel-spectrogram and `output_mel` is the mel-spectrogram output from the model.
        Both have shape (B, M, T)
    `mel_padding_mask` is True when mel-frames correspond to padding.
        shape (B, T)
    
        F.conv2d takes an input of shape (batch_size, in_channels, height, width) in its first positional argument
        `x` and `y` both have shape (B, 1, M, T) in our case
        `window` is input as the `weight` arg is F.conv2d which should have shape (out_channels, in_channels, new_height, new_width) assuming `groups=1` (which is the default behavior)
        `window` in our case has shape (1, 1, 11, 11)
        since our filter/window is basically an 11-by-11 matrix and we use padding=5, the height and width should be the same before/after the convolution,
        so the outputs all have shape (B, 1, M, T)
    
    Based implementation off of DiffSinger repo, see:
        https://github.com/MoonInTheRiver/DiffSinger/blob/ce7789f1427ddcdec647b3ab2bf2d1b12134e51e/tasks/tts/fs2.py#L166
        https://github.com/MoonInTheRiver/DiffSinger/blob/ce7789f1427ddcdec647b3ab2bf2d1b12134e51e/modules/commons/ssim.py#L354
    """
    mask = torch.logical_not(mel_padding_mask) # (B, T)
    B, M, T = output_mel.shape
    if ground_truth_mel.shape != output_mel.shape or T != mask.shape[-1] or B != mask.shape[0] or len(mask.shape) != 2:
        raise ValueError(f"{ground_truth_mel.shape=}, {output_mel.shape=}, {mask.shape=}, {B=}, {M=}, {T=}")

    # treat mel as a 1-channel 2D image: (B, 1, M, T)
    x = output_mel[:, None, :, :] + bias # (B, 1, M, T)
    y = ground_truth_mel[:, None, :, :] + bias # (B, 1, M, T)

    # build 11x11 isotropic Gaussian window, shape [1, 1, 11, 11]
    coords = torch.arange(11, dtype=torch.float32, device=x.device) - 5 #   tensor([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.,  5.])
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2)) # gaussian with sigma=1.5  tensor([0.0039, 0.0286, 0.1353, 0.4111, 0.8007, 1.0000, 0.8007, 0.4111, 0.1353, 0.0286, 0.0039])
    g = g / g.sum() # normalize to probability distribution                 tensor([0.0010, 0.0076, 0.0360, 0.1094, 0.2130, 0.2660, 0.2130, 0.1094, 0.0360, 0.0076, 0.0010])
    window = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # shape (1, 1, 11, 11)

    mu_x  = F.conv2d(input=x, weight=window, padding=5) # (B, 1, M, T)
    mu_y  = F.conv2d(input=y, weight=window, padding=5) # (B, 1, M, T)
    mu_x_sq = mu_x ** 2 # (B, 1, M, T)
    mu_y_sq = mu_y ** 2 # (B, 1, M, T)
    mu_xy = mu_x * mu_y # (B, 1, M, T)
    var_x = F.conv2d(input=x * x, weight=window, padding=5) - mu_x_sq # (B, 1, M, T)
    var_y = F.conv2d(input=y * y, weight=window, padding=5) - mu_y_sq # (B, 1, M, T)
    cov   = F.conv2d(input=x * y, weight=window, padding=5) - mu_xy # (B, 1, M, T)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = (2 * mu_xy + C1) * (2 * cov + C2) / ((mu_x_sq + mu_y_sq + C1) * (var_x + var_y + C2))  # (B, 1, M, T)

    ssim_map = ssim_map.squeeze(1) # (B, M, T)
    loss_map = 1 - ssim_map # 0 = identical, 2 = maximally different

    loss_map = loss_map * mask[:, None, :] # (B, M, T) * (B, 1, T) -- broadcasts over M
    return loss_map.sum() / (mask.sum() * M)