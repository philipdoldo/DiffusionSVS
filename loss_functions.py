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
    B, T, M = ground_truth_mel.shape
    target = ground_truth_mel * mask # (B, M, T)
    pred = output_mel * mask # (B, M, T)
    diff = target - pred # (B, M, T)
    loss = torch.abs(diff).sum(dim=(1,2)) / (counts * M) # shape (B,) -- normalize by T * M * B to take the average over all terms, but we use `counts` instead of `T` because we don't want to average over padding terms
    loss = torch.sum(loss) / B # average over all batches
    return loss


# TODO, define ssim loss because they use linear combination of l1 and ssim, see https://github.com/MoonInTheRiver/DiffSinger/blob/ce7789f1427ddcdec647b3ab2bf2d1b12134e51e/modules/commons/ssim.py#L354

def ssim_loss(ground_truth_mel, output_mel, mel_padding_mask, bias=6.0):
    """
    `ground_truth_mel` is the ground-truth mel-spectrogram and `output_mel` is the mel-spectrogram output from the model.
        Both have shape (B, M, T)
    `mel_padding_mask` is True when mel-frames correspond to padding.
        shape (B, T)
    
        TODO check correctness of this function
    """
    mask = torch.logical_not(mel_padding_mask)

    # treat mel as a 1-channel 2D image: [B, 1, T, n_mel]
    x = output_mel[:, None] + bias
    y = ground_truth_mel[:, None] + bias

    # build 11x11 isotropic Gaussian window, shape [1, 1, 11, 11]
    coords = torch.arange(11, dtype=torch.float32, device=x.device) - 5
    g = torch.exp(-coords ** 2 / (2 * 1.5 ** 2))
    g = g / g.sum()
    window = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)  # [1, 1, 11, 11]

    def local_stats(a, b):
        mu_a  = F.conv2d(a, window, padding=5)
        mu_b  = F.conv2d(b, window, padding=5)
        var_a = F.conv2d(a * a, window, padding=5) - mu_a ** 2
        var_b = F.conv2d(b * b, window, padding=5) - mu_b ** 2
        cov   = F.conv2d(a * b, window, padding=5) - mu_a * mu_b
        return mu_a, mu_b, var_a, var_b, cov

    mu_x, mu_y, var_x, var_y, cov_xy = local_stats(x, y)

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (2 * mu_x * mu_y + C1) * (2 * cov_xy + C2) / ((mu_x**2 + mu_y**2 + C1) * (var_x + var_y + C2))  # [B, 1, T, n_mel]

    ssim_map = ssim_map.squeeze(1)        # [B, T, n_mel]
    loss_map = 1 - ssim_map               # 0 = identical, 2 = maximally different

    loss_map = loss_map * mask.unsqueeze(-1)
    return loss_map.sum() / mask.sum()