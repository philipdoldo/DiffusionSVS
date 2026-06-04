import torch

class ExponentialMovingAverage:
    """
    Maintains exponential moving average of model parameters, i.e., ema_params = (1-a)*new_params + a*ema_params for a in [0,1]
    Based on: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion/blob/main/model/ema.py#L10

    When training diffusion models, people often use an EMA of weights for inference instead of the actual weights used during training
    """
    def __init__(self, params, decay=0.9999):
        """
            `params`: Iterable of `torch.nn.Parameter`; usually the result of `model.parameters()`.
            `decay` : float in [0,1]
        """
        if decay < 0 or decay > 1:
            raise ValueError(f"Decay must be in [0,1], but {decay=}")
        self.decay = decay
        self.ema_params = [p.clone().detach() for p in params if p.requires_grad]
        self.copied_params = []

    def update(self, params):
        """
        Update currently maintained parameters.
        Call this every time the parameters are updated, such as the result of the `optimizer.step()` call.
        Args:
            params: Iterable of `torch.nn.Parameter`; usually the same set of parameters used to initialize this object.
        """
        with torch.no_grad():
            for ema_p, p in zip(self.ema_params, [p for p in params if p.requires_grad]):
                ema_p.mul_(self.decay).add_(p, alpha=1 - self.decay) # ema_p = decay*ema_p + (1-decay)*p, update ema params in-place

    def copy_to(self, params):
        """
        Copy EMA parameters into given collection of parameters.
        Args:
            params: Iterable of `torch.nn.Parameter`; the parameters to be updated with the stored moving averages.
        """
        for ema_p, p in zip(self.ema_params, [p for p in params if p.requires_grad]):
            p.data.copy_(ema_p.data)

    def store(self, params):
        """
        Save the current parameters for restoring later.
        Args:
            params: Iterable of `torch.nn.Parameter`; the parameters to be temporarily stored.
        """
        self.copied_params = [p.clone() for p in params]

    def restore(self, params):
        """
        Restore the parameters stored with the `store` method. Useful to validate the model with EMA parameters without affecting the
        original optimization process. Store the parameters before the `copy_to` method. After validation (or model saving), use this
        to restore the former parameters.
        Args:
            params: Iterable of `torch.nn.Parameter`; the parameters to be updated with the stored parameters.
        """
        for t, p in zip(self.copied_params, params):
            p.data.copy_(t.data)

    def state_dict(self):
        return dict(decay=self.decay, ema_params=self.ema_params)

    def load_state_dict(self, state_dict, device=None):
        self.decay = state_dict['decay']
        self.ema_params = state_dict['ema_params']
        if device is not None:
            self.ema_params = [ema_p.to(device) for ema_p in self.ema_params]