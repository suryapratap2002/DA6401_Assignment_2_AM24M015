

import torch
import torch.nn as nn


class CustomDropout(nn.Module):


    def __init__(self, p: float = 0.5):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"Dropout probability must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p == 0.0:
            return x
        # Build a Bernoulli keep-mask: each element is kept with probability (1-p)
        keep_prob = 1.0 - self.p
        # torch.bernoulli samples from Bernoulli(keep_prob) for every element
        mask = torch.bernoulli(torch.full(x.shape, keep_prob, dtype=x.dtype, device=x.device))
        # Inverted dropout: scale up by 1/keep_prob to preserve expected value
        return x * mask / keep_prob

    def extra_repr(self) -> str:
        return f"p={self.p}"
