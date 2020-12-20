"""Normalization layers.

"""
import torch

from . import functional as F

__all__ = ['L1Normalization', 'L2Normalization']


class L1Normalization(torch.nn.Module):
    """Normalizes a tensor using L1-norm.

        Args:
            eps (float): Small value to avoid division by zero.

    """

    def __init__(self, eps=1e-12):
        super(L1Normalization, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.l1_normalize(x, self.eps)


class L2Normalization(torch.nn.Module):
    """Normalizes a tensor using L2-norm.

        Args:
            eps (float): Small value to avoid division by zero.

    """

    def __init__(self, eps=1e-12):
        super(L2Normalization, self).__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.l2_normalize(x, self.eps)
