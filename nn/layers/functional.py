"""Functional interface.

"""
import torch

__all__ = ['l1_normalize', 'l2_normalize']


# Normalization layers
# --------------------


def l1_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalizes the input tensor using L1-norm.

    Args:
        x: Tensor to be normalized.
        eps: Small value to avoid division by zero.

    Returns:
        Normalized tensor.

    """
    return x / (torch.norm(x, p=1, dim=1, keepdim=True) + eps).expand_as(x)


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalizes the input tensor using L2-norm.

    Args:
        x: Tensor to be normalized.
        eps: Small value to avoid division by zero.

    Returns:
        Normalized tensor.

    """
    return x / (torch.norm(x, p=2, dim=1, keepdim=True) + eps).expand_as(x)
