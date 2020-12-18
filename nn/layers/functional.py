"""Functional interface.

"""
import torch
import torch.nn.functional as F

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


# Pooling layers
# --------------


def global_sum_pool2d(x: torch.Tensor) -> torch.Tensor:
    """Performs global sum pooling over the input tensor composed of N 2D feature maps. The result of such operation
    can be interpreted as an N-dimensional vector where each component is a sum of the corresponding feature map.

    Args:
        x: Tensor.

    Returns:
        Tensor transformed via Global sum pooling.

    """
    return torch.sum(x.view(x.shape[0], x.shape[1], -1), dim=2).view(x.shape[0], x.shape[1], 1, 1)


def mac(x: torch.Tensor) -> torch.Tensor:
    """Performs Maximum Activation of Convolutions (MAC) pooling over the input tensor composed of N 2D
    feature maps. The result can be interpreted as an N-dimensional vector of maximum values from each of the feature
    maps.

    Args:
        x: Tensor.

    Returns:
        Tensor transformed via MAC pooling.

    """
    return F.adaptive_max_pool2d(x, (1, 1))
