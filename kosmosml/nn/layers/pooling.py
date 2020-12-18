"""Pooling layers.

"""
import torch

from . import functional as F

__all__ = ['GlobalSumPool2d', 'MAC']


class GlobalSumPool2d(torch.nn.Module):
    """Performs global sum pooling over a tensor composed of N 2D feature maps. The result of such operation
    can be interpreted as an N-dimensional vector where each component is a sum of the corresponding feature map.

    """

    def __init__(self):
        super(GlobalSumPool2d, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.global_sum_pool2d(x)


class MAC(torch.nn.Module):
    """Performs Maximum Activation of Convolutions (MAC) pooling over the input tensor composed of N 2D
    feature maps. The result can be interpreted as an N-dimensional vector of maximum values from each of the feature
    maps.

    """

    def __init__(self):
        super(MAC, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mac(x)
