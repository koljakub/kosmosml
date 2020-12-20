"""An implementation of a CNN architecture from Neural Codes for Image Retrieval <https://arxiv.org/abs/1404.1777>.

"""
import torch
import torchvision.models as models

from kosmosml.nn.layers import L2Normalization, GlobalSumPool2d

__all__ = ['neuralcodes_11', 'neuralcodes_13', 'neuralcodes_16', 'neuralcodes_19']


def _neuralcodes(arch: str) -> torch.nn.Sequential:
    """Builds a VGG network and augments it with Global Sum Pooling and L2 normalization layers.

    Args:
        arch: Architecture of an underlying VGG network.

    Returns:
        VGG model augmented with Global Sum Pooling and L2 normalization layers.

    """
    base_model = getattr(models, arch)(pretrained=True)
    base_layers = list(base_model.features)[:-2]
    base_layers.extend([GlobalSumPool2d(), L2Normalization()])
    return torch.nn.Sequential(*base_layers)


def neuralcodes_11() -> torch.nn.Sequential:
    """VGG 11-layer model from Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf> augmented with Global Sum Pooling and L2 normalization layers as presented in
    Neural Codes for Image Retrieval <https://arxiv.org/abs/1404.1777>.

    Returns:
        VGG-11 model augmented with Global Sum Pooling and L2 normalization layers.

    """
    return _neuralcodes('vgg11')


def neuralcodes_13() -> torch.nn.Sequential:
    """VGG 13-layer model from Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf> augmented with Global Sum Pooling and L2 normalization layers as presented in
    Neural Codes for Image Retrieval <https://arxiv.org/abs/1404.1777>.


    Returns:
        VGG-13 model augmented with Global Sum Pooling and L2 normalization layers.

    """
    return _neuralcodes('vgg13')


def neuralcodes_16() -> torch.nn.Sequential:
    """VGG 16-layer model from Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf> augmented with Global Sum Pooling and L2 normalization layers as presented in
    Neural Codes for Image Retrieval <https://arxiv.org/abs/1404.1777>.

    Returns:
        VGG-16 model augmented with Global Sum Pooling and L2 normalization layers.

    """
    return _neuralcodes('vgg16')


def neuralcodes_19() -> torch.nn.Sequential:
    """VGG 19-layer model from Very Deep Convolutional Networks For Large-Scale Image Recognition
    <https://arxiv.org/pdf/1409.1556.pdf> augmented with Global Sum Pooling and L2 normalization layers as presented in
    Neural Codes for Image Retrieval <https://arxiv.org/abs/1404.1777>.

    Returns:
        VGG-19 model augmented with Global Sum Pooling and L2 normalization layers.

    """
    return _neuralcodes('vgg19')
