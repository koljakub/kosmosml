# Neural Codes for Image Retrieval <https://arxiv.org/abs/1404.1777>.
from __future__ import annotations

from io import BytesIO
from typing import Union, List, Generator

import joblib
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

from kosmosml.dim_reducers import PCADimReducer
from kosmosml.nn.backbones.cbir import neuralcodes_16


class NeuralCodes16:
    """An implementation of a model described in Neural Codes for Image Retrieval <https://arxiv.org/abs/1404.1777>.

    The model consists of the following components:
    - VGG-16 convolutional neural network (CNN) for extraction of base image descriptors.
      (augmented with Global Sum Pooling and L2 normalization layers; without classification head)
    - Incremental PCA + PCA whitening. Number of components: 128
    - L2 normalization

    Args:
        device: 'cuda:X' for CUDA-capable device, where X stands for the ID of such device, eg. 'cuda:0'.
                'cpu' for CPU. It's recommended to use a GPU for training and a CPU for inference.

    """

    def __init__(self, device: str):
        self.device = device
        self._backbone = neuralcodes_16().to(device)
        self._backbone.eval()
        self._dim_reducer = PCADimReducer(n_components=128, whiten=True)
        self._transform_fn = Compose([Resize((224, 224)),
                                      ToTensor(),
                                      Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    @torch.no_grad()
    def _extract_backbone_emb(self, batch_imgs: List[Image.Image]) -> np.ndarray:
        # Extracts image descriptors from the provided batch of images.
        batch_tensor = torch.cat([self._transform_fn(img)[None, :] for img in batch_imgs]).to(self.device)
        return self._backbone(batch_tensor).squeeze().cpu().numpy()

    def _backbone_emb_gen(self, images: List[Image.Image], batch_size: int) -> Generator[np.ndarray, None, None]:
        # Generator. Computes image descriptor for each of the provided images.
        # Uses batch processing.
        batch_imgs = []
        for img in images:
            if len(batch_imgs) == batch_size:
                emb_vecs = self._extract_backbone_emb(batch_imgs)
                batch_imgs = []
                yield emb_vecs
            if len(batch_imgs) < batch_size:
                batch_imgs.append(img)
        if len(batch_imgs) > 0:
            yield self._extract_backbone_emb(batch_imgs)

    def _pca_data_gen(self, gen_batches: Generator[np.ndarray, None, None]) -> Generator[np.ndarray, None, None]:
        # Generator. Processes the batches of image descriptors and yields single image descriptors.
        # Used by the underlying PCA model in order to apply batch processing.
        for batch in gen_batches:
            for vec in batch:
                yield vec

    def fit(self, images: Union[List[Image.Image], Generator[Image.Image, None, None]], batch_size_backbone: int,
            batch_size_pca: int) -> None:
        """Fits the model to the training data.
        
        Args:
            images: A list of PIL Images or a generator which yield PIL Images.
            batch_size_backbone: Number of images to be processed by the backbone network.
                                 Choose an appropriate value with respect to the device - VRAM of a GPU / RAM of a CPU.
            batch_size_pca: Number of samples to be processed by the underlying Incremental PCA model.
                            The value must be greater than 128 and is inferred automatically if set to None.
            Note: The minimal size of the training dataset (number of images) must be greater or equal to the value of
                  "batch_size_pca" parameter. If "batch_size_pca" is None, than the minimal size of the dataset
                  must be greater than or equal to 640.

        """
        gen_batches = self._backbone_emb_gen(images, batch_size_backbone)
        X_gen = self._pca_data_gen(gen_batches)
        self._dim_reducer.fit_generator(X_gen, batch_size=batch_size_pca)

    def transform(self, images: Union[Image.Image, List[Image.Image]]) -> List[np.ndarray]:
        """Transforms PIL Image(s) into corresponding 128-dim. embedding vector(s).

        Args:
            images: A single PIL Image or a list of PIL Images.

        Returns:
            A list of 128-dim. embedding vectors. If a single vector is provided, the resulting embedding vector is
            wrapped into a list.

        """
        if not isinstance(images, list):
            images = [images]
        images = self._extract_backbone_emb(images)
        if len(images.shape) == 1:
            images = images.reshape(1, -1)
        return self._dim_reducer.transform(images)

    def set_device(self, device: str) -> None:
        """Sets the device used by the underlying feature extractor network (backbone).

        Args:
            device: 'cuda:X' for CUDA-capable device, where X stands for the ID of such device, eg. 'cuda:0'.
                    'cpu' for CPU. It's recommended to use a GPU for training and a CPU for inference.

        """
        self.device = device
        self._backbone.to(device)

    def save(self, target: Union[str, BytesIO]) -> None:
        """Saves the model to a target file or a binary stream.

        Args:
            target: Path to a file (Recommended file extension: ".joblib") or a binary stream (BytesIO).

        """
        joblib.dump(self, target)

    @staticmethod
    def load(source: Union[str, BytesIO]) -> NeuralCodes16:
        """Loads the model from a file or a binary stream.

        Args:
            source: Path to a file or a binary stream (BytesIO).

        """
        return joblib.load(source)
