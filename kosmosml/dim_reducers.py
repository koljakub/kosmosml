"""Dimensionality reduction algorithms.

"""
from __future__ import annotations

from io import BytesIO
from typing import List, Generator, Union

import joblib
import numpy as np
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.preprocessing import normalize

__all__ = ['PCADimReducer']


class PCADimReducer:
    """An auxiliary wrapper class for sklearn's Principal Component Analysis (PCA). The class provides a convenient
    way to fit and use both PCA and Incremental PCA models.

    The class provides the following core methods:
    * fit: Fits a PCA model - training data completely in RAM, no batch processing.
    * incremental_fit: Fits an Incremental PCA model - training data completely in RAM, batch processing.
    * fit_generator: Fits an Incremental PCA model - training data yielded by a generator, batch processing.

    """

    def __init__(self, n_components: int, whiten: bool = True) -> None:
        self.n_components = n_components
        self.whiten = whiten
        self._model = None

    def fit(self, X: np.ndarray) -> None:
        """Fits a PCA model to X.

        More information can be found on:
        <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html>

        Args:
            X: Training data.

        """
        self._model = PCA(n_components=self.n_components, whiten=self.whiten)
        self._model.fit(X)

    def fit_generator(self, X_gen: Generator[np.ndarray, None, None], batch_size: int = None) -> None:
        """Fits an Incremental PCA model to X_gen while minimizing the RAM consumption as the training samples are
        yielded by a generator and processed using batch processing during the process of fitting.

        More information can be found on:
        <https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.IncrementalPCA.html>

        Args:
            X_gen: Training data - generator.
            batch_size: Batch size. If None, the batch size is inferred automatically from the input data.

        """
        if batch_size is None:
            # The following calculation of batch_size is derived from scikit-learn.
            batch_size = 5 * next(X_gen).shape[0]
        self._model = IncrementalPCA(n_components=self.n_components, whiten=self.whiten)
        batch = []

        for x in X_gen:
            batch.append(x)
            if len(batch) == batch_size:
                self._model.partial_fit(np.array(batch))
                batch = []

    def transform(self, vectors: List[np.ndarray], normalization: str = 'l2') -> List[np.ndarray]:
        """Applies a dimensionality reduction to a list of vectors.

        Args:
            vectors: List of vectors to be reduced.
            normalization: Type of normalization to be applied to the reduced vectors.
                           The following methods are supported: "l1", "l2", "max", None

        Returns:
            List of vectors whose dimension has been reduced.

        """
        X = self._model.transform(np.array(vectors))
        if normalization:
            X = normalize(X, axis=1, norm=normalization)
        return list(X)

    def save(self, target: Union[str, BytesIO]) -> None:
        """Saves the reducer to a target file or a binary stream.

        Args:
            target: Path to a file (Recommended file extension: ".joblib") or a binary stream (BytesIO).

        """
        joblib.dump(self, target)

    @staticmethod
    def load(source: Union[str, BytesIO]) -> PCADimReducer:
        """Loads the reducer from a file or a binary stream.

        Args:
            source: Path to a file or a binary stream (BytesIO).

        """
        return joblib.load(source)
