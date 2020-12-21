"""Indexers for fast Nearest Neighbor Search (NNS).

"""
from __future__ import annotations

import os
from typing import List, Tuple, Generator

import hnswlib
import joblib
import numpy as np


class HnswIndexer():
    """Wrapper class for hnswlib's <https://github.com/nmslib/hnswlib> implementation of the HNSW algorithm.
    HNSW is a fast and robust algorithm for an approximate nearest neighbor search.

    Important: HNSW's parameters are described here: <https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md>.

    Args:
        embedding_dim: Dimension of embedding vectors.
        M: Defines tha maximum number of outgoing connections in the graph.
        ef_construction: Defines a construction time/accuracy trade-off.
        ef: Size of the dynamic list for the nearest neighbors (used during the search).
                  Higher 'ef' leads to more accurate but slower search. Can be set to a different value afterwards.
        metric: Type of a distance function, 'l2' by default, where 'l2' represents squared Euclidean distance.

    """

    def __init__(self, embedding_dim: int, M: int, ef_construction: int, ef: int, metric: str = 'l2',
                 random_seed: int = 42):
        self.embedding_dim = embedding_dim
        self.M = M
        self.ef_construction = ef_construction
        self.ef = ef
        self.metric = metric
        self.random_seed = random_seed
        self._index = hnswlib.Index(space=self.metric, dim=self.embedding_dim)
        self._label2int = None
        self._int2label = None

    def _init_index(self, num_elements: int) -> None:
        # Auxiliary method. Initializes the indexer.
        self._index.init_index(max_elements=num_elements, M=self.M, ef_construction=self.ef_construction,
                               random_seed=self.random_seed)
        self._index.set_ef(self.ef)

    def set_ef(self, ef: int) -> None:
        """Sets the parameter 'ef' to a new value.

        Args:
            ef: Size of the dynamic list for the nearest neighbors (used during the search).
                Higher 'ef' leads to more accurate but slower search.

        """
        self._index.set_ef(ef)

    def get_item_count(self) -> int:
        """Returns the number of items stored in the index.

        Returns:
            Number of items stored in the index.

        """
        return self._index.get_current_count()

    def _process_batch(self, batch: List[Tuple[str, np.ndarray]], num_processes: int) -> None:
        # Auxiliary method. Processes a single batch of data and inserts it into the index.
        labels, emb_vectors = zip(*batch)
        emb_vectors_arr = np.array(emb_vectors)
        labels_int_arr = np.array([self._label2int[label] for label in labels])
        self._index.add_items(data=emb_vectors_arr, ids=labels_int_arr, num_threads=num_processes)

    def fit_generator(self, gen_emb_vectors: Generator[Tuple[str, np.ndarray], None, None], max_elements: int,
                      batch_size: int, num_processes: int = 1) -> None:
        """Fits the indexer to the input data while minimizing the RAM consumption as the vectors and their IDs are
        yielded by a generator and processed via batch processing.

        Args:
            gen_emb_vectors: Generator which yields tuples of IDs (str) and embedding vectors (Numpy ndarray).
            max_elements: Maximum number of items in the final index.
            batch_size: Batch size.
            num_processes: Number of parallel sub-processes (1 per CPU core). Speeds up indexing significantly.
                           If set to -1, all CPU cores will be utilized.

        """
        self._init_index(max_elements)
        self._label2int = {}

        batch = []
        item_id = 0
        for item in gen_emb_vectors:
            self._label2int[item[0]] = item_id
            item_id += 1
            batch.append(item)
            if len(batch) == batch_size:
                self._process_batch(batch, num_processes)
                batch = []
        if len(batch) > 0:
            self._process_batch(batch, num_processes)

        self._int2label = {v: k for k, v in self._label2int.items()}

    def most_similar(self, emb_vec: np.ndarray, n: int, **kwargs) -> List[Tuple[str, float]]:
        """Finds N nearest neighbors to the input embedding vector.

        Args:
            emb_vec: Embedding vector.
            n: Number of nearest neighbors.
            num_processes: Number of parallel sub-processes (1 per CPU core). Speeds up indexing significantly.
                           If set to -1, all CPU cores will be utilized.

        Keyword Args:
            num_processes: Number of parallel sub-processes. Speeds up the search process.
                           If set to -1, all CPU cores will be utilized.

        Returns:
            List of tuples in the following format: (string identifier, distance to the source vector).

        """
        if 'num_processes' in kwargs:
            num_processes = kwargs['num_processes']
        else:
            num_processes = 1
        neighbors, dists = self._index.knn_query(emb_vec, k=n, num_threads=num_processes)
        return [(self._int2label[int_id], y) for int_id, y in zip(neighbors.ravel(), dists.ravel())]

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._index = None

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["_index"]
        return state

    def save(self, dpath: str) -> None:
        """Saves the indexer to a target directory.

        Args:
            dpath: Path to the target directory. Serialized indexer consists of the following 2 files:
                   index.bin, hnsw_indexer.joblib.

        """
        joblib.dump(self, os.path.join(dpath, 'hnsw_indexer.joblib'))
        self._index.save_index(os.path.join(dpath, 'index.bin'))

    @staticmethod
    def load(dpath: str) -> HnswIndexer:
        """Loads the indexer from the specified directory. The directory must include the following files: index.bin,
        hnsw_indexer.joblib.

        Args:
            dpath: Path to the directory where the serialized indexer is stored.

        """
        hnsw_indexer = joblib.load(os.path.join(dpath, 'hnsw_indexer.joblib'))
        index = hnswlib.Index(space=hnsw_indexer.metric, dim=hnsw_indexer.embedding_dim)
        index.load_index(os.path.join(dpath, 'index.bin'), max_elements=len(hnsw_indexer._label2int))
        index.set_ef(hnsw_indexer.ef)  # EF needs to be re-set after index loading
        hnsw_indexer._index = index
        return hnsw_indexer
