from typing import Union

import numpy as np
import faiss


class FaissKNeighbors:
    def __init__(self, k: int = 20, metric: Union["euclid", "cosine"] = "euclid"):
        self.index = None
        self.d = None
        self.k = k
        self.metric = metric

    def fit(self, X: np.ndarray):
        X = X.copy(order="C")
        self.d = X.shape[1]
        X = X.astype(np.float32)
        if self.metric == "cosine":
            self.index = faiss.IndexFlatIP(self.d)  # cosine
            faiss.normalize_L2(X)
        elif self.metric == "euclid":
            self.index = faiss.IndexFlatL2(self.d)  # euclid
        self.index.add(X)

    def predict(self, X: np.ndarray):
        X = X.copy(order="C")
        X = np.reshape(X, (-1, self.d))
        X = X.astype(np.float32)
        if self.metric == "cosine":
            faiss.normalize_L2(X)
        distances, indices = self.index.search(X, k=self.k)
        if self.metric == "euclid":
            distances = np.sqrt(distances)
        if X.shape[0] == 1:
            return distances[0], indices[0]
        else:
            return distances, indices
