import numpy as np


class DistMetric:

    def __init__(self, metric, normalize=True, mean=None):
        assert metric in ("cosine", "euclidean"), f"{metric} not supported"
        if metric == "cosine":
            assert normalize, "normalize required for cosine"

        self.metric = metric
        self.normalize = normalize
        self.mean = mean

    def apply_norms(self, x, batch=False):
        if self.mean:
            x -= self.mean
        if self.normalize:
            axis = (1,) if batch else (0, 1)
            x /= np.linalg.norm(x, axis=axis, keepdims=True)
        return x

    def distance(self, u, v, batch=False):
        if self.metric == "cosine":
            return 1. - np.dot(u, v.T)
        else:
            axis = (1,) if batch else (0, 1)
            return np.linalg.norm(u - v, axis=axis, keepdims=True)

    def __str__(self):
        return f"{self.metric}" \
               f"+{'normalize' if self.normalize else ''}" \
               f"+{'mean' if self.mean else ''}"
