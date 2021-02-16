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
            axis = list(range(len(x.shape)))
            if batch:
                del axis[0]
            x /= np.linalg.norm(x, axis=tuple(axis))
        return x

    def distance(self, u, v):
        u = self.apply_norms(u)
        v = self.apply_norms(v)

        if self.metric == "cosine":
            return 1. - np.dot(u, v) / (np.linalg.norm(v) * np.linalg.norm(v))
        else:
            return np.linalg.norm(u - v)

    def __str__(self):
        return f"{self.metric}+{'normalize' if self.normalize else ''}"