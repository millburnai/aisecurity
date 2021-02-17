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
            x /= np.linalg.norm(x, axis=tuple(axis), keepdims=True)
        return x

    def distance(self, u, v):
        if self.metric == "cosine":
            return 1. - np.dot(u, v) / (np.linalg.norm(v) * np.linalg.norm(v))
        else:
            return np.linalg.norm(u - v)

    def __str__(self):
        constr = f"{self.metric}" \
                 f"+{'normalize' if self.normalize else ''}" \
                 f"+{'mean' if self.mean else ''}"
        return f"Distance ({constr})"
