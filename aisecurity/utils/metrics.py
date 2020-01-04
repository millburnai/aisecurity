"""

"aisecurity.utils.metrics"

Metrics for facial recognition.

"""

import numpy as np
from scipy.spatial.distance import cosine


class DistMetric:

    # AVAILABLE MODES
    MODES = {
        "euclidean": lambda a, b: 1. - np.linalg.norm(a - b),
        "cosine": lambda a, b: 1. - cosine(a, b)
    }

    NORMALIZATIONS = {
        "subtract_mean": {
            "func": lambda a, b: np.mean(np.concatenate([a, b]), axis=0),
            "action": lambda a, b, res: (a - res, b - res)
        }
    }


    # INITS
    def __init__(self, mode, normalizations=None):
        if "+" in mode:  # ex: DistMetric("euclidean+subtract_mean")
            mode_and_norms = mode.split("+")
            self.mode = mode_and_norms[0]
            self.normalizations = mode_and_norms[1:]
        else:  # ex: DistMetric("euclidean", ["subtract_mean"])
            self.mode = mode
            self.normalizations = normalizations

        assert self.mode in self.MODES, "supported modes are {}".format(list(self.MODES.keys()))
        assert self.normalizations is None or all(norm in self.NORMALIZATIONS for norm in self.normalizations), \
            "supported normalizations are {}".format(list(self.NORMALIZATIONS.keys()))


    # MAGIC FUNCTIONS
    def __call__(self, a, b):
        if a.shape != b.shape:
            a = a.reshape(b.shape)

        if self.normalizations is not None:
            for norm in self.normalizations:
                res = self.NORMALIZATIONS[norm]["func"](a, b)
                a, b = self.NORMALIZATIONS[norm]["action"](a, b, res)
        dist = self.MODES[self.mode](a, b)

        return dist
