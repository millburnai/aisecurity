"""

"aisecurity.utils.distance"

Distance metrics for facial recognition.

"""

import warnings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class DistMetric:

    # AVAILABLE MODES
    MODES = {
        "euclidean": lambda a, b: np.linalg.norm(a - b),
        "cosine": lambda a, b: cosine_similarity(a, b)
    }

    NORMALIZATIONS = {
        "subtract_mean": {
            "state": 2,
            "func": lambda a, b: (a, b, np.mean(np.concatenate([a, b]), axis=0)),
            "operation": lambda a, b, res: (a - res, b - res)
        },
        "l2_normalize": {
            "state": 2,
            "func:": lambda a, b: (a / np.sqrt(np.maximum(np.sum(np.square(a), axis=-1, keepdims=True), 1e-6)),
                                   b / np.sqrt(np.maximum(np.sum(np.square(b), axis=-1, keepdims=True), 1e-6)),
            )
        }
    }


    # INITS
    def __init__(self, mode, normalizations=None):
        if isinstance(mode, str):  # if mode and norms are from defaults
            if "+" in mode:  # ex: DistMetric("euclidean+subtract_mean+l2_normalize")
                self.mode, *self.normalizations = mode.split("+")
            else:  # ex: DistMetric("euclidean", ["subtract_mean", "l2_normalize"])
                self.mode = mode
                self.normalizations = normalizations if normalizations else []

            assert self.mode in self.MODES, "supported modes are {}".format(list(self.MODES.keys()))
            assert self.normalizations is None or all(norm in self.NORMALIZATIONS for norm in self.normalizations), \
                "supported normalizations are {}".format(list(self.NORMALIZATIONS.keys()))

        elif callable(mode):  # if custom mode and maybe norms are provided
            self.mode = mode
            self.normalizations = normalizations

            test_case = np.random.random((2, ))
            try:
                assert isinstance(self.__call__(*test_case), float)
            except Exception:
                raise ValueError("test failed: check that custom mode and norm are in the correct format")

            warnings.warn("custom mode and normalizations are not supported in FaceNet")


    # HELPER FUNCTIONS
    def apply_norms(self, *args):
        normalized = args

        for norm in self.normalizations:
            try:
                functional_norm = self.NORMALIZATIONS[norm]
            except (KeyError, TypeError):  # will arise if custom funcs provided
                functional_norm = norm

            actions = functional_norm.copy()
            state = actions.pop("state")

            if state == len(args):
                tmp = normalized
                for action in actions.values():
                    tmp = action(*tmp)
                normalized = tmp

        if hasattr(normalized, "__len__"):
            assert len(normalized) == len(args), "mismatch between length of normalized and length of original args"
            return normalized if len(normalized) != 1 else normalized[0]
        return normalized

    def dist(self, a, b):
        try:
            return self.MODES[self.mode](a, b)
        except (KeyError, TypeError):
            return self.mode(a, b)


    # MAGIC FUNCTIONS
    def __call__(self, a, b):
        a, b = np.array(a), np.array(b)
        shape = a.shape if a.shape != () else (-1, 1)
        a = a.reshape(shape)
        b = b.reshape(shape)

        a, b = self.apply_norms(a, b)
        dist = self.dist(a, b)
        dist = self.apply_norms(dist)

        return dist

    def __str__(self):
        result = "Distance ("
        result += self.mode.__str__()  # __str__ to account for possibility of mode/normalizations being custom funcs
        for norm in self.normalizations:
            result += "+{}".format(norm.__str__())
        result += ")"
        return result

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    for i, n in np.random.random((10, 2)):
        test_case = [np.array([i]).reshape(-1, 1), np.array([n]).reshape(-1, 1)]

        dist_metric = DistMetric("euclidean+subtract_mean")
        print("(DistMetric)", dist_metric, ":", dist_metric(*test_case))

        dist_metric = DistMetric("euclidean+l2_normalize")
        print("(DistMetric)", dist_metric, ":", dist_metric(*test_case))

        dist_metric = DistMetric("euclidean+subtract_mean+l2_normalize")
        print("(DistMetric)", dist_metric, ":", dist_metric(*test_case))

        norm = lambda x: x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), 1e-6))
        subtract_mean = lambda a, b: (a-np.mean(np.concatenate([a, b]), axis=0),b-np.mean(np.concatenate([a, b]), axis=0))
        a, b = subtract_mean(test_case[0], test_case[1])

        print("Euclidean, subtract_mean :", np.linalg.norm(a - b))
        print("Euclidean, l2_normalize :", np.linalg.norm(norm(test_case[0]) - norm(test_case[1])))
        print("Euclidean, subtract_mean, l2_normalize :", np.linalg.norm(norm(a) - norm(b)))

        print("\n")
