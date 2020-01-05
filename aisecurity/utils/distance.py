"""

"aisecurity.utils.distance"

Distance metrics for facial recognition.

"""

# FIXME: @orangese: this should be implemented as a preprocessing class, NOT a distance metric!

import warnings

import numpy as np


# HELPERS
_CHECKS = {
    "is_vector": lambda x: x.ndim <= 2 and (1 in x.shape or x.ndim == 1),
    "is_float_like": lambda x: isinstance(x, float) or isinstance(x, np.float),
    "is_shape_equal": lambda a, b: np.array(a).shape == np.array(b).shape
}


# DISTMETRIC
class DistMetric:


    NORMALIZATIONS = {
        # normalizations are applied before mode lambdas
        # ex: for "cosine+l2_normalize", L2 normalization is applied, then cosine normalization, then L2 distance
        "subtract_mean": {
            "apply_to": _CHECKS["is_vector"],
            "use_stat": True,
            "func": lambda x, stats: x - stats["mean"]
        },
        "l2_normalize": {
            "apply_to": _CHECKS["is_vector"],
            "use_stat": False,
            "func": lambda x: x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), 1e-6))
        },
        "sigmoid": {
            "apply_to": _CHECKS["is_float_like"],
            "use_stat": False,
            "func": lambda x: 1. / (1. + np.exp(-x))
        }
    }

    MODES = {
        # modes are the transformations to be applied before Euclidean distance is calculated
        # ex: for "euclidean", no transformation is applied --> lambda a, b: (a, b)
        "euclidean": lambda a, b: (a, b),
        "cosine": lambda a, b: (a, b)
    }



    # INITS
    def __init__(self, mode, normalizations=None, data=None):
        # if mode and norms are from defaults
        if isinstance(mode, str):
            if "+" in mode:
                # ex: DistMetric("euclidean+subtract_mean+l2_normalize")
                self.mode, *self.normalizations = mode.split("+")
            else:
                # ex: DistMetric("euclidean", ["subtract_mean", "l2_normalize"])
                self.mode = mode
                self.normalizations = normalizations if normalizations else []

            assert self.mode in self.MODES, "supported modes are {}".format(list(self.MODES.keys()))
            assert self.normalizations is None or all(norm in self.NORMALIZATIONS for norm in self.normalizations), \
                "supported normalizations are {}".format(list(self.NORMALIZATIONS.keys()))

        # if custom mode and norms are provided
        elif callable(mode):
            self.mode = mode
            self.normalizations = normalizations

            test_case = np.random.random((2, ))
            try:
                assert isinstance(self.__call__(*test_case), float)
            except Exception:
                raise ValueError("check failed: check that custom mode and norm are in the correct format")

            warnings.warn("custom mode and normalizations are not supported in FaceNet")

        # data setting
        if any(self.NORMALIZATIONS[norm]["use_stat"] for norm in self.normalizations):
            assert data is not None, "data must be provided for normalizations that use data statistics"

            self.stats = {
                "mean": np.mean(data),
                "std": np.std(data)
            }

        self.data = data


    # HELPER FUNCTIONS
    def apply_norms(self, arg):
        normalized = arg

        for norm_id in self.normalizations:
            try:
                functional_dict = self.NORMALIZATIONS[norm_id]
            except (KeyError, TypeError):  # will arise if custom funcs provided
                functional_dict = norm_id

            actions = functional_dict.copy()
            apply_to = actions.pop("apply_to")
            use_stat = actions.pop("use_stat")

            if apply_to(arg):
                tmp = normalized
                args = []

                if use_stat:
                    args.append(self.stats)

                for action in actions.values():
                    tmp = action(tmp, *args)

                normalized = tmp

        if hasattr(normalized, "__len__"):
            check_passes =  _CHECKS["is_shape_equal"](normalized, arg)
        else:
            check_passes = _CHECKS["is_float_like"](normalized)

        assert check_passes, "mismatch between normalized and original arg"

        return normalized


    # MAGIC FUNCTIONS
    def __call__(self, *args, mode="apply_norm"):
        if mode == "apply_norm":
            assert len(args) == 1, "'apply_norm' mode requires one arg only, got {} args".format(len(args))

            arg = args[0]
            arg = arg.reshape(np.array(arg).shape if np.array(arg).shape != () else (-1, 1))

            return self.apply_norms(arg)

        elif "calc" in mode:
            # always use the Euclidean norm because the K-NN algorithm will always use it (pyfuncs are too slow)
            assert len(args) == 2, "'calc' mode requires two args only, got {} arg(s)".format(len(args))

            a, b = args

            if "apply_norm" in mode:
                a, b = self.__call__(a), self.__call__(b)

            return self.apply_norms(np.linalg.norm(a - b))

        else:
            raise ValueError("supported modes are 'calc+{}' and 'apply_norm'")

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
    for trial_num, test in enumerate(np.random.random((10, 128, 1))):
        data = np.random.random((100, 1))

        differences = {}

        dist_metric = DistMetric("euclidean+subtract_mean", data=data)
        result = dist_metric(test)
        true_value = test - np.mean(data)
        differences[dist_metric.__repr__()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+l2_normalize", data=data)
        result = dist_metric(test)
        true_value = DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test)
        differences[dist_metric.__repr__()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+subtract_mean+l2_normalize", data=data)
        result = dist_metric(test)
        true_value = DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test - np.mean(data))
        differences[dist_metric.__repr__()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean", data=data)
        result = dist_metric(test)
        true_value = DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test) - np.mean(data)
        differences[dist_metric.__repr__()] = np.sum(true_value - result)

        second_test = np.random.random(test.shape)
        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean+sigmoid", data=data)
        result = dist_metric(test, second_test, mode="calc+apply_norms")
        true_value = DistMetric.NORMALIZATIONS["sigmoid"]["func"](
            np.linalg.norm(
                DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test) - np.mean(data) -
                (DistMetric.NORMALIZATIONS["l2_normalize"]["func"](second_test) - np.mean(data))
            )
        )
        differences[dist_metric.__repr__()] = np.sum(true_value - result)

        for metric in differences:
            if differences[metric] != 0:
                print("Error - {}: difference of {}".format(metric, differences[metric]))
        else:
            print("Test {} finished without error".format(trial_num + 1))
