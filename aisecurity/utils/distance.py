"""

"aisecurity.utils.distance"

Distance metrics for facial recognition.

"""

import warnings

import numpy as np


# LAMBDA DICTS
_FORMAT = {
    "apply_to": lambda x: callable(x),
    "use_stat": lambda x: isinstance(x, bool),
    "func": lambda x: callable(x)
}

_CHECKS = {
    "is_vector": lambda x: x.ndim <= 2 and (1 in x.shape or x.ndim == 1),
    "is_float_like": lambda x: isinstance(x, float) or isinstance(x, np.float),
    "is_shape_equal": lambda a, b: np.array(a).shape == np.array(b).shape
}


# FUNCTIONAL DICT CONSTRUCTORS
def construct_dist(func):
    def test(test_case):
        res = func(*test_case)
        assert isinstance(res, tuple) and len(res) == 2, "func must take two arrays as input and return two arrays"

    test_case = np.random.random((2, ))
    test(test_case)

    test_case = np.random.random((2, 1))
    test(test_case)

    return func


def construct_norm(**kwargs):
    checked = kwargs.copy()

    for value, test in _FORMAT.items():
        assert value in kwargs and test(kwargs[value]), "{} is missing or failed test".format(value)
        kwargs.pop(value)
    assert all(callable(extra) for extra in kwargs)

    return checked


# DISTMETRIC
class DistMetric:

    DISTS = {
        # dists are the transformations to be applied before Euclidean distance is calculated
        # ex: for "euclidean", no transformation is applied --> lambda a, b: (a, b)
        "euclidean": construct_dist(
            lambda a, b: (a, b)
        ),
        "cosine": construct_dist(
            lambda a, b: (
                a - np.mean(a) / np.maximum(np.std(a), 1e-6),
                b - np.mean(b) / np.maximum(np.std(b), 1e-6)
            )  # not so sure this is right... has to be a transformation s.t. x^T x = 1 for all x in a, b
        )
    }

    NORMALIZATIONS = {
        # normalizations are applied before dist lambdas
        # ex: for "cosine+l2_normalize", L2 normalization is applied, then cosine normalization, then Euclidean distance
        "subtract_mean": construct_norm(
            apply_to= _CHECKS["is_vector"],
            use_stat=True,
            func=lambda x, stats: x - stats["mean"]
        ),

        "l2_normalize": construct_norm(
            apply_to=_CHECKS["is_vector"],
            use_stat=False,
            func=lambda x: x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), 1e-6))
        ),
        "sigmoid": construct_norm(
            apply_to=_CHECKS["is_float_like"],
            use_stat=False,
            func=lambda x: 1. / (1. + np.exp(-x))
        )
    }


    # INITS
    def __init__(self, dist, normalizations=None, data=None, **kwargs):
        if "+" in dist:
            # ex: DistMetric("euclidean+subtract_mean+l2_normalize")
            self.dist, *self.normalizations = dist.split("+")
        else:
            # ex: DistMetric("euclidean", ["subtract_mean", "l2_normalize"])
            self.dist = dist
            self.normalizations = normalizations if normalizations else []

        assert self.dist in self.DISTS, "supported dists are {}".format(list(self.DISTS.keys()))
        assert self.normalizations is None or all(norm in self.NORMALIZATIONS for norm in self.normalizations), \
            "supported normalizations are {}".format(list(self.NORMALIZATIONS.keys()))

        # data setting
        if any(self.NORMALIZATIONS[norm_id]["use_stat"] for norm_id in self.normalizations):
            assert data is not None, "data must be provided for normalizations that use data statistics"

            self.stats = {
                "mean": np.mean(data, **kwargs),
                "std": np.std(data, **kwargs)
            }

        self.data = data


    # HELPER FUNCTIONS
    def apply_norms(self, arg):
        normalized = arg

        for norm_id in self.normalizations:
            functional_dict = self.NORMALIZATIONS[norm_id]

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
            assert len(args) == 1, "'apply_norm' dist requires one arg only, got {} args".format(len(args))

            arg = args[0]
            arg = arg.reshape(np.array(arg).shape if np.array(arg).shape != () else (-1, 1))

            return self.apply_norms(arg)

        elif "calc" in mode:
            # always use the Euclidean norm because the K-NN algorithm will always use it (pyfuncs are too slow)
            assert len(args) == 2, "'calc' dist requires two args only, got {} arg(s)".format(len(args))

            a, b = args

            if "apply_norm" in mode:
                a, b = self.__call__(a), self.__call__(b)

            return self.apply_norms(np.linalg.norm(a - b))

        else:
            raise ValueError("supported modes are 'calc+{}' and 'apply_norm'")

    def __str__(self):
        result = "Distance ("
        result += self.dist
        for norm_id in self.normalizations:
            result += "+{}".format(norm_id)
        result += ")"
        return result

    def __repr__(self):
        return self.__str__()


    # RETRIEVERS
    def get_config(self):
        return self.__str__().replace("Distance (", "").replace(")", "")


if __name__ == "__main__":
    for trial_num, test in enumerate(np.random.random((10, 128, 1))):
        data = np.random.random((100, 1))

        differences = {}

        dist_metric = DistMetric("euclidean+subtract_mean", data=data)
        result = dist_metric(test)
        true_value = test - np.mean(data)
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+l2_normalize", data=data)
        result = dist_metric(test)
        true_value = DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test)
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+subtract_mean+l2_normalize", data=data)
        result = dist_metric(test)
        true_value = DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test - np.mean(data))
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean", data=data)
        result = dist_metric(test)
        true_value = DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test) - np.mean(data)
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        second_test = np.random.random(test.shape)
        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean+sigmoid", data=data)
        result = dist_metric(test, second_test, mode="calc+apply_norms")
        true_value = DistMetric.NORMALIZATIONS["sigmoid"]["func"](
            np.linalg.norm(
                DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test) - np.mean(data) -
                (DistMetric.NORMALIZATIONS["l2_normalize"]["func"](second_test) - np.mean(data))
            )
        )
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        for metric in differences:
            if differences[metric] != 0:
                print("Error - {}: difference of {}".format(metric, differences[metric]))
        else:
            print("Test {} finished without error".format(trial_num + 1))
