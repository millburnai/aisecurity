"""

"aisecurity.utils.distance"

Distance metrics for facial recognition.

"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


# LAMBDA DICTS

# OFTEN-USED CHECKS
_CHECKS = {
    "is_vector": lambda x: x.ndim <= 2 and (1 in x.shape or x.ndim == 1),
    "is_float_like": lambda x: isinstance(x, float) or isinstance(x, np.float),
    "is_shape_equal": lambda a, b: np.array(a).shape == np.array(b).shape
}

# NORM TESTS
_NORM_FORMAT = {
    "apply_to": lambda x: callable(x),
    "use_stat": lambda x: isinstance(x, bool),
    "func": lambda x: callable(x)
}

# DIST TESTS
_DIST_FORMAT = {
    "norm": lambda x: callable(x),
    "calc": lambda x: callable(x)
}


# FUNCTIONAL DICT CONSTRUCTORS
def _test_format(format, **kwargs):
    checked = kwargs.copy()

    for value, test in format.items():
        assert value in kwargs and test(kwargs[value]), "{} is missing or failed test".format(value)
        kwargs.pop(value)
    assert all(callable(extra) for extra in kwargs)

    return checked


def construct_dist(**kwargs):
    return _test_format(_DIST_FORMAT, **kwargs)


def construct_norm(**kwargs):
    return _test_format(_NORM_FORMAT, **kwargs)


# DISTMETRIC
class DistMetric:

    DISTS = {
        # "norm" describes the transformations to be applied before Euclidean distance is calculated
        # "calc" is the actual calculation of distance
        # note: composition of "norm" and np.linalg.norm will produce same K-NN ordering as passing "calc" as metric
        # into K-NN but will not necessarily output the same distance
        "euclidean": construct_dist(
            norm=lambda x: x,
            calc=lambda a, b: np.linalg.norm(a - b)
        ),
        "cosine": construct_dist(
            # definitely not right... has to be a transformation s.t. x^T x = 1 for all x in a, b
            norm=lambda x: x / np.sqrt(np.maximum(np.sum(np.square(x), axis=-1, keepdims=True), 1e-6)),
            calc=lambda a, b: cosine_similarity(a.reshape(1, -1), b.reshape(1, -1))
        )
    }

    NORMALIZATIONS = {
        # normalizations are applied before dist lambdas
        # ex: for "cosine+l2_normalize", L2 normalization is applied, then cosine normalization, then Euclidean distance
        # "apply_to": what type of objects to apply to (callable)
        # "use_stat": use statistics like mean and std (requires data to be supplied
        # "func": norm func
        "subtract_mean": construct_norm(
            apply_to=_CHECKS["is_vector"],
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
    def _apply_norm(self, norm_id, arg, normalized=None):
        if normalized is None:
            normalized = arg

        actions = self.NORMALIZATIONS[norm_id].copy()
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

        return normalized

    def apply_norms(self, arg):
        normalized = arg

        for norm_id in self.normalizations:
            normalized = self._apply_norm(norm_id, arg, normalized)

        if _CHECKS["is_float_like"](arg):
            check_passes = _CHECKS["is_float_like"](normalized)
        else:
            check_passes = _CHECKS["is_shape_equal"](normalized, arg)

        assert check_passes, "mismatch between normalized and original arg"

        return normalized


    # MAGIC FUNCTIONS
    def __call__(self, *args, mode="norm", ignore=None):
        if mode == "norm":
            if len(args) == 1:
                arg = args[0]
                arg = np.array(arg).reshape(np.array(arg).shape if np.array(arg).shape != () else (-1, 1))

                normalized = self.apply_norms(arg)

            else:
                normalized = [self.__call__(arg) for arg in args]

            dist_normalized = np.array([self.DISTS[self.dist]["norm"](arr) for arr in normalized])
            return dist_normalized

        elif "calc" in mode:
            assert len(args) == 2, "'calc' requires two args only, got {} arg(s)".format(len(args))

            if "norm" in mode:
                args = list(args)

                if ignore is None:
                    ignore = {}

                # applying norm list arg by arg
                for idx, arg in enumerate(args):
                    if idx not in ignore.keys():
                        args[idx] = self.__call__(args[idx])

                    else:
                        # applying norms one by one for the 'ignore' arg
                        for norm_id in self.get_config().split("+")[1:]:  # cfg[0] is 'dist' mode
                            if norm_id not in ignore[idx]:
                                args[idx] = self._apply_norm(norm_id, args[idx])

            dist = self.DISTS[self.dist]["calc"](*args)
            normalized_dist = self.apply_norms(dist)

            return normalized_dist

        else:
            raise ValueError("supported modes are 'calc', 'norm', and 'calc+norm'")

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
    from timeit import default_timer as timer

    for trial_num, test in enumerate(np.random.random((10, 128, 1))):
        start = timer()

        data = np.random.random((100, 1))
        second_test = np.random.random(test.shape)

        differences = {}

        dist_metric = DistMetric("cosine")
        norm_test, norm_second_test = dist_metric(test.reshape(-1, 1), second_test.reshape(1, -1), mode="norm")
        print(DistMetric.NORMALIZATIONS["l2_normalize"]["func"](second_test) - norm_second_test)
        result = DistMetric("euclidean")(norm_test, norm_second_test, mode="calc")
        true_value = cosine_similarity(
            test.reshape(1, -1),
            second_test.reshape(1, -1)
        )
        differences[dist_metric.get_config() + "+calc_with_euclidean"] = np.sum(true_value - result)

        dist_metric = DistMetric("cosine+subtract_mean", data=data)
        result = dist_metric(test.reshape(-1, 1), second_test.reshape(1, -1), mode="calc+norm")
        true_value = cosine_similarity(
            (test - np.mean(data)).reshape(1, -1),
            (second_test - np.mean(data)).reshape(1, -1)
        )
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+subtract_mean", data=data)
        result = dist_metric(test)
        true_value = test - np.mean(data)
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+subtract_mean", data=data)
        result = dist_metric(test, second_test, mode="calc+norm")
        true_value = np.linalg.norm(test - np.mean(data) - (second_test - np.mean(data)))
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

        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean", data=data)
        result = dist_metric(test, second_test)
        true_value = np.array([
            DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test) - np.mean(data),
            DistMetric.NORMALIZATIONS["l2_normalize"]["func"](second_test) - np.mean(data)
        ])
        differences[dist_metric.get_config() + "+{recursive}"] = np.sum(np.sum(true_value) - np.sum(result))


        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean+sigmoid", data=data)
        result = dist_metric(test, second_test, mode="calc+norm")
        true_value = DistMetric.NORMALIZATIONS["sigmoid"]["func"](
            np.linalg.norm(
                (DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test) - np.mean(data)) -
                (DistMetric.NORMALIZATIONS["l2_normalize"]["func"](second_test) - np.mean(data))
            )
        )
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean", data=data)
        result = dist_metric(test, second_test, mode="calc+norm",
                             ignore={0: "euclidean+l2_normalize+subtract_mean", 1: "euclidean+l2_normalize"})
        true_value = np.linalg.norm(test - (second_test - np.mean(data)))
        differences[dist_metric.get_config() + "+ignore"] = np.sum(true_value - result)

        for metric in differences:
            if differences[metric] != 0.:
                print("Error - {}: difference of {}".format(metric, np.round(differences[metric], 5)))
            else:
                print("Test {} finished without error ({}s)".format(trial_num + 1, round(timer() - start, 5)))
