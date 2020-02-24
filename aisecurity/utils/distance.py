"""

"aisecurity.utils.distance"

Distance metrics for facial recognition.

"""

import numpy as np
from scipy.spatial.distance import cosine


################################ Lambda dictionaries ###############################

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


################################ Functional dictionary constructors ###############################
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


################################ Statistics and linear algebra ###############################
def svd_whiten(x):
    # https://stackoverflow.com/questions/6574782/how-to-whiten-matrix-in-pca
    u, s, vh = np.linalg.svd(x, full_matrices=False)
    return np.dot(u, vh)


################################ DistMetric ###############################
class DistMetric:

    DISTS = {
        # "norm" describes the transformations to be applied before Euclidean distance is calculated
        # "calc" is the actual calculation of distance
        # note: composition of "norm" and np.linalg.norm will produce same K-NN ordering as passing "calc" as metric
        # into K-NN but will not necessarily output the same distance (hence the 'calc' attr of dist functional dicts)
        "euclidean": construct_dist(
            norm=lambda x: x,
            calc=lambda a, b: np.linalg.norm(a - b)
        ),
        "cosine": construct_dist(
            # norm has to be a transformation s.t. x^T x = 1 for all x in a, b
            # (https://stackoverflow.com/questions/34144632/using-cosine-distance-with-scikit-learn-kneighborsclassifier)
            norm=lambda x: svd_whiten(x),
            calc=lambda a, b: cosine(a.flatten(), b.flatten())
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
        if "+" in dist:  # ex: DistMetric("euclidean+subtract_mean+l2_normalize")
            self.dist, *self.normalizations = dist.split("+")
        else:  # ex: DistMetric("euclidean", ["subtract_mean", "l2_normalize"])
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

    def _apply_norms(self, arg):
        normalized = arg

        for norm_id in self.normalizations:
            normalized = self._apply_norm(norm_id, arg, normalized)

        if _CHECKS["is_float_like"](arg):
            check_passes = _CHECKS["is_float_like"](normalized)
        else:
            check_passes = _CHECKS["is_shape_equal"](normalized, arg)

        assert check_passes, "mismatch between normalized and original arg"

        return normalized


    # "PUBLIC" FUNCTIONS
    def apply_norms(self, *args, dist_norm=True):
        if len(args) == 1:
            arg = np.array(args[0])
            arg = arg.reshape(arg.shape if arg.shape != () else (-1, 1))

            normalized = self._apply_norms(arg)

        else:
            normalized = [self.apply_norms(arg) for arg in args]

        if dist_norm:
            normalized = self.DISTS[self.dist]["norm"](normalized)

        return np.array(normalized)

    def distance(self, a, b, apply_norms=True, ignore_norms=None):
        if ignore_norms is None:
            ignore_norms = {}

        args = [a, b]

        if apply_norms:
            # applying norms arg by arg
            for idx, arg in enumerate(args):
                if idx > len(ignore_norms) - 1:
                    args[idx] = self.apply_norms(args[idx], dist_norm=False)

                else:
                    # applying norms one by one for the 'ignore' arg
                    for norm_id in self.get_config().split("+")[1:]:  # cfg[0] is 'dist' mode
                        if norm_id not in ignore_norms[idx]:
                            args[idx] = self._apply_norm(norm_id, args[idx])

        dist = self.DISTS[self.dist]["calc"](*args)
        normalized_dist = self._apply_norms(dist)

        return normalized_dist


    # RETRIEVERS
    def get_config(self):
        return self.__str__().replace("Distance (", "").replace(")", "")


    # MAGIC FUNCTIONS
    def __str__(self):
        result = "Distance ({}".format(self.dist)
        for norm_id in self.normalizations:
            result += "+{}".format(norm_id)
        result += ")"
        return result

    def __repr__(self):
        return self.__str__()


if __name__ == "__main__":
    # TODO: rewrite as unit test
    from timeit import default_timer as timer

    for trial_num, test in enumerate(np.random.random((10, 128, 1))):
        start = timer()

        data = np.random.random((100, 1))
        second_test = np.random.random(test.shape)

        differences = {}

        dist_metric = DistMetric("cosine")

        tests = [np.random.random(test.shape) for _ in range(100)]
        norm_tests = dist_metric.apply_norms(*tests)
        dists = [(idx, np.linalg.norm(a - b)) for (idx, a), b in zip(enumerate(norm_tests[:-1]), norm_tests[1:])]
        result = np.array(list(zip(*sorted(dists, key=lambda pair: pair[1])))[0])
        true_dists = [(idx, cosine(a, b)) for (idx, a), b in zip(enumerate(tests[:-1]), tests[1:])]
        true_value = np.array(list(zip(*sorted(true_dists, key=lambda pair: pair[1])))[0])
        differences[dist_metric.get_config() + "+{calc_with_euclidean}"] = np.sum(result - true_value)

        dist_metric = DistMetric("cosine")
        result = dist_metric.distance(test, second_test)
        true_value = cosine(test.flatten(), second_test.flatten())
        differences[dist_metric.get_config() + "+{calc}"] = np.sum(true_value - result)

        dist_metric = DistMetric("cosine+subtract_mean", data=data)
        result = dist_metric.distance(test, second_test, apply_norms=True)
        true_value = cosine(test - np.mean(data), second_test - np.mean(data))
        differences[dist_metric.get_config() + "+{calc}"] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+subtract_mean", data=data)
        result = dist_metric.apply_norms(test)
        true_value = test - np.mean(data)
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+subtract_mean", data=data)
        result = dist_metric.distance(test, second_test)
        true_value = np.linalg.norm(test - np.mean(data) - (second_test - np.mean(data)))
        differences[dist_metric.get_config() + "+{calc}"] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+l2_normalize", data=data)
        result = dist_metric.apply_norms(test)
        true_value = DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test)
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+subtract_mean+l2_normalize", data=data)
        result = dist_metric.apply_norms(test)
        true_value = DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test - np.mean(data))
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean", data=data)
        result = dist_metric.apply_norms(test)
        true_value = DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test) - np.mean(data)
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean", data=data)
        result = dist_metric.apply_norms(test, second_test)
        true_value = np.array([
            DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test) - np.mean(data),
            DistMetric.NORMALIZATIONS["l2_normalize"]["func"](second_test) - np.mean(data)
        ])
        differences[dist_metric.get_config() + "+{recursive}"] = np.sum(np.sum(true_value) - np.sum(result))

        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean", data=data)
        result = dist_metric.distance(test, second_test)
        true_value = np.linalg.norm(
            DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test) - np.mean(data)
            - (DistMetric.NORMALIZATIONS["l2_normalize"]["func"](second_test) - np.mean(data))
        )
        differences[dist_metric.get_config() + "+{calc}"] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean+sigmoid", data=data)
        result = dist_metric.distance(test, second_test)
        true_value = DistMetric.NORMALIZATIONS["sigmoid"]["func"](
            np.linalg.norm(
                (DistMetric.NORMALIZATIONS["l2_normalize"]["func"](test) - np.mean(data)) -
                (DistMetric.NORMALIZATIONS["l2_normalize"]["func"](second_test) - np.mean(data))
            )
        )
        differences[dist_metric.get_config()] = np.sum(true_value - result)

        dist_metric = DistMetric("euclidean+l2_normalize+subtract_mean", data=data)
        result = dist_metric.distance(
            test, second_test, ignore_norms=["euclidean+l2_normalize+subtract_mean", "euclidean+l2_normalize"]
        )
        true_value = np.linalg.norm(test - (second_test - np.mean(data)))
        differences[dist_metric.get_config() + "+ignore"] = np.sum(true_value - result)

        for metric in differences:
            if differences[metric] != 0.:
                print("Error - {}: difference of {}".format(metric, np.round(differences[metric], 5)))

        print("Test {} finished in {}s".format(trial_num + 1, round(timer() - start, 5)))
