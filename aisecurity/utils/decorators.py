"""

"aisecurity.utils.decorators"

Miscellaneous tools for time and event handling.

"""

import functools
from timeit import default_timer as timer
import warnings


# DECORATORS
def print_time(message="Time elapsed"):
    def _timer(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            start = timer()
            result = func(*args, **kwargs)
            print("{}: {}s".format(message, round(timer() - start, 4)))
            return result

        return _func

    return _timer


def check_fail(threshold=3):
    def _check_fail(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            failures = 0
            while failures < threshold:
                if func(*args, **kwargs):
                    return True
                failures += 1
            return False

        return _func

    return _check_fail


def in_dev(message="currently in development; do not use in production"):
    def _in_dev(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            result = func(*args, **kwargs)
            warnings.warn(message)
            return result

        return _func

    return _in_dev
