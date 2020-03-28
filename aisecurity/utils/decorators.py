"""

"aisecurity.utils.events"

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


def in_dev(message="currently in development; do not use in production"):
    def _in_dev(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            result = func(*args, **kwargs)
            warnings.warn(message)
            return result

        return _func

    return _in_dev
