"""

"aisecurity.utils.events"

Miscellaneous tools for time and event handling.

"""

import asyncio
import functools
import time
import warnings


# DECORATORS
def timer(message="Time elapsed"):
    def _timer(func):
        @functools.wraps(func)
        def _func(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            print("{}: {}s".format(message, round(time.time() - start, 3)))
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


# ASYNC
def run_async_method(func, *args, **kwargs):
    async def async_helper(func, *args, **kwargs):
        await func(*args, **kwargs)

    loop = asyncio.new_event_loop()
    task = loop.create_task(async_helper(func, *args, **kwargs))
    loop.run_until_complete(task)
