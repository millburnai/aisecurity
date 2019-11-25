"""

"aisecurity.utils.misc"

Miscellaneous tools.

"""

import concurrent.futures
import functools
import os
import sys
import time

from keras import backend as K


# PRINT HANDLING
class HidePrints(object):

    def __enter__(self):
        self.to_show = sys.stdout
        sys.stdout = open(os.devnull, "w")

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self.to_show


# TIMER
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


# LCD PROGRESS BAR
class LCDProgressBar(object):

    def __init__(self, total, lcd, length=16, marker="#"):
        self.total = total
        self.lcd = lcd
        self.length = 16
        self.marker = marker
        self.progress = 0

        self.is_on = False

    def display_off(self, msg=""):
        self.lcd.message = msg
        self.is_on = False

    def _update(self, percent, previous_msg=None):
        if not self.is_on:
            self.is_on = True

        self.progress += percent
        if self.progress > 1.:
            self.progress = 1.
        elif self.progress < 0.:
            self.progress = 0.

        bar_length = self.length - 2  # compensate for [] at beginning and end
        done = self.marker * round(self.progress * bar_length)
        left = " " * (bar_length - len(done))

        if previous_msg:
            self.lcd.message = "{}\n[{}{}]".format(previous_msg, done, left)
        else:
            self.lcd.message = "[{}{}]".format(done, left)

        self.is_on = True


    def update(self, amt=1, previous_msg=None):
        self._update(amt / self.total, previous_msg)


# PROGRESS BAR DECORATOR
def progress_bar(lcd, expected_time, msg=None, marker="#", sess=None):

    def _progress_bar(func):

        def timed_progress_bar(future, expected_time, marker="#", previous_msg=None):
            """
            Display progress bar for expected_time seconds.
            Complete early if future completes.
            Wait for future if it doesn't complete in expected_time.
            """
            pbar = LCDProgressBar(total=expected_time, lcd=lcd, marker=marker, previous_msg=previous_msg)

            for sec in range(expected_time - 1):
                if future.done():
                    pbar.update(expected_time - sec, previous_msg=previous_msg)
                    return
                else:
                    time.sleep(1)
                    pbar.update(previous_msg=previous_msg)
                # if the future still hasn't completed, wait for it.
            future.result()
            pbar.update(previous_msg=previous_msg)

        @functools.wraps(func)
        def _func(*args, **kwargs):
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                if sess is not None:
                    K.set_session(sess)
                future = pool.submit(func, *args, **kwargs)
                timed_progress_bar(future, expected_time, marker, msg)

            return future.result()

        return _func

    return _progress_bar
