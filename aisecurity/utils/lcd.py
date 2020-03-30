"""

"aisecurity.hardware.lcd"

LCD utils.

"""
import concurrent.futures
import functools
import time
from timeit import default_timer as timer
import warnings

from termcolor import cprint

from aisecurity.db import log
from aisecurity.utils import connection


################################ Inits ################################

# GLOBALS
LCD_DEVICE, PROGRESS_BAR = None, None


# LCD INIT
def init():
    global LCD_DEVICE, PROGRESS_BAR

    LCD_DEVICE = LCD(mode="sim")
    LCD_DEVICE.set_message("Loading...\n[ Initializing ]")

    PROGRESS_BAR = LCDProgressBar(lcd=LCD_DEVICE, total=log.THRESHOLDS["num_recognized"])


################################ Classes ################################

# LCD WRAPPER CLASS (WITH DEV SUPPORT)
class LCD:

    def __init__(self, mode="pi"):
        assert mode in ("pi", "sim"), "supported modes are physical (physical LCD) and dev (testing)"

        try:
            assert connection.SOCKET, "connection.SOCKET must be initialized by using connection.init()"
            self.mode = "pi"
            assert self.mode == mode  # making sure that physical doesn't override user choice\

        except (ValueError, NameError, AssertionError):
            self.mode = "sim"
            if self.mode != mode:
                warnings.warn("pi lcd mode requested but only simulation lcd available")


    # FUNCTIONALITY
    def set_message(self, message):

        if self.mode == "pi":
            connection.send({"LCD":message})
        elif self.mode == "sim":
            cprint(message, attrs=["bold"])


# LCD PROGRESS BAR
class LCDProgressBar:

    def __init__(self, lcd, total, length=16, marker="#"):
        self.lcd = lcd
        self.total = total
        self.bar_length = length - 2  # compensate for [] at beginning and end
        self.marker = marker
        self.progress = 0.
        self.empty = " " * self.bar_length

    def reset(self, previous_msg=None):
        self.progress = 0.
        if previous_msg:
            self.lcd.set_message("{}\n[{}]".format(previous_msg, " " * self.bar_length))

    def update(self, previous_msg=None):
        self.progress += 1/self.total

        done = (self.marker * round(self.progress * self.bar_length) + self.empty)[:self.bar_length]

        self.lcd.set_message("{}\n[{}]".format(previous_msg, done))

        if bool(self.progress>=1):
            self.progress = 0.


    # PROGRESS BAR DECORATOR
    @classmethod
    def progress_bar(cls, lcd, expected_time, msg=None, marker="#"):
        # https://stackoverflow.com/questions/59013308/python-progress-bar-for-non-loop-function

        def _progress_bar(func):

            def timed_progress_bar(future, expected_time, marker="#", previous_msg=None):
                # complete early if future completes; wait for future if it doesn't complete in expected_time
                pbar = cls(total=expected_time, lcd=lcd, marker=marker)

                for sec in range(expected_time - 1):
                    if future.done():
                        pbar.update(expected_time - sec, previous_msg=previous_msg)
                        return
                    else:
                        time.sleep(1)
                        pbar.update(previous_msg=previous_msg)

                future.result()
                pbar.update(previous_msg=previous_msg)

            @functools.wraps(func)
            def _func(*args, **kwargs):
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(func, *args, **kwargs)
                    timed_progress_bar(future, expected_time, marker, msg)

                return future.result()

            return _func

        return _progress_bar


################################ Functions ################################

# PERIODIC LCD CLEAR
def check_clear():
    lcd_clear = log.THRESHOLDS["num_recognized"] / log.THRESHOLDS["missed_frames"]
    if log.LAST_LOGGED - timer() > lcd_clear or log.UNK_LAST_LOGGED - timer() > lcd_clear:
        global PROGRESS_BAR
        PROGRESS_BAR.reset()


# PBAR UPDATE
def update_progress(update_recognized):
    global PROGRESS_BAR

    PROGRESS_BAR.update(previous_msg="Recognizing...")
