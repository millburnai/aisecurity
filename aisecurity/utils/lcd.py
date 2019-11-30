"""

"aisecurity.lcd"

LCD utils.

"""

import concurrent.futures
import functools
import time
import warnings

# CONSTANTS
COLORS = None


# AUTOINIT
try:
    import Jetson.GPIO as GPIO
    GPIO.cleanup()
    GPIO.setmode(GPIO.BCM)

    try:
        COLORS = [18, 23]
        for color in COLORS:
            GPIO.setup(color, GPIO.OUT)
    except RuntimeError:
        warnings.warn("Improper wire configuration")

except ImportError:
    warnings.warn("Jetson.GPIO not found")
    GPIO = None  # so that there are no import errors, even if Jetson.GPIO isn't found

try:
    from adafruit_character_lcd.character_lcd_i2c import Character_LCD_I2C as character_lcd
    import board
    import busio
except NotImplementedError:
    warnings.warn("LCD not supported")
    board, busio, Character_LCD_I2C = None, None, None


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
        self.flush(previous_msg=msg)

    def _update(self, percent, previous_msg=None):
        if not self.is_on:
            self.is_on = True

        self.progress += percent

        bar_length = self.length - 2  # compensate for [] at beginning and end
        done = self.marker * round(self.progress * bar_length)
        left = " " * (bar_length - len(done))

        if previous_msg:
            self.lcd.message = "{}\n[{}{}]".format(previous_msg, done, left)
        else:
            self.lcd.message = "[{}{}]".format(done, left)

        self.is_on = True

        if self.progress >= 1. or self.progress < 0.:
            self.progress = 0.

    def update(self, amt=1, previous_msg=None):
        self._update(amt / self.total, previous_msg)

    def flush(self, previous_msg=None):
        if previous_msg:
            self.lcd.message = "{}\n[{}]".format(previous_msg, " " * bar_length)
        else:
            self.lcd.message = "[{}]".format(" " * bar_length)


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
