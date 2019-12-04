"""

"aisecurity.lcd"

LCD utils.

"""

import concurrent.futures
import functools
import time
import warnings

import requests

from aisecurity.database.log import THRESHOLDS
from aisecurity.utils.paths import CONFIG


# INITS

# AUTOINIT
COLORS = None
LCD, PROGRESS_BAR = None, None

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

try:
    from adafruit_character_lcd.character_lcd_i2c import Character_LCD_I2C as character_lcd
    import board
    import busio
except (NotImplementedError, ModuleNotFoundError):
    warnings.warn("LCD not supported")

try:
    i2c = busio.I2C(board.SCL, board.SDA)
    i2c.scan()
except (RuntimeError, NameError) as error:
    if isinstance(error, RuntimeError):
        raise RuntimeError("Wire configuration incorrect")
    elif isinstance(error, NameError):
        warnings.warn("i2c not supported")


# LCD INIT
def init():
    global LCD, PROGRESS_BAR
    LCD = character_lcd(i2c, 16, 2, backlight_inverted=False)
    LCD.message = "Loading...\n[Initializing]"
    PROGRESS_BAR = LCDProgressBar(total=THRESHOLDS["num_recognized"], lcd=LCD)


# ADD LCD DISPLAY
def add_lcd_display(best_match, use_server):
    global LCD, PROGRESS_BAR

    def green_display():
        GPIO.output(COLORS[0], GPIO.HIGH)
        GPIO.output(COLORS[1], GPIO.HIGH)

    def red_display():
        GPIO.output(COLORS[0], GPIO.LOW)
        GPIO.output(COLORS[1], GPIO.LOW)

    def violet_display():
        GPIO.output(COLORS[0], GPIO.LOW)
        GPIO.output(COLORS[1], GPIO.HIGH)

    def white_display():
        GPIO.output(COLORS[0], GPIO.HIGH)
        GPIO.output(COLORS[1], GPIO.LOW)

    if LCD is None or PROGRESS_BAR is None:
        return -1

    LCD.clear()
    PROGRESS_BAR.display_off()

    best_match = best_match.replace("_", " ").title()

    if use_server:
        request = requests.get(CONFIG["server_address"])
        data = request.json()

        if data["accept"]:
            LCD.message = "ID Accepted\n{}".format(best_match)
            green_display()
        elif "visitor" in best_match.lower():
            LCD.message = "Welcome to MHS,\n{}".format(best_match)
            violet_display()
        else:
            LCD.message = "No Senior Priv\n{}".format(best_match)
            red_display()

    else:
        if "visitor" in best_match.lower():
            LCD.message = "Welcome to MHS,\n{}".format(best_match)
            violet_display()
        else:
            LCD.message = "[Server Error]\n{}".format(best_match)
            green_display()


# LCD PROGRESS BAR
class LCDProgressBar(object):

    def __init__(self, total, lcd, length=16, marker="#"):
        self.total = total
        self.lcd = lcd
        self.bar_length = length - 2  # compensate for [] at beginning and end
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

        done = self.marker * round(self.progress * self.bar_length)
        left = " " * (self.bar_length - len(done))

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
            self.lcd.message = "{}\n[{}]".format(previous_msg, " " * self.bar_length)
        else:
            self.lcd.message = "[{}]".format(" " * self.bar_length)




# PROGRESS BAR DECORATOR
def progress_bar(lcd, expected_time, msg=None, marker="#", sess=None):

    def _progress_bar(func):

        def timed_progress_bar(future, expected_time, marker="#", previous_msg=None):
            """
            Display progress bar for expected_time seconds.
            Complete early if future completes.
            Wait for future if it doesn't complete in expected_time.
            """
            pbar = LCDProgressBar(total=expected_time, lcd=lcd, marker=marker)

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
