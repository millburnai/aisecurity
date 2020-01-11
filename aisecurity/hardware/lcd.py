"""

"aisecurity.hardware.lcd"

LCD utils.

"""

import concurrent.futures
import functools
import time
import warnings

from keras import backend as K
from termcolor import cprint
import requests

from aisecurity.database.log import THRESHOLDS
from aisecurity.utils.paths import CONFIG


################################ INITS ################################

# AUTOINIT
COLORS = None
LCD_DEVICE, PROGRESS_BAR, GPIO = None, None, None

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
except (NotImplementedError, ModuleNotFoundError, ValueError): #ValueError- a different mode has already been set
    warnings.warn("LCD not found")

try:
    i2c = busio.I2C(board.SCL, board.SDA)
    i2c.scan()
except (RuntimeError, NameError) as error:
    if isinstance(error, RuntimeError):
        raise RuntimeError("Wire configuration incorrect")
    elif isinstance(error, NameError):
        warnings.warn("i2c not found")


# LCD INIT
def init():
    global LCD_DEVICE, PROGRESS_BAR

    LCD_DEVICE = LCD()
    LCD_DEVICE.set_message("Loading...\n[Initializing]")

    PROGRESS_BAR = LCDProgressBar(total=THRESHOLDS["num_recognized"], lcd=LCD_DEVICE)


################################ CLASSES ################################

# LCD WRAPPER CLASS (WITH DEV SUPPORT)
class LCD:

    def __init__(self, mode="physical"):
        assert mode == "physical" or mode == "dev", "supported modes are physical (physical LCD) and dev (testing)"
        self.__lcd = None

        try:
            self.__lcd = character_lcd(i2c, 16, 2, backlight_inverted=False)
            self.mode = "physical"
            assert self.mode == mode  # making sure that physical doesn't override user choice
        except (NameError, AssertionError):
            self.__lcd = LCDSimulation()
            self.mode = "dev"

            if self.mode != mode:
                warnings.warn("physical lcd mode requested but only dev lcd available")
            warnings.warn("dev lcd does not support colors")


    # FUNCTIONALITY
    def set_message(self, message, backlight="white"):

        # TODO: get correct colors
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

        display_dict = {
            "green": green_display,
            "red": red_display,
            "violet": violet_display,
            "white": white_display
        }

        assert backlight in display_dict.keys(), "backlight must be green, red, violet, or white"

        self.__lcd.message = message

        if self.mode == "physical":
            # set backlight if supported
            display_dict[backlight]()


    # RETRIEVERS
    @property
    def lcd(self):
        return self.__lcd

    @property
    def message(self):
        return self.__lcd.message


# SIMULATION SUPPORT FOR DEV
class LCDSimulation:

    def __init__(self):
        self.message = None

    def __setattr__(self, key, value):
        if key == "message" and value:
            cprint(value, attrs=["bold"])
        super(LCDSimulation, self).__setattr__(key, value)


# LCD PROGRESS BAR
class LCDProgressBar:

    def __init__(self, total, lcd, length=16, marker="#"):
        self.total = total
        self.lcd = lcd
        self.bar_length = length - 2  # compensate for [] at beginning and end
        self.marker = marker
        self.progress = 0

        self.is_on = False

    def display_off(self, msg=""):
        self.lcd.set_message(msg)
        self.is_on = False
        self.flush(previous_msg=msg)

    def _update(self, percent, previous_msg=None):
        if not self.is_on:
            self.is_on = True

        self.progress += percent

        done = self.marker * round(self.progress * self.bar_length)
        left = " " * (self.bar_length - len(done))

        if previous_msg:
            self.lcd.set_message("{}\n[{}{}]".format(previous_msg, done, left))
        else:
            self.lcd.set_message("[{}{}]".format(done, left))

        self.is_on = True

        if self.progress >= 1. or self.progress < 0.:
            self.progress = 0.

    def update(self, amt=1, previous_msg=None):
        self._update(amt / self.total, previous_msg)

    def flush(self, previous_msg=None):
        if previous_msg:
            self.lcd.set_message("{}\n[{}]".format(previous_msg, " " * self.bar_length))
        else:
            self.lcd.set_message("[{}]".format(" " * self.bar_length))


################################ FUNCTIONS AND DECORATORS ################################

# ADD DISPLAY
def add_lcd_display(best_match, use_server):
    global LCD_DEVICE, PROGRESS_BAR

    LCD_DEVICE.clear()
    PROGRESS_BAR.display_off()

    best_match = best_match.replace("_", " ").title()

    if use_server:
        request = requests.get(CONFIG["server_address"])
        data = request.json()

        if data["accept"]:
            LCD_DEVICE.set_message("ID Accepted\n{}".format(best_match), color="green")
        elif "visitor" in best_match.lower():
            LCD_DEVICE.set_message("Welcome to MHS,\n{}".format(best_match), color="violet")
        else:
            LCD_DEVICE.set_message("No Senior Priv\n{}".format(best_match), color="red")

    else:
        if "visitor" in best_match.lower():
            LCD_DEVICE.set_message("Welcome to MHS,\n{}".format(best_match), color="violet")
        else:
            LCD_DEVICE.set_message("[Server Error]\n{}".format(best_match), color="green")


# PROGRESS BAR DECORATOR
def progress_bar(lcd, expected_time, msg=None, marker="#", sess=None):

    def _progress_bar(func):

        def timed_progress_bar(future, expected_time, marker="#", previous_msg=None):
            # complete early if future completes; wait for future if it doesn't complete in expected_time
            pbar = LCDProgressBar(total=expected_time, lcd=lcd, marker=marker)

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
                if sess:
                    K.set_session(sess)
                future = pool.submit(func, *args, **kwargs)
                timed_progress_bar(future, expected_time, marker, msg)

            return future.result()

        return _func

    return _progress_bar
