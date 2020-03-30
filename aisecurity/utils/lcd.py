"""

"aisecurity.hardware.lcd"

LCD utils.

"""

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
    LCD_DEVICE.set_message("Loading...\n[Initializing]")

    PROGRESS_BAR = LCDProgressBar(lcd=LCD_DEVICE, total=log.THRESHOLDS["num_recognized"])


################################ Classes ################################

# LCD WRAPPER CLASS (WITH DEV SUPPORT)
class LCD:

    def __init__(self, mode="pi"):
        assert mode in ("pi", "sim"), "supported modes are physical (physical LCD) and dev (testing)"
        self.message = None

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
        self.message = message

        if self.mode == "pi":
            connection.send(lcd_message=self.message)
        elif self.mode == "sim":
            cprint(self.message, attrs=["bold"])

    def clear(self):
        self.set_message("<Message cleared>")


# LCD PROGRESS BAR
class LCDProgressBar:

    def __init__(self, lcd, total, length=16, marker="#"):
        self.lcd = lcd
        self.total = total
        self.bar_length = length - 2  # compensate for [] at beginning and end
        self.marker = marker
        self.progress = 0

    def reset(self, previous_msg=None):
        self.progress = 0.
        if previous_msg:
            self.lcd.set_message("{}\n[{}]".format(previous_msg, " " * self.bar_length))

    def _update(self, percent, previous_msg=None):
        self.progress += percent

        reset = False
        if self.progress >= 1.:
            self.progress = 1.
            reset = True

        done = self.marker * round(self.progress * self.bar_length)
        left = " " * (self.bar_length - len(done))

        if previous_msg:
            self.lcd.set_message("{}\n[{}{}]".format(previous_msg, done, left))
        else:
            self.lcd.set_message("[{}{}]".format(done, left))

        if reset:
            self.reset(previous_msg=previous_msg)

    def update(self, amt=1, previous_msg=None):
        if amt > self.total:
            amt = self.total
        elif amt < 0:
            raise ValueError("amt cannot be negative")
        self._update(amt / self.total, previous_msg)


################################ Functions ################################

# RESET
def reset():
    global LCD_DEVICE, PROGRESS_BAR

    LCD_DEVICE.clear()
    PROGRESS_BAR.reset()


# PERIODIC LCD CLEAR
def check_clear():
    lcd_clear = log.THRESHOLDS["num_recognized"] / log.THRESHOLDS["missed_frames"]
    if log.LAST_LOGGED - timer() > lcd_clear or log.UNK_LAST_LOGGED - timer() > lcd_clear:
        reset()


# PBAR UPDATE
def update_progress(update_recognized):
    global PROGRESS_BAR

    if update_recognized:
        PROGRESS_BAR.update(amt=PROGRESS_BAR.total, previous_msg="Recognizing...")
    elif not 1. / PROGRESS_BAR.total + PROGRESS_BAR.progress >= 1.:
        PROGRESS_BAR.update(previous_msg="Recognizing...")
