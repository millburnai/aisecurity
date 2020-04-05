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
PROGRESS_BAR = None


# LCD INIT
def init():
    global PROGRESS_BAR

    PROGRESS_BAR = LCDProgressBar(mode="pi" if connection.SOCKET else "sim", total=log.THRESHOLDS["num_recognized"])
    PROGRESS_BAR.set_message("Loading...\n[ Initializing ]")


################################ Classes ################################

# LCD PROGRESS BAR
class LCDProgressBar:

    def __init__(self, mode, total, length=16, marker="#"):
        assert mode in ("pi", "sim"), "supported modes are physical (physical LCD) and dev (testing)"

        try:
            assert connection.SOCKET, "connection.SOCKET must be initialized by using connection.init()"
            self.mode = "pi"
            assert self.mode == mode  # making sure that physical doesn't override user choice\

        except (ValueError, NameError, AssertionError):
            self.mode = "sim"
            if self.mode != mode:
                warnings.warn("pi lcd mode requested but only simulation lcd available")

        self.total = total
        self.bar_length = length - 2  # compensate for [] at beginning and end
        self.marker = marker
        self.progress = 0.
        self.blank = " " * self.bar_length

    def set_message(self, message):
        if self.mode == "pi":
            connection.send(lcd=message)
        elif self.mode == "sim":
            cprint(message, attrs=["bold"])

    def reset(self, message=None):
        self.progress = 0.

        if message:
            self.set_message("{}\n[{}]".format(message, self.blank))

    def update(self, amt=1., message=None):
        self.progress += amt / self.total

        done = (self.marker * round(min(1, self.progress) * self.bar_length) + self.blank)[:self.bar_length]

        self.set_message("{}\n[{}]".format(message, done))

        if self.progress >= 1.:
            self.progress = 0.


################################ Functions ################################

# PERIODIC LCD CLEAR
def check_clear():
    global PROGRESS_BAR

    lcd_clear = log.THRESHOLDS["num_recognized"] / log.THRESHOLDS["missed_frames"]
    if log.LAST_LOGGED - timer() > lcd_clear or log.UNK_LAST_LOGGED - timer() > lcd_clear:
        PROGRESS_BAR.reset()


# PBAR UPDATE
def update_progress(update_recognized):
    global PROGRESS_BAR

    if update_recognized:
        PROGRESS_BAR.update(amt=PROGRESS_BAR.total, message="Recognizing...")
    elif 1. / PROGRESS_BAR.total + PROGRESS_BAR.progress < 1.:
        PROGRESS_BAR.update(message="Recognizing...")
