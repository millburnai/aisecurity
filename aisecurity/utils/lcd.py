"""

"aisecurity.hardware.lcd"

LCD utils.

"""

from timeit import default_timer as timer
import warnings

from termcolor import cprint

from aisecurity.db import log
from aisecurity.utils import connection

################################ Classes ################################

# LCD PROGRESS BAR
class LCDProgressBar:

    def __init__(self, mode="pi" if connection.SOCKET else "sim", length=16, marker="#"):
        assert mode in ("pi", "sim"), "supported modes are physical (physical LCD) and dev (testing)"

        try:
            if mode == "pi":
                assert connection.SOCKET, "connection.SOCKET must be initialized by using connection.init()"
            self.mode = mode

        except (ValueError, NameError, AssertionError):
            self.mode = "sim"
            if self.mode != mode:
                warnings.warn("pi lcd mode requested but only simulation lcd available")

        self.bar_length = length - 2  # compensate for [] at beginning and end
        self.marker = marker
        self.progress = 0.
        self.blank = " " * self.bar_length

        self.set_message("Loading...\n[ Initializing ]")

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
        self.progress += amt / log.THRESHOLDS["num_recognized"]

        done = (self.marker * round(min(1, self.progress) * self.bar_length) + self.blank)[:self.bar_length]

        self.set_message("{}\n[{}]".format(message, done))

        if self.progress >= 1.:
            self.progress = 0.

    # PERIODIC LCD CLEAR
    def check_clear(self):

        lcd_clear = log.THRESHOLDS["num_recognized"] / log.THRESHOLDS["missed_frames"]
        if log.LAST_LOGGED - timer() > lcd_clear or log.UNK_LAST_LOGGED - timer() > lcd_clear:
            self.reset()

    # PBAR UPDATE
    def update_progress(self, update_recognized):

        if update_recognized:
            self.update(amt=log.THRESHOLDS["num_recognized"], message="Recognizing...")
        elif 1. / log.THRESHOLDS["num_recognized"] + self.progress < 1.:
            self.update(message="Recognizing...")

