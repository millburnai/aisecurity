"""LCD utils.
"""

from timeit import default_timer as timer

from termcolor import cprint


class LCDProgressBar:

    def __init__(self, mode, total, length=16, marker="#", websocket=None):
        assert mode in ("pi", "sim"), \
            "supported modes are physical (physical LCD) and dev (testing)"

        try:
            assert websocket and "pi" == mode
            self.mode = "pi"

        except (ValueError, NameError, AssertionError):
            self.mode = "sim"
            if self.mode != mode:
                print("[DEBUG] pi lcd mode requested but "
                      "only simulation lcd available")

        self.total = total
        self.bar_length = length - 2  # compensate for [] at beginning and end
        self.marker = marker
        self.progress = 0.
        self.blank = " " * self.bar_length
        self.websocket = websocket

    def set_message(self, message):
        if self.mode == "pi":
            self.websocket.send(lcd=message)
        elif self.mode == "sim":
            cprint(message, attrs=["bold"])

    def reset(self, message=None):
        self.progress = 0.

        if message:
            self.set_message("{}\n[{}]".format(message, self.blank))

    def update(self, amt=1., message=None):
        self.progress += amt / self.total

        ct = self.marker * round(min(1, self.progress) * self.bar_length)
        done = (ct + self.blank)[:self.bar_length]

        self.set_message("{}\n[{}]".format(message, done))

        if self.progress >= 1.:
            self.progress = 0.


# LCDProgressBar + logging
class IntegratedLCDProgressBar:

    def __init__(self, logger, websocket=None):
        self.logger = logger

        self.pbar = LCDProgressBar(mode="pi",
                                   total=self.logger.num_recognized,
                                   websocket=websocket)
        self.pbar.set_message("Loading...\n[ Initializing ]")

    def check_clear(self):
        lcd_clear = self.logger.num_recognized / self.logger.missed_frames

        ck_logged = self.logger.last_logged - timer() > lcd_clear
        ck_unk_logged = self.logger.unk_last_logged - timer() > lcd_clear

        if ck_logged or ck_unk_logged:
            self.pbar.reset()

    def update_progress(self, update_recognized):
        if update_recognized:
            self.pbar.update(amt=self.pbar.total, message="Recognizing...")
        elif 1. / self.pbar.total + self.pbar.progress < 1.:
            self.pbar.update(message="Recognizing...")

    def update(self, *args, **kwargs):
        self.pbar.update(*args, **kwargs)

    def reset(self, *args, **kwargs):
        self.pbar.reset(*args,  **kwargs)
