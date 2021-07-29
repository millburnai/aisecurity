import json
from termcolor import cprint


class ProgressBar:

    def __init__(self, logger, length=16, marker="#", ws=None):
        self.logger = logger
        self.bar_length = length - 2  # compensate for [] at beginning and end
        self.marker = marker
        self.progress = 0.
        self.blank = " " * self.bar_length
        self.ws = ws
        self.set_message("Initializing....")

    def set_message(self, message):
        if self.ws is not None:
            self.ws.send(json.dumps({"pbar": message}))
        else:
            cprint(message, attrs=["bold"])

    def reset(self, message=None):
        self.progress = 0.
        if message:
            self.set_message(f"{message}\n[{self.blank}]")

    def _update(self, amt, message):
        self.progress += amt / self.logger.frame_threshold
        num_done = int(round(min(1., self.progress) * self.bar_length))
        done = (self.marker * num_done + self.blank)[:self.bar_length]
        self.set_message(f"{message}\n[{done}]")
        if self.progress >= 1.:
            self.reset("Face rec running")

    def update(self, amt=1., message="Recognizing.....", end=False):
        if end or self.progress + amt / self.logger.frame_threshold <= 1:
            self._update(amt, message)
