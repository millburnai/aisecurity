"""Logging of people's presence.
TODO: refactor this algorithm.
"""

import time
from timeit import default_timer as timer

from termcolor import cprint


def get_now(seconds):
    return time.strftime('%Y-%m-%d %H:%M:%S',
                         time.localtime(seconds)).split(" ")


def get_id(name):
    # TODO: get id
    if "visitor" in name.lower():
        return "V0000"
    else:
        return "00000"


def get_percent_diff(item, log):
    try:
        comp = (len(log[item]) / len([i for n in log.values() for i in n]))
    except KeyError:
        comp = 0
    return 1 - comp


class Logger:

    def __init__(self, num_recognized=3, num_unknown=3, percent_diff=0.2,
                 cooldown=5., missed_frames=10):
        self.curr_recognized, self.curr_unknown = 0, 0
        self.last_logged, self.unk_last_logged = timer(), timer()
        self.log, self.dists = {"current": {}, "logged": {}}, []

        self.num_recognized = num_recognized
        self.num_unknown = num_unknown
        self.percent_diff = percent_diff
        self.cooldown = cooldown
        self.missed_frames = missed_frames

    def cooldown_ok(self, elapsed, best_match=None):
        try:
            last_student = max(self.log["logged"],
                               key=lambda person: self.log["logged"][person])
            if best_match == last_student:
                return timer() - elapsed > self.cooldown
            else:
                return True

        except ValueError:
            return True

    def update(self, is_recognized, best_match):
        update_progress = False
        now = timer()
        cooled = self.cooldown_ok(self.last_logged, best_match)

        if len(self.dists) >= self.num_recognized + self.num_unknown:
            self.flush_current(mode="unknown+known", flush_times=False)
            flushed = True
        else:
            flushed = False

        percent_diff = get_percent_diff(best_match, self.log["current"])
        percent_diff_ok = percent_diff <= self.percent_diff

        if is_recognized and cooled:
            if best_match not in self.log["current"]:
                self.log["current"][best_match] = [now]
            else:
                self.log["current"][best_match].append(now)

            if percent_diff_ok or flushed:
                self.curr_recognized += 1
                self.curr_unknown = 0

                if percent_diff_ok and not flushed:
                    update_progress = True

        elif not is_recognized:
            self.curr_unknown += 1
            if self.curr_unknown >= self.num_unknown:
                self.curr_recognized = 0

        update_recognized = self.curr_recognized >= self.num_recognized \
                            and cooled and percent_diff_ok
        update_unrecognized = self.curr_unknown >= self.num_unknown \
                              and self.cooldown_ok(self.unk_last_logged)

        if update_recognized:
            self.log["logged"][self.log_person()] = now
        elif update_unrecognized:
            self.log_unknown()

        return update_progress, update_recognized, update_unrecognized

    def flush_current(self, mode="known", flush_times=True):
        if "known" in mode:
            self.dists = []
            self.log["current"] = {}
            self.curr_recognized = 0

            if flush_times:
                self.last_logged = timer()

        if "unknown" in mode:
            self.dists = []
            self.curr_unknown = 0

            if flush_times:
                self.unk_last_logged = timer()
                
    
    # LOGGING IN DATABASE
    def log_person(self):
        student_name = max(self.log["current"],
                           key=lambda person: len(self.log["current"][person]))
        times = self.log["current"][student_name]
        now = get_now(sum(times) / len(times))

        cprint(f"{student_name} logged at {now[0]}:{now[1]}",
               color="green", attrs=["bold"])

        self.flush_current(mode="known")
        return student_name

    def log_unknown(self):
        now = get_now(timer())
        cprint(f"Unknown activity logged at {now[0]}:{now[1]}",
               color="red", attrs=["bold"])

        self.flush_current(mode="unknown")


class IntegratedLogger:

    def __init__(self, facenet, logger, pbar=None, websocket=None,
                 data_mutable=False, dynamic_log=False):
        self.facenet = facenet
        self.logger = logger
        self.pbar = pbar
        self.websocket = websocket
        self.data_mutable = data_mutable
        self.dynamic_log = dynamic_log

        self.absent_frames = 0
        self.frames = 0
        self.start_time = timer()

    def log_activity(self, best_match, embedding, dist):
        if best_match is None:
            self.absent_frames += 1
            if self.absent_frames > self.logger.missed_frames:
                self.absent_frames = 0
                self.logger.flush_current(mode="known+unknown", flush_times=False)

        else:
            is_recognized = dist <= self.facenet.alpha
            update_progress, update_recognized, update_unrecognized = \
                self.logger.update(is_recognized, best_match)

            if self.pbar and update_progress:
                self.pbar.update_progress(update_recognized)

            if update_recognized and self.websocket:
                self.websocket.send(best_match=best_match)
                self.websocket.receive()

            elif update_unrecognized:
                if self.pbar:
                    self.pbar.reset(message="Recognizing...")

                if self.dynamic_log:
                    filtered = filter(lambda x: "visitor" in x,
                                      self.facenet.data)
                    visitor_num = len(list(filtered)) + 1
                    self.facenet.update_data(f"visitor_{visitor_num}",
                                             [embedding])

                    self.pbar.update(amt=self.pbar.pbar.amt,
                                     message=f"Visitor {visitor_num} created")
                    cprint(f"Visitor {visitor_num} activity logged",
                           color="magenta", attrs=["bold"])

            update = (update_recognized or update_unrecognized)
            if self.data_mutable and update:
                if self.websocket:
                    is_correct = not bool(self.websocket.recv)
                else:
                    prompt = f"Are you {best_match.replace('_', '')}? ".title()
                    user_input = input(prompt).lower()
                    is_correct = bool(len(user_input) == 0
                                      or user_input[0] == "y")

                if is_correct:
                    self.facenet.update_data(best_match, [embedding])
                else:
                    if self.websocket:
                        name, self.websocket.recv = self.websocket.recv, None
                    else:
                        name = input("Who are you? ").lower().replace(" ", "_")

                    if name in self.facenet.data:
                        self.facenet.update_data(name, [embedding])
                        cprint(f"Static entry for '{name}' updated",
                               color="blue", attrs=["bold"])
                    else:
                        cprint(f"'{name}' is not in database", attrs=["bold"])

            if self.pbar:
                self.pbar.check_clear()

            self.logger.dists.append(dist)
            self.frames += 1

    def close(self):
        elapsed = max(timer() - self.start_time, 1e-6)
        print(f"{round(elapsed, 2)}s elapsed, "
              f"{self.frames} frames captured"
              f" = {round(self.frames / elapsed, 2)} fps")
