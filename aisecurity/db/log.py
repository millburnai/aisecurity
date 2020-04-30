"""MySQL and Firebase database handling.

"""

import json
import time
from timeit import default_timer as timer
import warnings

import mysql.connector
import pyrebase
from termcolor import cprint

from aisecurity.utils.paths import config_home, config


################################ Logging ###############################

# BASIC LOGGER
class Logger:

    # INIT
    def __init__(self, mode, num_recognized=3, num_unknown=3, percent_diff=0.2, cooldown=5., missed_frames=10):
        # database init
        self.mode = mode

        if self.mode == "mysql":
            try:
                self._db = mysql.connector.connect(
                    host="localhost",
                    user=config["mysql_user"],
                    passwd=config["mysql_password"],
                    database="Log"
                )
                self._cursor = self._db.cursor()

                self._cursor.execute("USE Log;")
                self._db.commit()

            except (mysql.connector.errors.DatabaseError, mysql.connector.errors.InterfaceError):
                self.mode = "<no database>"
                warnings.warn("MySQL database credentials missing or incorrect")

        elif self.mode == "firebase":
            try:
                self._firebase = pyrebase.initialize_app(
                    json.load(open(config_home + "/logging/firebase.json", encoding="utf-8"))
                )
                self._db = self._firebase.database()

            except (FileNotFoundError, json.JSONDecodeError):
                self.mode = "<no database>"
                warnings.warn(config_home + "/logging/firebase.json and a key file are needed to use firebase")

        else:
            self.mode = "<no database>"
            warnings.warn("{} not a supported logging option. No logging will occur".format(mode))
            
        # stuff to keep track of
        self.curr_recognized, self.curr_unknown = 0, 0
        self.last_logged, self.unk_last_logged = timer(), timer()
        self.log, self.dists = {"current": {}, "logged": {}}, []

        # thresholds
        self.num_recognized = num_recognized
        self.num_unknown = num_unknown
        self.percent_diff = percent_diff
        self.cooldown = cooldown
        self.missed_frames = missed_frames
        
    
    # HELPERS
    @staticmethod
    def _get_now(seconds):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(seconds)).split(" ")

    @staticmethod
    def _get_id(name):
        # will be filled in later
        if "visitor" in name.lower():
            return "V0000"
        else:
            return "00000"

    @staticmethod
    def _get_percent_diff(item, log):
        try:
            return 1. - (len(log[item]) / len([i for n in log.values() for i in n]))
        except KeyError:
            return 1.

    def _cooldown_ok(self, elapsed, best_match=None):
        try:
            last_student = max(self.log["logged"], key=lambda person: self.log["logged"][person])
            if best_match == last_student:
                return timer() - elapsed > self.cooldown
            else:
                return True

        except ValueError:
            return True
        

    # UPDATE LOGS
    def update(self, is_recognized, best_match):
        update_progress = False
        now = timer()
        cooled = self._cooldown_ok(self.last_logged, best_match)

        if len(self.dists) >= self.num_recognized + self.num_unknown:
            self.flush_current(mode="unknown+known", flush_times=False)
            flushed = True
        else:
            flushed = False

        if is_recognized and cooled:
            if best_match not in self.log["current"]:
                self.log["current"][best_match] = [now]
            else:
                self.log["current"][best_match].append(now)

            percent_diff_ok = self._get_percent_diff(best_match, self.log["current"]) <= self.percent_diff

            if percent_diff_ok or flushed:
                self.curr_recognized += 1
                self.curr_unknown = 0

                if percent_diff_ok and not flushed:
                    update_progress = True

        elif not is_recognized:
            self.curr_unknown += 1
            if self.curr_unknown >= self.num_unknown:
                self.curr_recognized = 0

        update_recognized = self.curr_recognized >= self.num_recognized and cooled \
                            and self._get_percent_diff(best_match, self.log["current"]) <= self.percent_diff
        update_unrecognized = self.curr_unknown >= self.num_unknown and self._cooldown_ok(self.unk_last_logged)

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
        student_name = max(self.log["current"], key=lambda person: len(self.log["current"][person]))
        times = self.log["current"][student_name]
        now = self._get_now(sum(times) / len(times))

        if self.mode == "mysql":
            add = "INSERT INTO Activity (id, name, date, time) VALUES ({}, '{}', '{}', '{}');".format(
                self._get_id(student_name), student_name.replace("_", " ").title(), *now)
            self._cursor.execute(add)
            self._db.commit()

        elif self.mode == "firebase":
            data = {
                "id": self._get_id(student_name),
                "name": student_name.replace("_", " ").title(),
                "date": now[0],
                "time": now[1]
            }
            self._db.child("known").child(*self._get_now(timer())).set(data)

        self.flush_current(mode="known")

        cprint("Regular activity ({}) logged with {}".format(student_name, self.mode), color="green", attrs=["bold"])

        return student_name

    def log_unknown(self):
        now = self._get_now(timer())

        if self.mode == "mysql":
            add = "INSERT INTO Unknown (path_to_img, date, time) VALUES ('{}', '{}', '{}');".format(
                "<DEPRECATED>", *now)
            self._cursor.execute(add)
            self._db.commit()

        elif self.mode == "firebase":
            data = {
                "path_to_img": "<DEPRECATED>",
                "date": now[0],
                "time": now[1]
            }
            self._db.child("unknown").child(*self._get_now(timer())).set(data)

        self.flush_current(mode="unknown")

        cprint("Unknown activity logged with {}".format(self.mode), color="red", attrs=["bold"])


# FACENET LOGGER
class IntegratedLogger:

    def __init__(self, facenet, logger, pbar=None, websocket=None, data_mutable=False, dynamic_log=False):
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
            is_recognized = dist <= self.facenet.dist_metric.alpha
            update_progress, update_recognized, update_unrecognized = self.logger.update(is_recognized, best_match)

            if self.pbar and update_progress:
                self.pbar.update_progress(update_recognized)

            if update_recognized and self.websocket:
                self.websocket.send(best_match=best_match)
                self.websocket.receive()

            elif update_unrecognized:
                if self.pbar:
                    self.pbar.reset(message="Recognizing...")

                if self.dynamic_log:
                    visitor_num = len([person for person in self.facenet.data if "visitor" in person]) + 1
                    self.facenet.update_data("visitor_{}".format(visitor_num), [embedding])

                    self.pbar.update(amt=self.pbar.amt, message="Visitor {} created".format(visitor_num))
                    cprint("Visitor {} activity logged".format(visitor_num), color="magenta", attrs=["bold"])

            if self.data_mutable and (update_recognized or update_unrecognized):
                if self.websocket:
                    is_correct = not bool(self.websocket.recv)
                else:
                    user_input = input("Are you {}? ".format(best_match.replace("_", " ").title())).lower()
                    is_correct = bool(len(user_input) == 0 or user_input[0] == "y")

                if is_correct:
                    self.facenet.update_data(best_match, [embedding])
                else:
                    if self.websocket:
                        name, self.websocket.recv = self.websocket.recv, None
                    else:
                        name = input("Who are you? ").lower().replace(" ", "_")

                    if name in self.facenet.data:
                        self.facenet.update_data(name, [embedding])
                        cprint("Static entry for '{}' updated".format(name), color="blue", attrs=["bold"])
                    else:
                        cprint("'{}' is not in database".format(name), attrs=["bold"])

            if self.pbar:
                self.pbar.check_clear()

            self.logger.dists.append(dist)
            self.frames += 1


    def close(self):
        elapsed = max(timer() - self.start_time, 1e-6)
        ms_elapsed = round(elapsed * 1000., 2)
        fps = round(self.frames / elapsed, 2)

        print("{} ms elapsed, {} frames captured = {} fps".format(ms_elapsed, self.frames, fps))
