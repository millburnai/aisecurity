"""

"aisecurity.logging.log"

MySQL and Firebase logging handling.

"""

import json
import time
import warnings

import mysql.connector
import pyrebase
from pyrebase import *

from aisecurity.utils.paths import CONFIG_HOME, CONFIG


# SETUP
DATABASE = None
CURSOR = None
FIREBASE = None

THRESHOLDS = {
    "num_recognized": 3,
    "num_unknown": 5,
    "percent_diff": 0.2,
    "cooldown": 10.,
    "missed_frames": 10,
    "refresh_rate": 3.
}

num_recognized = 0
num_unknown = 0

last_logged = time.time() - THRESHOLDS["cooldown"] + 0.1  # don't log for first 0.1s- it's just warming up then
unk_last_logged = time.time() - THRESHOLDS["cooldown"] + 0.1

current_log = {}
current_log_start = time.time() - THRESHOLDS["cooldown"] + 0.1


# LOGGING INIT AND HELPERS
def init(flush=False, thresholds=None, logging="firebase"):
    global DATABASE, CURSOR, FIREBASE

    if logging == "mysql":

        try:
            DATABASE = mysql.connector.connect(
                host="localhost",
                user=CONFIG["mysql_user"],
                passwd=CONFIG["mysql_password"],
                database="LOG"
            )
            CURSOR = DATABASE.cursor()

        except (mysql.connector.errors.DatabaseError, mysql.connector.errors.InterfaceError):
            warnings.warn("MySQL database credentials missing or incorrect")

        CURSOR.execute("USE LOG;")
        DATABASE.commit()

        if flush:
            instructions = open(CONFIG_HOME + "/bin/drop.sql")
            for cmd in instructions:
                if not cmd.startswith(" ") and not cmd.startswith("*/") and not cmd.startswith("/*"):
                    CURSOR.execute(cmd)
                    DATABASE.commit()

    elif logging == "firebase":
        try:
            FIREBASE = pyrebase.initialize_app(json.load(open(CONFIG_HOME + "/logging/firebase.json")))
            DATABASE = FIREBASE.database()
        except FileNotFoundError:
            raise FileNotFoundError(CONFIG_HOME + "/logging/firebase.json and a key file are needed to use firebase")

    if thresholds:
        global THRESHOLDS
        THRESHOLDS = {**THRESHOLDS, **thresholds}


def get_now(seconds):
    date_and_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(seconds))
    return date_and_time.split(" ")


def get_id(name):
    # will be filled in later
    return "00000"


def get_percent_diff(best_match):
    return 1. - (len(current_log[best_match]) / len([item for sublist in current_log.values() for item in sublist]))


def update_current_logs(is_recognized, best_match):
    global current_log, num_recognized, num_unknown

    if time.time() - current_log_start >= THRESHOLDS["refresh_rate"]:
        flush_current()

    if is_recognized:
        now = time.time()

        if best_match not in current_log:
            current_log[best_match] = [now]
        else:
            current_log[best_match].append(now)

        if len(current_log[best_match]) == 1 or get_percent_diff(best_match) <= THRESHOLDS["percent_diff"] + 0.2:
            num_recognized += 1
            num_unknown = 0

    else:
        num_unknown += 1
        if num_unknown >= THRESHOLDS["num_unknown"]:
            num_recognized = 0


# LOGGING FUNCTIONS
def log_person(student_name, times, firebase=True):
    now = get_now(sum(times) / len(times))

    if not firebase:
        add = "INSERT INTO Activity (student_id, student_name, date, time) VALUES ({}, '{}', '{}', '{}');".format(
            get_id(student_name), student_name.replace("_", " ").title(), *now)
        CURSOR.execute(add)
        DATABASE.commit()

    else:
        path = DATABASE.child("known")
        data = {
            "student_id": get_id(student_name),
            "student_name": student_name.replace("_", " ").title(),
            "date": now[0],
            "time": now[1]
        }
        if path.get().val() is None:
            DATABASE.child("known").set(data)
        else:
            DATABASE.child("known").update(data)


    global last_logged
    last_logged = time.time()

    flush_current(regular_activity=True)


def log_unknown(path_to_img, firebase=True):
    now = get_now(time.time())

    if not firebase:
        add = "INSERT INTO Unknown (path_to_img, date, time) VALUES ('{}', '{}', '{}');".format(
            path_to_img, *now)
        CURSOR.execute(add)
        DATABASE.commit()

    else:
        path = DATABASE.child("known")
        data = {
            "path_to_img": path_to_img,
            "date": now[0],
            "time": now[1]
        }
        if path.get().val() is None:
            DATABASE.child("unknown").set(data)
        else:
            DATABASE.child("unknown").update(data)

    global unk_last_logged
    unk_last_logged = time.time()

    flush_current(regular_activity=False)


def flush_current(regular_activity=True):
    global current_log, num_recognized, num_unknown, current_log_start

    if regular_activity:
        current_log = {}
        num_recognized = 0
        current_log_start = time.time()
    else:
        num_unknown = 0
