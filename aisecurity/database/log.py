"""

"aisecurity.database.log"

MySQL and Firebase database handling.

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
    "NUM_RECOGNIZED": 3,
    "NUM_UNKNOWN": 3,
    "percent_diff": 0.2,
    "cooldown": 0.,
    "missed_frames": 10,
}

NUM_RECOGNIZED, NUM_UNKNOWN = None, None
LAST_LOGGED, UNK_LAST_LOGGED = None, None
CURRENT_LOG, L2_DISTS = None, None


# LOGGING INIT AND HELPERS
def init(flush=False, thresholds=None, logging="firebase"):
    global NUM_RECOGNIZED, NUM_UNKNOWN, LAST_LOGGED, UNK_LAST_LOGGED, CURRENT_LOG, L2_DISTS, THRESHOLDS
    global DATABASE, CURSOR, FIREBASE

    NUM_RECOGNIZED, NUM_UNKNOWN = 0, 0
    LAST_LOGGED, UNK_LAST_LOGGED = time.time(), time.time()
    CURRENT_LOG, L2_DISTS = {}, []


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
        THRESHOLDS = {**THRESHOLDS, **thresholds}


def get_now(seconds):
    date_and_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(seconds))
    return date_and_time.split(" ")


def get_id(name):
    # will be filled in later
    if "visitor" in name.lower():
        return "V0000"
    else:
        return "00000"


def get_percent_diff(item, log):
    try:
        return 1. - (len(log[item]) / len([i for n in log.values() for i in n]))
    except KeyError:
        return 1.0


def update_current_logs(is_recognized, best_match):
    global CURRENT_LOG, NUM_RECOGNIZED, NUM_UNKNOWN

    if len(L2_DISTS) >= THRESHOLDS["NUM_RECOGNIZED"] + THRESHOLDS["NUM_UNKNOWN"]:
        flush_current(mode=["unknown", "known"])

    if is_recognized:
        now = time.time()

        if best_match not in CURRENT_LOG:
            CURRENT_LOG[best_match] = [now]
        else:
            CURRENT_LOG[best_match].append(now)

        if len(CURRENT_LOG[best_match]) == 1 or get_percent_diff(best_match, CURRENT_LOG) <= THRESHOLDS["percent_diff"]:
            NUM_RECOGNIZED += 1
            NUM_UNKNOWN = 0

    else:
        NUM_UNKNOWN += 1
        if NUM_UNKNOWN >= THRESHOLDS["NUM_UNKNOWN"]:
            NUM_RECOGNIZED = 0


# LOGGING FUNCTIONS
def log_person(student_name, times, firebase=True):
    now = get_now(sum(times) / len(times))

    if not firebase:
        add = "INSERT INTO Activity (id, name, date, time) VALUES ({}, '{}', '{}', '{}');".format(
            get_id(student_name), student_name.replace("_", " ").title(), *now)
        CURSOR.execute(add)
        DATABASE.commit()

    else:
        data = {
            "id": get_id(student_name),
            "name": student_name.replace("_", " ").title(),
            "date": now[0],
            "time": now[1]
        }
        DATABASE.child("known").child(*get_now(time.time())).set(data)

    flush_current(mode="known")


def log_unknown(path_to_img, firebase=True):
    now = get_now(time.time())

    if not firebase:
        add = "INSERT INTO Unknown (path_to_img, date, time) VALUES ('{}', '{}', '{}');".format(
            path_to_img, *now)
        CURSOR.execute(add)
        DATABASE.commit()

    else:
        data = {
            "path_to_img": path_to_img,
            "date": now[0],
            "time": now[1]
        }
        DATABASE.child("unknown").child(*get_now(time.time())).set(data)

    flush_current(mode="unknown")


def flush_current(mode="known", flush_times=True):
    global CURRENT_LOG, NUM_RECOGNIZED, NUM_UNKNOWN, L2_DISTS, LAST_LOGGED, UNK_LAST_LOGGED

    if "known" in mode:
        CURRENT_LOG = {}
        NUM_RECOGNIZED = 0
        if flush_times:
            LAST_LOGGED = time.time()
    if "unknown" in mode:
        L2_DISTS = []
        NUM_UNKNOWN = 0
        if flush_times:
            UNK_LAST_LOGGED = time.time()
