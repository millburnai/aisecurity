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


# SETUP
THRESHOLDS = {
    "num_recognized": 3,
    "num_unknown": 3,
    "percent_diff": 0.2,
    "cooldown": 5.,
    "missed_frames": 10,
}

MODE = None

DATABASE = None
CURSOR = None
FIREBASE = None

NUM_RECOGNIZED, NUM_UNKNOWN = 0, 0
LAST_LOGGED, UNK_LAST_LOGGED = None, None
LOG, DISTS = {"current": {}, "logged": {}}, []


# LOGGING INIT AND HELPERS
def init(logging, flush=False, thresholds=None):
    global NUM_RECOGNIZED, NUM_UNKNOWN, LAST_LOGGED, UNK_LAST_LOGGED, LOG, DISTS, THRESHOLDS
    global MODE, DATABASE, CURSOR, FIREBASE

    MODE = logging

    LAST_LOGGED, UNK_LAST_LOGGED = timer(), timer()

    if logging == "mysql":
        try:
            DATABASE = mysql.connector.connect(
                host="localhost",
                user=config["mysql_user"],
                passwd=config["mysql_password"],
                database="LOG"
            )
            CURSOR = DATABASE.cursor()

            CURSOR.execute("USE LOG;")
            DATABASE.commit()

            if flush:
                instructions = open(config_home + "/bin/drop.sql", encoding="utf-8")
                for cmd in instructions:
                    if not cmd.startswith(" ") and not cmd.startswith("*/") and not cmd.startswith("/*"):
                        CURSOR.execute(cmd)
                        DATABASE.commit()

        except (mysql.connector.errors.DatabaseError, mysql.connector.errors.InterfaceError):
            MODE = "<no database>"
            warnings.warn("MySQL database credentials missing or incorrect")

    elif logging == "firebase":
        try:
            FIREBASE = pyrebase.initialize_app(
                json.load(open(config_home + "/logging/firebase.json", encoding="utf-8"))
            )
            DATABASE = FIREBASE.database()

        except (FileNotFoundError, json.JSONDecodeError):
            MODE = "<no database>"
            warnings.warn(config_home + "/logging/firebase.json and a key file are needed to use firebase")

    else:
        MODE = "<no database>"
        warnings.warn("{} not a supported logging option. No logging will occur".format(logging))

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
        return 1.


def update(is_recognized, best_match):
    global LOG, NUM_RECOGNIZED, NUM_UNKNOWN

    update_progress = False
    now = timer()
    cooled = cooldown_ok(LAST_LOGGED, best_match)

    if len(DISTS) >= THRESHOLDS["num_recognized"] + THRESHOLDS["num_unknown"]:
        flush_current(mode="unknown+known", flush_times=False)
        flushed = True
    else:
        flushed = False

    if is_recognized and cooled:
        if best_match not in LOG["current"]:
            LOG["current"][best_match] = [now]
        else:
            LOG["current"][best_match].append(now)

        percent_diff_ok = get_percent_diff(best_match, LOG["current"]) <= THRESHOLDS["percent_diff"]

        if percent_diff_ok or flushed:
            NUM_RECOGNIZED += 1
            NUM_UNKNOWN = 0

            if percent_diff_ok and not flushed:
                update_progress = True

    elif not is_recognized:
        NUM_UNKNOWN += 1
        if NUM_UNKNOWN >= THRESHOLDS["num_unknown"]:
            NUM_RECOGNIZED = 0

    update_recognized = NUM_RECOGNIZED >= THRESHOLDS["num_recognized"] and cooled \
                        and get_percent_diff(best_match, LOG["current"]) <= THRESHOLDS["percent_diff"]
    update_unrecognized = NUM_UNKNOWN >= THRESHOLDS["num_unknown"] and cooldown_ok(UNK_LAST_LOGGED)

    if update_recognized:
        LOG["logged"][log_person()] = now
    elif update_unrecognized:
        log_unknown()

    return update_progress, update_recognized, update_unrecognized


def cooldown_ok(elapsed, best_match=None):
    try:
        last_student = max(LOG["logged"], key=lambda person: LOG["logged"][person])
        if best_match == last_student:
            return timer() - elapsed > THRESHOLDS["cooldown"]
        else:
            return True

    except ValueError:
        return True


# LOGGING FUNCTIONS
def log_person():
    student_name = max(LOG["current"], key=lambda person: len(LOG["current"][person]))
    times = LOG["current"][student_name]
    now = get_now(sum(times) / len(times))

    if MODE == "mysql":
        add = "INSERT INTO Activity (id, name, date, time) VALUES ({}, '{}', '{}', '{}');".format(
            get_id(student_name), student_name.replace("_", " ").title(), *now)
        CURSOR.execute(add)
        DATABASE.commit()

    elif MODE == "firebase":
        data = {
            "id": get_id(student_name),
            "name": student_name.replace("_", " ").title(),
            "date": now[0],
            "time": now[1]
        }
        DATABASE.child("known").child(*get_now(timer())).set(data)

    flush_current(mode="known")

    cprint("Regular activity ({}) logged with {}".format(student_name, MODE), color="green", attrs=["bold"])

    return student_name


def log_unknown():
    now = get_now(timer())

    path_to_img = "<DEPRECATED>"

    if MODE == "mysql":
        add = "INSERT INTO Unknown (path_to_img, date, time) VALUES ('{}', '{}', '{}');".format(
            path_to_img, *now)
        CURSOR.execute(add)
        DATABASE.commit()

    elif MODE == "firebase":
        data = {
            "path_to_img": path_to_img,
            "date": now[0],
            "time": now[1]
        }
        DATABASE.child("unknown").child(*get_now(timer())).set(data)

    flush_current(mode="unknown")

    cprint("Unknown activity logged with {}".format(MODE), color="red", attrs=["bold"])


def flush_current(mode="known", flush_times=True):
    global LOG, NUM_RECOGNIZED, NUM_UNKNOWN, DISTS, LAST_LOGGED, UNK_LAST_LOGGED

    if "known" in mode:
        DISTS = []
        LOG["current"] = {}
        NUM_RECOGNIZED = 0

        if flush_times:
            LAST_LOGGED = timer()

    if "unknown" in mode:
        DISTS = []
        NUM_UNKNOWN = 0

        if flush_times:
            UNK_LAST_LOGGED = timer()
