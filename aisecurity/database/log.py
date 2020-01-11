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
import requests
from termcolor import cprint

from aisecurity.utils.paths import CONFIG_HOME, CONFIG


# SETUP
THRESHOLDS = {
    "num_recognized": 3,
    "num_unknown": 3,
    "percent_diff": 0.2,
    "cooldown": 0.,
    "missed_frames": 10,
}

MODE = None

DATABASE = None
CURSOR = None
FIREBASE = None

NUM_RECOGNIZED, NUM_UNKNOWN = None, None
LAST_LOGGED, UNK_LAST_LOGGED = None, None
CURRENT_LOG, DISTS = None, None

USE_SERVER = None


# LOGGING INIT AND HELPERS
def init(logging, flush=False, thresholds=None):
    global NUM_RECOGNIZED, NUM_UNKNOWN, LAST_LOGGED, UNK_LAST_LOGGED, CURRENT_LOG, DISTS, THRESHOLDS
    global MODE, DATABASE, CURSOR, FIREBASE

    MODE = logging

    NUM_RECOGNIZED, NUM_UNKNOWN = 0, 0
    LAST_LOGGED, UNK_LAST_LOGGED = time.time(), time.time()
    CURRENT_LOG, DISTS = {}, []

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
        except (FileNotFoundError, json.JSONDecodeError):
            raise ValueError(CONFIG_HOME + "/logging/firebase.json and a key file are needed to use firebase")

    elif logging == "django":
        # TODO: fill in init for django logging
        try:
            # initialize django database
            # for now, hardcode any links/ip addresses that you might need to use to connect to a db
            pass
        except Exception:
            # replace Exception with correct django error that might occur
            raise Exception("django logging not able to be initialized")

    else:
        MODE = None
        warnings.warn("{} not a supported logging option. No logging will occur".format(logging))

    if thresholds:
        THRESHOLDS = {**THRESHOLDS, **thresholds}


def server_init():
    global USE_SERVER

    try:
        print("Connecting to server...")
        requests.get(CONFIG["server_address"], timeout=1.)
        USE_SERVER = True
    except (requests.exceptions.Timeout, requests.exceptions.MissingSchema, KeyError) as error:
        if isinstance(error, requests.exceptions.Timeout):
            warnings.warn("ID server unreachable")
        elif isinstance(error, requests.exceptions.MissingSchema):
            warnings.warn("Invalid server address in config")
        elif isinstance(error, KeyError):
            warnings.warn("Server address missing in config")
        USE_SERVER = False


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

    if len(DISTS) >= THRESHOLDS["num_recognized"] + THRESHOLDS["num_unknown"]:
        flush_current(mode="unknown+known")

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
        if NUM_UNKNOWN >= THRESHOLDS["num_unknown"]:
            NUM_RECOGNIZED = 0


# LOGGING FUNCTIONS
def log_person(student_name, times):
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
        DATABASE.child("known").child(*get_now(time.time())).set(data)

    elif MODE == "django":
        student_id = get_id(student_name)
        student_name = student_name.replace("_", " ").title()
        # TODO: log {*now: student_id, student_name} in django db
            # DJANGO: s = StudentLog(student_id=student_id, name=student_name, time=datetime.datetime.now()) 
            #         s.save()
            # from models.py; import studentlog, datetime
        
        pass

    flush_current(mode="known")

    cprint("Regular activity logged ({})".format(student_name), color="green", attrs=["bold"])


def log_unknown(path_to_img):
    now = get_now(time.time())

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
        DATABASE.child("unknown").child(*get_now(time.time())).set(data)

    elif MODE == "django":
        # TODO: log {*get_now(time.time()): path_to_img} in django db
            # DJANGO: u = UnknownLog(time=datetime.datetime.now(), path_to_img=path_to_img)
            #         u.save()
            # from models.py; import datetime, unknownlog
        pass

    flush_current(mode="unknown")

    cprint("Unknown activity logged", color="red", attrs=["bold"])


def flush_current(mode="known", flush_times=True):
    global CURRENT_LOG, NUM_RECOGNIZED, NUM_UNKNOWN, DISTS, LAST_LOGGED, UNK_LAST_LOGGED

    if "known" in mode:
        CURRENT_LOG = {}
        NUM_RECOGNIZED = 0
        if flush_times:
            LAST_LOGGED = time.time()
    if "unknown" in mode:
        DISTS = []
        NUM_UNKNOWN = 0
        if flush_times:
            UNK_LAST_LOGGED = time.time()
