
"""

"log.py"

MySQL logging handling.

"""

import time
import warnings

import mysql.connector

from extras.paths import Paths

# SETUP
THRESHOLDS = {
  "num_recognized": 3.0,
  "num_unrecognized": 25.0,
  "percent_diff": 0.2,
  "cooldown": 10.0,
  "time_since_previous": 3.0,
  "max_error": 1.0
}

num_recognized = 0
num_unrecognized = 0

rec_last_logged = time.time() - THRESHOLDS["num_recognized"] + 0.1 # don't log for first 0.1 secs
unrec_last_logged = time.time() - THRESHOLDS["num_unrecognized"] + 0.1

current_log = {}

try:
  database = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="Blast314" if "ryan" in Paths.HOME else "KittyCat123",
      database="LOG"
      )
  cursor = database.cursor()
except mysql.connector.errors.DatabaseError:
  warnings.warn("Database credentials missing or incorrect")

# LOGGING INIT AND HELPERS
def init(flush=False, thresholds=None):
  cursor.execute("USE LOG;")
  database.commit()

  if flush:
    instructions = open(Paths.HOME + "/logs/drop.sql", "r")
    for cmd in instructions:
      if not cmd.startswith(" ") and not cmd.startswith("*/") and not cmd.startswith("/*"): # allows for docstrings
        cursor.execute(cmd)
        database.commit()

  if thresholds:
    global THRESHOLDS
    THRESHOLDS = {**THRESHOLDS, **thresholds} # combining and overwriting THRESHOLDS with thresholds param

def get_now(seconds):
  date_and_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(seconds))
  return date_and_time.split(" ")

def get_id(name):
  # will be filled in later
  return "00000"

def get_percent_diff(best_match):
  return 1.0 - (len(current_log[best_match]) / len([item for sublist in current_log.values() for item in sublist]))

def update_current_logs(is_recognized, best_match, l2_dist):
  global current_log, num_recognized, num_unrecognized

  if is_recognized and l2_dist <= THRESHOLDS["max_error"]:
    now = time.time()

    if best_match not in current_log:
      current_log[best_match] = [now]
    else:
      current_log[best_match].append(now)

    if len(current_log[best_match]) == 1 or get_percent_diff(best_match) <= THRESHOLDS["percent_diff"]:
      num_recognized += 1
      num_unrecognized = 0

  else:
    num_unrecognized += 1
    num_recognized = 0

# LOGGING FUNCTIONS
def log_person(student_name, times):
  add = "INSERT INTO Transactions (student_id, student_name, date, time) VALUES ({}, '{}', '{}', '{}');".format(
    get_id(student_name), student_name, *get_now(sum(times) / len(times)))
  cursor.execute(add)
  database.commit()

  flush_current(regular_activity=True)

def log_suspicious(path_to_img):
  add = "INSERT INTO Suspicious (path_to_img, date, time) VALUES ('{}', '{}', '{}');".format(
    path_to_img, *get_now(time.time()))
  cursor.execute(add)
  database.commit()

  flush_current(regular_activity=False)

def flush_current(regular_activity=True):
  global current_log, num_recognized, rec_last_logged, num_unrecognized, unrec_last_logged
  if regular_activity:
    current_log = {}
    num_recognized = 0
    rec_last_logged = time.time()
  else:
    num_unrecognized = 0
    unrec_last_logged = time.time()
