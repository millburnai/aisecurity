
"""

"log.py"

MySQL logging handling.

"""

import time
import warnings

import mysql.connector

from extras.paths import Paths
from datetime import datetime


# SETUP
THRESHOLDS = {"num_recognized": 10,
              "percent_same": 0.2,
              "cooldown": 10, 
              "num_unrecognized": 5}

num_recognized = 0
num_unrecognized = 0
current_log = {}
unrec_last_logged = datetime.now()
rec_last_logged = datetime.now()

try:
  database = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="KittyCat123",
      database="LOG"
      )
  cursor = database.cursor()
except mysql.connector.errors.DatabaseError:
  warnings.warn("Database credentials missing or incorrect")

# LOGGING INIT AND HELPERS
def init(flush=False):
  cursor.execute("USE LOG;")
  database.commit()

  if flush:
    instructions = open(Paths.HOME + "/logs/drop.sql", "r")
    for cmd in instructions:
      if not cmd.startswith(" ") and not cmd.startswith("*/") and not cmd.startswith("/*"): # allows for docstrings
        cursor.execute(cmd)
        database.commit()

def get_now(seconds):
  date_and_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(seconds))
  return date_and_time.split(" ")

def get_id(name):
  # will be filled in later
  return 00000

# LOGGING FUNCTIONS
def get_now_liam(compare=False):
  now = datetime.now()
  if compare:
    return now
  else:
    time = datetime.strptime("{}:{}:{}".format(now.hour, now.minute, now.second), "%H:%M:%S")
    date = datetime.strptime("{}/{}/{}".format(now.day, now.month, now.year), "%d/%m/%Y")
    return date, time

def log_person(student_name, times):
  add = "INSERT INTO Transactions (student_id, student_name, date, time) VALUES ({}, '{}', '{}', '{}');".format(
    get_id(student_name), student_name, *get_now(sum(times) / len(times)))
  cursor.execute(add)
  database.commit()

  global rec_last_logged
  rec_last_logged = datetime.now()

  _flush_current()

def log_suspicious(path_to_img):
	date, time = get_now_liam()
	add = "INSERT INTO Suspicious (path_to_img, date, time) VALUES ('{}', '{}', '{}');".format(path_to_img, date, time)
	cursor.execute(add)
	database.commit()
	_flush_current(person=False)
	global unrec_last_logged
	unrec_last_logged = datetime.now()

  

def _flush_current(person=True):
  if person:
	  global current_log, num_recognized
	  current_log = {}
	  num_recognized = 0
  else:
  	global num_unrecognized
  	num_unrecognized = 0

"""

# SETUP
THRESHOLD = 15

rec_threshold = 0
unrec_threshold = 0
start_time = {}

suspicious = get_now(True)
current_match = None
transaction_id = 0
num_suspicious = 0

try:
  num_suspicious = len(os.listdir(Paths.HOME + "/images/_suspicious"))
except FileNotFoundError:
  num_suspicious = None
  warnings.warn("No \"suspicious\" directory found")

try:
  database = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="KittyCat123",
      database="LOG"
      )
  cursor = database.cursor()
except mysql.connector.errors.DatabaseError:
  warnings.warn("Database credentials missing or incorrect")

# LOGGING INIT AND HELPERS
def init():
  instructions = open(Paths.HOME + "/logs/init.sql", "r")
  for cmd in instructions:
    cursor.execute(cmd)
    database.commit()

def get_now(compare=False):
  now = datetime.now()
  if compare:
    return now
  else:
    time = datetime.strptime("{}:{}:{}".format(now.hour, now.minute, now.second), "%H:%M:%S")
    date = datetime.strptime("{}/{}/{}".format(now.day, now.month, now.year), "%d/%m/%Y")
    return date, time

def get_id(name):
  # will be filled in later
  return 00000

def verify_repeat(best_match):
  if best_match not in start_time.keys():
    return True
  else:
    return (get_now(True) - start_time[best_match]).total_seconds() > THRESHOLD

def update_rec_threshold(is_recognized):
  return rec_threshold + 1 if is_recognized else 0

def update_unrec_threshold(is_recognized):
  return unrec_threshold + 1 if not is_recognized else 0

# LOGGING FUNCTIONS
def add_transaction(student_name):
  global transaction_id, current_match, start_time

  add = "INSERT INTO Transactions (id, student_id, name_, date_, time_) VALUES ({}, {}, '{}', '{}', '{}')".format(
    transaction_id, get_id(student_name), student_name, *get_now())
  cursor.execute(add)
  database.commit()

  current_match = student_name
  start_time[current_match] = get_now(True)
  transaction_id += 1

def add_suspicious(path):
  add = "INSERT INTO Suspicious (path_to_img, date, time) VALUES ('{}', '{}', '{}')".format(path, *get_now())
  cursor.execute(add)
  database.commit()

  global transaction_id, num_suspicious, suspicious
  transaction_id += 1
  num_suspicious += 1
  suspicious = get_now(True)
  
"""