
"""

"log.py"

MySQL logging handling.

"""

import warnings
import os
from datetime import *

import mysql.connector

# SETUP
THRESHOLD = 5

rec_threshold = 0
unrec_threshold = 0
start_time = {}
suspicious = None
current_match = None
transaction_id = 0
num_suspicious = 0

try:
  num_suspicious = len(os.listdir(os.getenv("HOME") + "/Desktop/facial-recognition/images/_suspicious"))
except FileNotFoundError:
  num_suspicious = None
  warnings.warn("No \"suspicious\" directory found")

try:
  database = mysql.connector.connect(
      host="localhost",
      user="root",
      passwd="KittyCat123",
      database="KIOSK"
  )
  cursor = database.cursor()
except mysql.connector.errors.DatabaseError:
  warnings.warn("No MySQL database found")

# LOGGING INIT AND HELPERS
def init():
  with open("init.sql", "r") as instructions:
    for command in instructions:
      cursor.execute(command)
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
  return unrec_threshold + 1 if is_recognized else 0

# LOGGING FUNCTIONS
def add_transaction(student_name):
  global transaction_id, current_match, start_time

  add = "INSERT INTO Transactions (id, student_id, name_, date_, time_) VALUES ({}, {}, \'{}\', \'{}\', \'{}\')".format(
    transaction_id, get_id(student_name), student_name, *get_now())
  cursor.execute(add)
  database.commit()

  current_match = student_name
  start_time[current_match] = get_now(True)
  transaction_id += 1

def add_suspicious(path):
  add = "INSERT INTO Suspicious (path_to_img, date_, time_) VALUES (\'{}\', \'{}\', \'{}\')".format(path, *get_now())
  cursor.execute(add)
  database.commit()

  global transaction_id, num_suspicious, suspicious
  transaction_id += 1
  num_suspicious += 1
  suspicious = get_now(True)
