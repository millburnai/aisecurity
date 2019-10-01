import mysql.connector
from datetime import *
import os
'''
import asyncio
first = datetime.now()
async def now():
  await asyncio.sleep(1)
loop = asyncio.new_event_loop()
task = loop.create_task(now())
loop.run_until_complete(task)
print((first-datetime.now()).total_seconds())

print(datetime.now().hour)
'''
yeet = "hi"
rec_threshold = 0
unrec_threshold = 0
start_time = {}
num_suspicious = len(os.listdir("/Users/michaelpilarski/Desktop/facial-recognition/images/suspicious"))

mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    passwd="KittyCat123",
    database="KIOSK"
)

mycursor = mydb.cursor()

def init():
  file = open("init.sql", 'r')
  for line in file:
    mycursor.execute(line)
    mydb.commit()

def getNow(compare=False):
  now = datetime.now()
  if compare: return now
  time = datetime.strptime("{}:{}:{}".format(now.hour, now.minute, now.second), "%H:%M:%S")
  date = datetime.strptime("{}/{}/{}".format(now.day, now.month, now.year), "%d/%m/%Y")
  return date, time

def newTransaction(transid, studentid, studentname):
  date, time = getNow()
  mycursor.execute("INSERT INTO Transactions (transid, studentid, studentname, date_, time_) VALUES ({}, {}, \'{}\', \'{}\', \'{}\')".format(transid, studentid, studentname, date, time))
  mydb.commit()

def suspiciousActivity(pathToImg):
  date, time = getNow()
  mycursor.execute("INSERT INTO Suspicious (path_to_img, date_, time_) VALUES (\'{}\', \'{}\', \'{}\')".format(pathToImg, date, time))
  mydb.commit()

suspicious = getNow(True)
