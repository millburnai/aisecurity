"""

"aisecurity.hardware.keypad"

Keypad utils.

"""

import time
import warnings

import asyncio
import requests


# ---------------- INITS ----------------

# AUTOINIT
from lcd import GPIO, LCD_DEVICE

# CONFIG AND GLOBALS
CONFIG = {
    "server_address": "",  # TODO: set up (@database)
    "kiosk_num": 1,
    "rows": [16, 6, 12, 13],
    "columns": [19, 26, 21],  # [19, 20, 21]
    "submit_id": 12,
    "delete_id": 10,
    "button_ids": [11, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    "id_length": 5,
    "buffer_size": 4,  # for "ID: " string at beginning of every entry
    "seconds_to_input": 3
}

USE_KEYPAD = None


# MANUAL INIT
def init():
    global USE_KEYPAD

    try:
        for row in CONFIG["rows"]:
            GPIO.setup(row, GPIO.OUT)
            GPIO.output(row, GPIO.LOW)

            for column in CONFIG["columns"]:
                GPIO.setup(column, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
                USE_KEYPAD = True
    except NameError:  # not sure that this is the right error... will check later
        warnings.warn("Keypad not supported")
        USE_KEYPAD = False


# ---------------- FUNCTIONS ----------------
def submit(student_id):
    params = {
        "id": str(student_id),
        "kiosk": str(CONFIG["kiosk_num"])
    }
    r = requests.get(url=CONFIG["server_address"], params=params)
    student_info = r.json()
    return student_info


async def monitor():
    # FIXME:
    #  1. monitor should be asynchronous. previous asyncio code needed to be heavily refactored so best just to rewrite.
    #  2. setting the LCD_DEVICE from a non-lcd.py function is not recommended, as it could screw with the lcd code
    #  3. add a config entry for 0.021 seconds and 3 and 1 (line ~79)-- what does it do?

    LCD_DEVICE.set_message("ID: ")
    start = time.time()

    while time.time() - start < CONFIG["seconds_to_input"]:
        time.sleep(0.021)

        for row in CONFIG["rows"]:
            GPIO.output(row, GPIO.HIGH)

            for column in CONFIG["columns"]:
                if GPIO.input(column):
                    button_id = (CONFIG["rows"].index(row) * 3) + (CONFIG["columns"].index(column) + 1)
                    press(button_id)
                    start = time.time()


                time.sleep(0.021)

            GPIO.output(row, GPIO.LOW)


def press(button_id):
    # FIXME:
    #  1. what does press do? why is it called press?
    #  2. numbers 4, 12, 5, 9, 11, 10, and 0 should go in the config dict (need to have meaningful key entries)
    #  3. what is the point of student_info? it's not returned

    lcd_display = LCD_DEVICE.message[CONFIG["buffer_size"]:len(LCD_DEVICE.message)]
    lcd_display_length = len(LCD_DEVICE.message[CONFIG["buffer_size"]:len(LCD_DEVICE.message)])

    if button_id == CONFIG["submit_id"] and lcd_display_length == CONFIG["id_length"]:
        student_info = submit(lcd_display)
    elif button_id in CONFIG["button_ids"] and lcd_display_length <= CONFIG["id_length"]:
        button_index = CONFIG["button_ids"].index(button_id)
        LCD_DEVICE.set_message(LCD_DEVICE.message + str(button_index))
    elif button_id == CONFIG["delete_id"] and lcd_display_length > 0:
        LCD_DEVICE.set_message(LCD_DEVICE.message[0:lcd_display_length - 1])
