"""

"aisecurity.utils.keypad"

Keypad utils.

"""

import time
import warnings

import requests


# ---------------- INITS ----------------

# AUTOINIT
try:
	from lcd import GPIO, LCD_DEVICE
except ImportError:
	warnings.warn("Keypad not supported")


# CONFIG AND GLOBALS
CONFIG = {
	"server_address": "",  # TODO: set up (@database)
	"kiosk_num": 1,
	"rows": [16, 6, 12, 13],
	"columns": [19, 26, 21]  # [19, 20, 21]
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


def monitor(seconds):
	# FIXME:
	#  1. monitor should be asynchronous. previous asyncio code needed to be heavily refactored so best just to rewrite.
	#  2. setting the LCD_DEVICE from a non-lcd.py function is not recommended, as it could screw with the lcd code
	#  3. add a config entry for 0.021 seconds and 3 and 1 (line ~79)-- what does it do?

	LCD_DEVICE.set_message = "ID: "
	start = time.time()

	while time.time() - start < seconds:
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

	lcd_display = LCD_DEVICE.message[4:len(LCD_DEVICE.message)]
	lcd_display_length = len(LCD_DEVICE.message[4:len(LCD_DEVICE.message)])

	if button_id == 12 and lcd_display_length == 5:
		student_info = submit(lcd_display)
	elif (button_id <= 9 or button_id == 11) and lcd_display_length <= 5:
		LCD_DEVICE.set_message(LCD_DEVICE.message + button_id if button_id != 11 else "0")
	elif button_id == 10 and lcd_display_length > 0:
		LCD_DEVICE.set_message(LCD_DEVICE.message[0:lcd_display_length - 1])
