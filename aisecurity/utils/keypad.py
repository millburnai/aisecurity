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
    await recognize_func(*args, **kwargs)

	lcd_display = LCD_DEVICE.message[4:len(LCD_DEVICE.message)]
	lcd_display_length = len(LCD_DEVICE.message[4:len(LCD_DEVICE.message)])

	if button_id == 12 and lcd_display_length == 5:
		student_info = submit(lcd_display)
	elif (button_id <= 9 or button_id == 11) and lcd_display_length <= 5:
		LCD_DEVICE.set_message(LCD_DEVICE.message + button_id if button_id != 11 else "0")
	elif button_id == 10 and lcd_display_length > 0:
		LCD_DEVICE.set_message(LCD_DEVICE.message[0:lcd_display_length - 1])
