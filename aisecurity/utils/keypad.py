import asyncio
import warnings
try:
	from lcd import GPIO, LCD_DEVICE
except ImportError:
    warnings.warn("Board not properly set up. Keypad will be unusable")

server_address = None 
#To be found and set up by database team
kiosk_num = 1
#Differs between kiosks

class keypad(object):
	rows = [16, 6, 12, 13]
	columns = [19, 26, 21] #[19, 20 21]

	@staticmethod
	def submit(student_id):
		r = requests.get(url = server_address, params = {"id":str(student_id), "kiosk":str(kiosk_num)})
		student_info = r.json()
		return student_info

	def current_input(lcd):
		return lcd.message[4 : len(lcd.message)]

	@staticmethod
	def init():
	    for row in rows:
	        GPIO.setup(row, GPIO.OUT)
	        GPIO.output(row, GPIO.LOW)
	    for column in columns:
	        GPIO.setup(column, GPIO.IN, pull_up_down = GPIO.PUD_DOWN)

	@staticmethod
	def monitor(seconds):
		LCD_DEVICE.set_message = "ID: "
		start = time.time()
		while (time.time() - start < seconds):
			time.sleep(0.021)
			for row in rows:
				GPIO.output(row, GPIO.HIGH)
				for column in columns:
					if(GPIO.input(column)):
						button_id = (rows.index(row) * 3) + (columns.index(column) + 1)
						press(button_id)
						start = time.time()
					time.sleep(.021)
				GPIO.output(row, GPIO.LOW)

	def press(button_id, lcd):

		if button_id == 12 and current_input(lcd) == 5:
			student_info = submit(current_input(lcd))
		else if button_id <= 9 or button_id == 11:
			LCD_DEVICE.set_message(LCD_DEVICE.message + str(button_id))
		else if button_id == 10 and current_input(lcd) > 0:
			LCD_DEVICE.set_message(LCD_DEVICE.message[0 : len(LCD_DEVICE.message) - 1])



async def async_helper(recognize_func, *args, **kwargs):
    await recognize_func(*args, **kwargs)

async def time():
	import time
	start = time.time()
	await asyncio.sleep(2)
	print(time.time()-start)

def test(func):
	loop = asyncio.new_event_loop()
	task = loop.create_task(async_helper(func))
	loop.run_until_complete(task)