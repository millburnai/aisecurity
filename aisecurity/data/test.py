from aisecurity.utils.misc import time_limit, TimeoutException

try:
    with time_limit(1):
        while(True): print("hi")
except TimeoutException as e:
    print("Timed out")