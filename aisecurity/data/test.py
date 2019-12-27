#from aisecurity.utils.misc import time_limit, TimeoutException

'''
try:
    with time_limit(1):
        while(True): print("hi")
except TimeoutException as e:
    print("Timed out")
'''

import cv2
from mtcnn.mtcnn import MTCNN

img = cv2.imread("parsed_images/9/kevin_xu.jpg")
mtcnn = MTCNN()
box = mtcnn.detect_faces(img)[0]['box']
print(box)
crop_img = img[int(.75 * box[0]): int(1.25 * (box[0]+box[2])),int(.75 * box[1]):int(1.25*(box[1]+box[3]))]
cv2.imshow("cropped", crop_img)
cv2.waitKey(0)

#input("Press enter")
#print("yay")
import numpy as np

cap = cv2.VideoCapture('/Users/michaelpilarski/Desktop/movie.mov')
while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()