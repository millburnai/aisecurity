#from aisecurity.utils.misc import time_limit, TimeoutException

'''
try:
    with time_limit(1):
        while(True): print("hi")
except TimeoutException as e:
    print("Timed out")
'''

#import cv2
#from mtcnn.mtcnn import MTCNN

import os
from aisecurity.utils.paths import CONFIG_HOME
import subprocess

def pr():
    os.chdir(CONFIG_HOME+"/bin/")
    print(os.getcwd())
    os.system('sh dump_embeds.sh "s_r0LBk5J9AAAAAAAAAAp0YbeqVBc0LJI9KBy7oapaz2Fso7qvVJedWxauwvR3rC" "/Homework/test.py" "/Users/michaelpilarski/Desktop/dump_embeds.sh"')
pr()
'''
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
'''
'''
face_cascade = cv2.CascadeClassifier('/Users/michaelpilarski/Desktop/haarcascade_frontalface_default.xml')
# Read the input image
img = cv2.imread('/Users/michaelpilarski/Desktop/julia_aronovich_copy.png')
# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(img, scaleFactor=1.1)
# Draw rectangle around the faces


for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
# Display the output
cv2.imshow('img', img)
cv2.waitKey()
'''