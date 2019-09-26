
import cv2
from mtcnn.mtcnn import MTCNN
'''
detector = MTCNN()
print("YEET")
image = cv2.imread("liam.jpg")
result = detector.detect_faces(image)
print(result)
bounding_box = result[0]['box']
keypoints = result[0]['keypoints']
cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              (0,155,255),
              2)

cv2.imwrite("liam.jpg", image)
cv2.namedWindow("image")
cv2.imshow("image",image)
cv2.waitKey(0)
'''
detector = MTCNN()
cap = cv2.VideoCapture(0)
while True: 
    __, frame = cap.read()
    result = detector.detect_faces(frame)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']
    
            cv2.rectangle(frame,
                          (bounding_box[0], bounding_box[1]),
                          (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                          (0,155,255),
                          2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
