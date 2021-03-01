import sys
import cv2

sys.path.insert(1, "../")
from util.visuals import Camera


if __name__ == "__main__":
    cam = Camera()

    while True:
        _, frame = cam.read()
        cv2.imshow("cam test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

