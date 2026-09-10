import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

time.sleep(1)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Không đọc được frame")
        continue

    cv2.imshow("OpenCV Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()