import cv2
import time
from deepface import DeepFace

import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime


# =============================
# GOOGLE SHEET CONNECTION
# =============================

scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

creds = Credentials.from_service_account_file(
    "credentials.json",
    scopes=scope
)

client = gspread.authorize(creds)

sheet = client.open_by_key(
"17hAejyXdg_FlHLapTWuiuLeSkI4tFFy7_lqk5-DUAes"
).sheet1


# =============================
# TẠO / TÌM CỘT NGÀY HÔM NAY
# =============================

today = datetime.now().strftime("%d/%m/%Y")

header = sheet.row_values(1)

if today not in header:

    sheet.update_cell(1, len(header)+1, today)

    header = sheet.row_values(1)

date_col = header.index(today) + 1


# =============================
# CAMERA
# =============================

dataset_path = "dataset"

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

time.sleep(1)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

marked_today = set()

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,1.3,5)

    for (x,y,w,h) in faces:

        face_img = frame[y:y+h, x:x+w]

        try:

            result = DeepFace.find(
                img_path = face_img,
                db_path = dataset_path,
                model_name = "ArcFace",
                enforce_detection=False
            )

            if len(result[0]) > 0:

                identity_path = result[0].iloc[0]['identity']
                name = identity_path.split("/")[-2]

            else:
                name = "Unknown"

        except:
            name = "Unknown"


        # =============================
        # ATTENDANCE
        # =============================

        if name != "Unknown" and name not in marked_today:

            names = sheet.col_values(1)

            if name in names:

                row = names.index(name) + 1

                sheet.update_cell(row, date_col, "X")

                print("Attendance marked:", name)

                marked_today.add(name)


        cv2.putText(frame,name,(x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0,255,0),2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


    cv2.imshow("AI Face Recognition",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()