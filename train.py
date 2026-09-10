import cv2
import os
import numpy as np
import json

# Google sheet
import gspread
from google.oauth2.service_account import Credentials


dataset_path = "dataset"

faces = []
labels = []
label_map = {}

label_id = 0

for person in os.listdir(dataset_path):

    person_path = os.path.join(dataset_path, person)

    if not os.path.isdir(person_path):
        continue

    label_map[label_id] = person

    for img_name in os.listdir(person_path):

        img_path = os.path.join(person_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img, (200,200))

        faces.append(img)
        labels.append(label_id)

    label_id += 1


faces = np.array(faces)
labels = np.array(labels)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, labels)

recognizer.save("face_model.yml")

# lưu label map
with open("labels.json","w") as f:
    json.dump(label_map,f)

print("Train xong model")
print(label_map)


# =========================
# UPDATE GOOGLE SHEET
# =========================

# =========================
# UPDATE GOOGLE SHEET
# =========================

scope = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

env_credentials = os.environ.get("GOOGLE_CREDENTIALS_JSON")
if env_credentials:
    creds = Credentials.from_service_account_info(
        json.loads(env_credentials), scopes=scope
    )
else:
    creds = Credentials.from_service_account_file(
        "credentials.json",
        scopes=scope
    )

client = gspread.authorize(creds)

sheet = client.open_by_key(
"17hAejyXdg_FlHLapTWuiuLeSkI4tFFy7_lqk5-DUAes"
).sheet1


# lấy danh sách hiện tại trong sheet
sheet_names = sheet.col_values(1)

# bỏ header
if "Name" in sheet_names:
    sheet_names.remove("Name")


# thêm người mới vào cuối bảng (không sort)
for name in label_map.values():

    if name not in sheet_names:

        sheet.append_row([name])

        print("Đã thêm vào Google Sheet:", name)