import sys
import cv2
import os
from deepface import DeepFace

image_path = sys.argv[1]
name = sys.argv[2]

dataset_path = f"dataset/{name}"
os.makedirs(dataset_path, exist_ok=True)

img = cv2.imread(image_path)

if img is None:
    print("Không đọc được ảnh")
    exit()

count = len(os.listdir(dataset_path))

try:
    faces = DeepFace.extract_faces(
        img_path=image_path,
        detector_backend="retinaface",
        enforce_detection=False
    )

    for face_obj in faces:

        face = face_obj["face"]

        # 🔥 FIX QUAN TRỌNG
        face = (face * 255).astype("uint8")

        # tránh lỗi ảnh rỗng
        if face is None or face.size == 0:
            continue

        # convert sang grayscale
        face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)

        face = cv2.resize(face, (200,200))

        filename = f"{dataset_path}/{count}.jpg"

        cv2.imwrite(filename, face)

        count += 1

        print("Saved:", filename)

except Exception as e:
    print("Lỗi:", e)