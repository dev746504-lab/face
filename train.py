import cv2
import os
import numpy as np
import json


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