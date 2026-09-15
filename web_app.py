import base64
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
MODEL_PATH = BASE_DIR / "face_model.yml"
LABELS_PATH = BASE_DIR / "labels.json"
# LBPH confidence is a distance (lower = better match); above this, treat as Unknown.
CONFIDENCE_THRESHOLD = 80

app = Flask(__name__)
marked_today = set()
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

_recognizer = None
_labels = None
_model_mtime = None


def decode_frame(data):
    if not data:
        raise ValueError("Missing image data")
    if "," in data:
        data = data.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(data)
        image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid image data") from error
    if image is None:
        raise ValueError("Could not decode image")
    return image


def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 0:
        return None
    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    return gray[y : y + h, x : x + w]


def extract_face_image(image):
    face = detect_face(image)
    if face is None or face.size == 0:
        return None
    return cv2.resize(face, (200, 200))


def mark_attendance(name):
    today = datetime.now().strftime("%d/%m/%Y")
    attendance_key = f"{today}:{name}"
    if attendance_key in marked_today:
        return False
    marked_today.add(attendance_key)
    return True


def get_recognizer():
    global _recognizer, _labels, _model_mtime
    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        return None, None
    mtime = MODEL_PATH.stat().st_mtime
    if _recognizer is None or mtime != _model_mtime:
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(str(MODEL_PATH))
        with open(LABELS_PATH) as label_file:
            labels = json.load(label_file)
        _recognizer, _labels, _model_mtime = recognizer, labels, mtime
    return _recognizer, _labels


def recognize_image(image):
    recognizer, labels = get_recognizer()
    if recognizer is None:
        return "Unknown"
    face = extract_face_image(image)
    if face is None:
        return "Unknown"
    label_id, confidence = recognizer.predict(face)
    if confidence > CONFIDENCE_THRESHOLD:
        return "Unknown"
    return labels.get(str(label_id), "Unknown")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "dataset_exists": DATASET_DIR.exists(),
        "model_trained": MODEL_PATH.exists(),
    })


@app.post("/api/recognize")
def recognize():
    try:
        payload = request.get_json(silent=True) or {}
        image = decode_frame(payload.get("image"))
        name = recognize_image(image)
        marked = False
        if name != "Unknown":
            marked = mark_attendance(name)
        return jsonify({"ok": True, "name": name, "marked": marked})
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 400


@app.post("/api/capture")
def capture():
    try:
        payload = request.get_json(silent=True) or {}
        name = (payload.get("name") or "").strip()
        stage = (payload.get("stage") or "front").strip()
        if not name:
            raise ValueError("Name is required")
        safe_name = Path(name).name
        safe_stage = Path(stage).name
        image = decode_frame(payload.get("image"))
        face = extract_face_image(image)
        if face is None:
            raise ValueError("No face found")
        person_dir = DATASET_DIR / safe_name
        person_dir.mkdir(parents=True, exist_ok=True)
        existing_count = len(list(person_dir.glob("*.jpg")))
        output_path = person_dir / f"{safe_stage}_{existing_count}.jpg"
        if not cv2.imwrite(str(output_path), face):
            raise RuntimeError("Could not save captured image")
        return jsonify({"ok": True, "path": str(output_path.relative_to(BASE_DIR))})
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 400


@app.post("/api/upload")
def upload():
    try:
        name = (request.form.get("name") or "").strip()
        if not name:
            raise ValueError("Name is required")
        files = request.files.getlist("files")
        if not files:
            raise ValueError("At least one image is required")
        person_dir = DATASET_DIR / Path(name).name
        person_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        for uploaded_file in files:
            image = cv2.imdecode(
                np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                continue
            face = extract_face_image(image)
            if face is None:
                continue
            output_path = person_dir / f"upload_{len(list(person_dir.glob('*.jpg')))}.jpg"
            cv2.imwrite(str(output_path), face)
            saved += 1
        return jsonify({"ok": True, "saved": saved})
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 400


@app.post("/api/train")
def train():
    try:
        result = subprocess.run(
            [sys.executable, str(BASE_DIR / "train.py")],
            cwd=BASE_DIR,
            capture_output=True,
            text=True,
            timeout=600,
        )
        output = (result.stdout + "\n" + result.stderr).strip()
        status = 200 if result.returncode == 0 else 400
        return jsonify({"ok": result.returncode == 0, "output": output}), status
    except subprocess.TimeoutExpired:
        return jsonify({"ok": False, "error": "Training timed out"}), 408
    except Exception as error:
        return jsonify({"ok": False, "error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")), debug=False)
