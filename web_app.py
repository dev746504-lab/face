import base64
import io
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import cv2
import gspread
import numpy as np
from deepface import DeepFace
from flask import Flask, jsonify, render_template, request
from google.oauth2.service_account import Credentials


BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
CREDENTIALS_PATH = BASE_DIR / "credentials.json"
CREDENTIALS_ENV_VAR = "GOOGLE_CREDENTIALS_JSON"
SHEET_ID = "17hAejyXdg_FlHLapTWuiuLeSkI4tFFy7_lqk5-DUAes"

app = Flask(__name__)
marked_today = set()
_sheet = None


def credentials_available():
    return bool(os.environ.get(CREDENTIALS_ENV_VAR)) or CREDENTIALS_PATH.exists()


def get_sheet():
    global _sheet
    if _sheet is None:
        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]
        env_credentials = os.environ.get(CREDENTIALS_ENV_VAR)
        if env_credentials:
            credentials = Credentials.from_service_account_info(
                json.loads(env_credentials), scopes=scopes
            )
        elif CREDENTIALS_PATH.exists():
            credentials = Credentials.from_service_account_file(
                str(CREDENTIALS_PATH), scopes=scopes
            )
        else:
            raise RuntimeError(
                f"Missing credentials: set {CREDENTIALS_ENV_VAR} or provide credentials.json"
            )
        _sheet = gspread.authorize(credentials).open_by_key(SHEET_ID).sheet1
    return _sheet


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


def extract_face_image(image):
    faces = DeepFace.extract_faces(
        img_path=image,
        detector_backend="retinaface",
        enforce_detection=False,
    )
    for face_object in faces:
        face = (face_object["face"] * 255).astype("uint8")
        if face.size == 0:
            continue
        gray_face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray_face, (200, 200))
    return None


def mark_attendance(name):
    today = datetime.now().strftime("%d/%m/%Y")
    attendance_key = f"{today}:{name}"
    if attendance_key in marked_today:
        return False

    sheet = get_sheet()
    header = sheet.row_values(1)
    if today not in header:
        sheet.update_cell(1, len(header) + 1, today)
        header = sheet.row_values(1)
    date_column = header.index(today) + 1

    names = sheet.col_values(1)
    if name not in names:
        return False
    row = names.index(name) + 1
    sheet.update_cell(row, date_column, "X")
    marked_today.add(attendance_key)
    return True


def recognize_image(image):
    results = DeepFace.find(
        img_path=image,
        db_path=str(DATASET_DIR),
        model_name="ArcFace",
        enforce_detection=False,
    )
    if not results or len(results[0]) == 0:
        return "Unknown"
    identity_path = results[0].iloc[0]["identity"]
    return Path(identity_path).parent.name


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/health")
def health():
    return jsonify({
        "ok": True,
        "dataset_exists": DATASET_DIR.exists(),
        "credentials_exists": credentials_available(),
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
