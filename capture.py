import sys
import cv2
import os
import time
from deepface import DeepFace

from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QInputDialog
from PyQt6.QtCore import Qt


class FaceApp(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Face Capture")
        self.resize(300,200)

        layout = QVBoxLayout()

        self.label = QLabel("Nhấn nút để chụp khuôn mặt")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.button = QPushButton("Capture Face")
        self.button.clicked.connect(self.capture_face)

        layout.addWidget(self.label)
        layout.addWidget(self.button)

        self.setLayout(layout)


    # =========================
    # CAPTURE 1 STAGE
    # =========================
    def capture_stage(self, cap, dataset_path, stage, message):

        count = 0

        while True:

            ret, frame = cap.read()
            if not ret:
                continue

            display = frame.copy()

            # hiển thị hướng dẫn
            cv2.putText(display, message, (30,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,(0,255,255),2)

            cv2.putText(display, "Press SPACE to capture", (30,80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(0,255,0),2)

            cv2.putText(display, f"{stage}: {count}/10", (30,120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,(255,255,0),2)

            cv2.imshow("Capture Face", display)

            key = cv2.waitKey(1)

            # SPACE để chụp
            if key == 32:

                while count < 10:

                    ret, frame = cap.read()
                    if not ret:
                        continue

                    try:
                        faces = DeepFace.extract_faces(
                            img_path=frame,
                            detector_backend="retinaface",
                            enforce_detection=False
                        )

                        for face_obj in faces:

                            face = face_obj["face"]

                            # 🔥 FIX dữ liệu
                            face = (face * 255).astype("uint8")

                            if face is None or face.size == 0:
                                continue

                            face = cv2.cvtColor(face, cv2.COLOR_RGB2GRAY)
                            face = cv2.resize(face,(200,200))

                            filename = f"{dataset_path}/{stage}_{count}.jpg"

                            cv2.imwrite(filename, face)

                            count += 1

                            print(f"{stage}: {count}/10")

                            time.sleep(0.3)

                    except:
                        pass

                break

            # ESC để thoát
            if key == 27:
                return


    # =========================
    # MAIN CAPTURE
    # =========================
    def capture_face(self):

        name, ok = QInputDialog.getText(self,"Tên","Nhập tên người:")

        if not ok or name == "":
            return

        dataset_path = "dataset/" + name
        os.makedirs(dataset_path, exist_ok=True)

        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

        time.sleep(1)

        stages = [
            ("front","Nhìn thẳng vào camera"),
            ("right","Quay mặt sang phải"),
            ("left","Quay mặt sang trái"),
            ("mask","Đeo khâue trang")
        ]

        for stage,message in stages:

            self.capture_stage(cap, dataset_path, stage, message)

        cap.release()
        cv2.destroyAllWindows()

        self.label.setText("Đã lưu 40 ảnh khuôn mặt")


# =========================
# RUN APP
# =========================
app = QApplication(sys.argv)

window = FaceApp()
window.show()

sys.exit(app.exec())