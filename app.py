import sys
import subprocess
from PyQt6.QtWidgets import (
    QApplication, QWidget, QPushButton,
    QVBoxLayout, QLabel, QFileDialog, QInputDialog
)
from PyQt6.QtCore import Qt


class FaceApp(QWidget):

    def __init__(self):
        super().__init__()

        self.setWindowTitle("AI Face Recognition")
        self.resize(400, 350)

        layout = QVBoxLayout()
        layout.setSpacing(20)

        title = QLabel("AI FACE RECOGNITION")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("""
            font-size: 22px;
            font-weight: bold;
            color: white;
        """)

        btn_capture = QPushButton("📸 Chụp hinh anh")
        btn_capture.clicked.connect(self.capture)

        btn_upload = QPushButton("📁 Upload hinh anh")
        btn_upload.clicked.connect(self.upload_dataset)

        btn_train = QPushButton("🧠 Train Model")
        btn_train.clicked.connect(self.train)

        btn_recognize = QPushButton("🔍 Diem Danh")
        btn_recognize.clicked.connect(self.recognize)

        for btn in [btn_capture, btn_upload, btn_train, btn_recognize]:
            btn.setFixedHeight(45)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    border-radius: 10px;
                    font-size: 16px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)

        layout.addWidget(title)
        layout.addWidget(btn_capture)
        layout.addWidget(btn_upload)
        layout.addWidget(btn_train)
        layout.addWidget(btn_recognize)

        self.setLayout(layout)

        self.setStyleSheet("""
            QWidget {
                background-color: #2c3e50;
                font-family: Arial;
            }
        """)

    def capture(self):
        subprocess.run(["python3", "capture.py"])

    def train(self):
        subprocess.run(["python3", "train.py"])

    def recognize(self):
        subprocess.run(["python3", "recognize.py"])


    # =========================
    # UPLOAD DATASET
    # =========================
    def upload_dataset(self):

        name, ok = QInputDialog.getText(self, "Tên", "Nhập tên người:")

        if not ok or name == "":
            return

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Chọn ảnh",
            "",
            "Images (*.png *.jpg *.jpeg)"
        )

        if not files:
            return

        for file in files:
            subprocess.run(["python3", "upload_dataset.py", file, name])


app = QApplication(sys.argv)

window = FaceApp()
window.show()

sys.exit(app.exec())