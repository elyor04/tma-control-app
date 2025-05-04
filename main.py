import os
import sys
import cv2
import numpy as np
import insightface_recognition as ifr

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QImage, QPixmap, QCloseEvent
from PySide6.QtCore import QThread, Signal, QObject

import requests
import time

BACKEND_BASE_URL = "http://localhost:8000"


class FaceRecognition(QThread):
    image_update = Signal(QImage)

    def __init__(self, parent: QObject = None):
        super().__init__(parent)
        MODEL_ROOT = os.path.join(os.path.expanduser("~"), ".insightface")

        self.detect = ifr.get_app(
            model_name="buffalo_l",
            root=MODEL_ROOT,
            allowed_modules=["detection"],
        )
        self.recognize = ifr.get_app(
            model_name="buffalo_l",
            root=MODEL_ROOT,
            allowed_modules=["detection", "recognition"],
        )

        self.cooldown_end_time = 0
        self.face_detected_start = None
        self.last_encoding = None
        self.last_result_frame = None

    def run(self):
        self.capture = cv2.VideoCapture(0)
        self.session = requests.Session()

        if not self.capture.isOpened():
            return

        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height, width = self.getScreenSize(height, width)

        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)

        height = self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = self.capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        faceMinSize = int(height * 0.3), int(width * 0.3)

        while True:
            current_time = time.time()

            # Cooldown handling
            if current_time < self.cooldown_end_time:
                if self.last_result_frame is not None:
                    self.image_update.emit(self.cvImageToQtImage(self.last_result_frame))
                continue

            # Frame processing
            ret, original_frame = self.capture.read()
            if not ret:
                break

            frame = original_frame.copy()
            faces = ifr.get_faces(frame, maxNum=1, minSize=faceMinSize, app=self.detect)
            locations = ifr.face_locations(frame, faces=faces)

            # Draw detection rectangles
            for top, right, bottom, left in locations:
                cv2.rectangle(original_frame, (left, top), (right, bottom), (255, 0, 0), 2)

            if not faces:
                self.face_detected_start = None
                self.image_update.emit(self.cvImageToQtImage(original_frame))
                continue

            # Start face detection timer
            if self.face_detected_start is None:
                self.face_detected_start = current_time
                self.image_update.emit(self.cvImageToQtImage(original_frame))
                continue

            # Check 2-second threshold
            if (current_time - self.face_detected_start) < 2:
                self.image_update.emit(self.cvImageToQtImage(original_frame))
                continue

            # Face recognition processing
            faces = ifr.get_faces(frame, maxNum=1, minSize=faceMinSize, app=self.recognize)
            encodings = ifr.face_encodings(frame, faces=faces)

            if not encodings:
                self.face_detected_start = None
                self.image_update.emit(self.cvImageToQtImage(original_frame))
                continue

            current_encoding = encodings[0]
            response = None
            color = (255, 0, 0)  # Default blue

            # Check if same person
            if self.last_encoding is not None:
                similarity = ifr.face_similarity([self.last_encoding], current_encoding)[0]
                if similarity > 0.7:
                    color = (0, 255, 0)  # Green for recognized
                    self.cooldown_end_time = current_time + 1
                else:
                    # New person - send to API
                    try:
                        response = self.session.post(f"{BACKEND_BASE_URL}/api/recognize/", data=current_encoding.tobytes())
                    except: pass
            else:
                # First recognition - send to API
                try:
                    response = self.session.post(f"{BACKEND_BASE_URL}/api/recognize/", data=current_encoding.tobytes())
                except: pass

            # Handle API response
            if response is not None:
                if response.ok:
                    color = (0, 255, 0)  # Green
                    self.last_encoding = current_encoding
                else:
                    color = (0, 0, 255)  # Red
                    self.last_encoding = None

            # Update frame and cooldown
            for top, right, bottom, left in locations:
                cv2.rectangle(original_frame, (left, top), (right, bottom), color, 2)

            self.cooldown_end_time = current_time + 1
            self.last_result_frame = original_frame.copy()
            self.image_update.emit(self.cvImageToQtImage(original_frame))
            self.face_detected_start = None

    def stop(self):
        self.capture.release()
        self.session.close()
        self.quit()

    def cvImageToQtImage(self, image: np.ndarray):
        image = self.adjustToScreenSize(image)
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        return QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)

    def getScreenSize(self, hi: int, wi: int):
        hs, ws = self.parent().screenSize
        ri, rs = wi / hi, ws / hs

        wn = int(wi * hs / hi) if (rs > ri) else ws
        hn = hs if (rs > ri) else int(hi * ws / wi)
        wn, hn = max(wn, 1), max(hn, 1)

        return hn, wn

    def adjustToScreenSize(self, image: np.ndarray):
        hi, wi = image.shape[:2]
        hn, wn = self.getScreenSize(hi, wi)

        if (wn * hn) < (wi * hi):
            return cv2.resize(image, (wn, hn), interpolation=cv2.INTER_AREA)
        return cv2.resize(image, (wn, hn), interpolation=cv2.INTER_LINEAR)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Face Recognition")
        self.image_label = QLabel()

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.fr_thread = FaceRecognition(self)
        self.fr_thread.image_update.connect(self.update_image)
        self.fr_thread.start()

    def update_image(self, qt_image: QImage):
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event: QCloseEvent):
        self.fr_thread.stop()
        event.accept()

    @property
    def screenSize(self):
        return self.image_label.height(), self.image_label.width()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.showFullScreen()
    sys.exit(app.exec())
