import os
import sys
import cv2
import numpy as np
import insightface_recognition as ifr

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget
from PySide6.QtGui import QImage, QPixmap, QCloseEvent
from PySide6.QtCore import QThread, Signal, QObject

import requests


class VideoThread(QThread):
    image_update = Signal(QImage)

    def __init__(self, screenSize: tuple[int, int], parent: QObject = None):
        ifr.set_default_app(root=os.path.join(os.path.expanduser("~"), ".insightface"))
        super().__init__(parent)
        self.screenSize = screenSize

    def run(self):
        self.capture = cv2.VideoCapture(0)
        self.session = requests.Session()
        lastEncoding = None

        while True:
            ret, frame = self.capture.read()
            if not ret:
                break

            print(frame.dtype)

            resized = self.adjustToScreenSize(frame)
            faces = ifr.get_faces(resized, maxNum=1)

            locations = ifr.face_locations(resized, faces=faces)
            encodings = ifr.face_encodings(resized, faces=faces)

            for top, right, bottom, left in locations:
                cv2.rectangle(resized, (left, top), (right, bottom), (0, 255, 0), 2)

            qt_image = self.cvImageToQtImage(resized)
            self.image_update.emit(qt_image)

            if encodings:
                if lastEncoding is not None:
                    similarity = ifr.face_similarity([lastEncoding], encodings[0])[0]
                    if similarity > 0.65:
                        continue

                response = self.session.post("http://192.168.2.67:8080/api/recognize/", data=encodings[0].tobytes())
                lastEncoding = encodings[0]
                print(response.json())

            else:
                lastEncoding = None

    def stop(self):
        self.capture.release()
        self.session.close()
        self.quit()

    def adjustToScreenSize(self, image: np.ndarray):
        hi, wi = image.shape[:2]
        ws, hs = self.screenSize
        ri, rs = wi / hi, ws / hs

        wn = int(wi * hs / hi) if (rs > ri) else ws
        hn = hs if (rs > ri) else int(hi * ws / wi)
        wn, hn = max(wn, 1), max(hn, 1)

        if (wn * hn) < (wi * hi):
            return cv2.resize(image, (wn, hn), interpolation=cv2.INTER_AREA)
        return cv2.resize(image, (wn, hn), interpolation=cv2.INTER_LINEAR)

    def cvImageToQtImage(self, image: np.ndarray):
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgbImage.shape
        bytesPerLine = ch * w
        qtImage = QImage(rgbImage.data, w, h, bytesPerLine, QImage.Format.Format_RGB888)
        return qtImage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-Time Face Recognition")
        self.image_label = QLabel()
        self.image_label.setFixedSize(640, 480)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.v_thread = VideoThread((640, 480))
        self.v_thread.image_update.connect(self.update_image)
        self.v_thread.start()

    def update_image(self, qt_image: QImage):
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))

    def closeEvent(self, event: QCloseEvent):
        self.v_thread.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
