import sys
import os
import cv2
import joblib
import torch
import numpy as np
import urllib.request

from PyQt5 import QtWidgets, QtGui, QtCore, uic
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QGraphicsScene,
    QMessageBox,
    QInputDialog,
    QLineEdit,
)
from PyQt5.QtCore import QTimer, QDateTime

from ultralytics import YOLO
from facenet_pytorch import InceptionResnetV1
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import mediapipe as mp

UI_FILE = "_vi_ocv_Copy.ui"

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def finger_angle(a, b, c):
    a = np.array([a.x, a.y, a.z])
    b = np.array([b.x, b.y, b.z])
    c = np.array([c.x, c.y, c.z])
    ba = a - b
    bc = c - b

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


class FaceTrainerApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi(UI_FILE, self)

        # UI elements
        self.Label_5 = self.findChild(QtWidgets.QLabel, "Label_5")
        if self.Label_5 is None:
            self.Label_5 = self.findChild(QtWidgets.QLabel, "label_5")

        self.detected_person = self.findChild(QtWidgets.QLabel, "detected_person")
        self.detected_finger_number = self.findChild(QtWidgets.QLabel, "detected_finger_number")
        self.blink_count = self.findChild(QtWidgets.QLabel, "blink_count")
        self.my_terminal = self.findChild(QtWidgets.QTextEdit, "my_terminal")

        if self.detected_person is not None:
            self.detected_person.setText("NA")
        if self.detected_finger_number is not None:
            self.detected_finger_number.setText("0")
        if self.blink_count is not None:
            self.blink_count.setText("0")
        if self.my_terminal is not None:
            self.my_terminal.setText("System Ready\n")

        # Graphics view
        self.graphics_view = self.findChild(QtWidgets.QGraphicsView, "graphicsView")
        self.scene = QGraphicsScene(self)
        self.graphics_view.setScene(self.scene)
        self.pixmap_item = None

        # Menu actions
        self.actionCamera_SVM.triggered.connect(self.start_camera)
        self.actionCamera_SVM_off.triggered.connect(self.stop_camera)
        self.actionLoad_SVM.triggered.connect(self.load_svm)
        self.actionLogin.triggered.connect(self.login_user)
        self.actionLogout.triggered.connect(self.logout_user)
        self.actionHow_to_use_ui.triggered.connect(self.show_how_to_use)

        # Disable until login
        self.set_logged_in_state(False)

        # Models
        print("Loading YOLO face model...")
        self.model = YOLO("yolov8n-face.pt")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading FaceNet...")
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

        # Face landmark model
        print("Loading face landmarker...")
        self.face_landmarker_path = "face_landmarker.task"
        if not os.path.exists(self.face_landmarker_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                self.face_landmarker_path
            )

        face_opts = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.face_landmarker_path),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(face_opts)

        # Hand model
        print("Loading hand landmarker...")
        self.hand_model_path = "hand_landmarker.task"
        if not os.path.exists(self.hand_model_path):
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
                self.hand_model_path
            )

        hand_opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.hand_model_path),
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_opts)

        # Timers
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self.update_time)
        self.time_timer.start(1000)

        # Runtime state
        self.cap = None
        self.svm_model = None
        self.blink_counter = 0
        self.blink_closed = False
        self.blink_threshold = 0.5

        self.last_logged_person = None
        self.last_logged_fingers = None
        self.last_logged_blinks = None

    # -------------------------------------------------
    def set_logged_in_state(self, logged):
        self.actionLogin.setEnabled(not logged)
        self.actionLogout.setEnabled(logged)
        self.actionCamera_SVM.setEnabled(logged)
        self.actionCamera_SVM_off.setEnabled(logged)
        self.actionLoad_SVM.setEnabled(logged)

    def log(self, msg):
        if self.my_terminal is not None:
            self.my_terminal.append(f"[{QtCore.QTime.currentTime().toString()}] {msg}")

    # ------User login-------------------------------------------
    def login_user(self):
        u, ok_u = QInputDialog.getText(self, "Login", "User:")
        if not ok_u:
            self.log("Login cancelled")
            return

        p, ok_p = QInputDialog.getText(self, "Login", "Password:", QLineEdit.Password)
        if not ok_p:
            self.log("Login cancelled")
            return

        if u == "Vidura" and p == "Testing123":
            self.set_logged_in_state(True)
            self.log("Login OK")
            self.statusBar().showMessage("Logged in")
            QMessageBox.information(self, "Login", "Login successful.")
        else:
            self.log("Login failed")
            self.statusBar().showMessage("Login failed")
            QMessageBox.warning(self, "Login Failed", "Invalid username or password.")

    def logout_user(self):
        reply = QMessageBox.question(
            self,
            "Confirm Logout",
            "Are you sure you want to log out?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            self.set_logged_in_state(False)
            self.stop_camera()

            self.blink_counter = 0
            self.blink_closed = False
            self.last_logged_person = None
            self.last_logged_fingers = None
            self.last_logged_blinks = None
            self.svm_model = None

            if self.detected_person is not None:
                self.detected_person.setText("NA")
            if self.detected_finger_number is not None:
                self.detected_finger_number.setText("0")
            if self.blink_count is not None:
                self.blink_count.setText("0")

            self.log("Logged out")
            self.statusBar().showMessage("Logged out")

    # -------------------------------------------------
    def show_how_to_use(self):
        QMessageBox.information(
            self,
            "How to Use UI",
            "Log in with your user name and password.\n\n"
            "Then click File -> Load Model and select your SVM model.\n\n"
            "Then under the File menu, select Camera On.\n\n"
            "Select Camera Off to close SVM."
        )
    #-------------------------------------------------
    def update_time(self):
        now = QDateTime.currentDateTime()
        formatted = now.toString("dddd, dd MMMM yyyy\nhh:mm:ss AP")
        if self.Label_5 is not None:
            self.Label_5.setText(formatted)

        self.statusBar().showMessage("Ready")

    # -------------------------------------------------
    def load_svm(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load SVM", "", "*.pkl")
        if p:
            try:
                self.svm_model = joblib.load(p)
                self.log("SVM loaded")
                self.statusBar().showMessage("SVM loaded")
            except Exception as e:
                self.log(f"SVM load failed: {e}")
                self.statusBar().showMessage("SVM load failed")

    # -------------------------------------------------
    def start_camera(self):
        if self.svm_model is None:
            self.log("Load SVM first")
            QMessageBox.warning(self, "Error", "Please load SVM model first.")
            return

        if self.timer.isActive():
            self.log("Camera already running")
            return

        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.timer.start(30)
            self.log("Camera started")
            self.statusBar().showMessage("Camera started")
        else:
            self.log("Failed to open camera")
            self.statusBar().showMessage("Failed to open camera")

    def stop_camera(self):
        if self.timer.isActive():
            self.timer.stop()

        if self.cap:
            self.cap.release()

        self.cap = None
        self.scene.clear()
        self.pixmap_item = None

        self.statusBar().showMessage("Camera stopped")

    # -------------------------------------------------
    def update_frame(self):
        if self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            return

        global_smile = 0.0
        global_mouth_open = 0.0
        global_blink = 0.0
        detected_name = "NA"
        total_fingers = 0

        # Face detection
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy() if len(results) else []

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(v) for v in box[:4]]

            # Expanded crop for blendshape detection
            w = x2 - x1
            h = y2 - y1
            pad_w = int(w * 0.4)
            pad_h = int(h * 0.4)

            xx1 = max(0, x1 - pad_w)
            yy1 = max(0, y1 - pad_h)
            xx2 = min(frame.shape[1], x2 + pad_w)
            yy2 = min(frame.shape[0], y2 + pad_h)

            face = frame[yy1:yy2, xx1:xx2]
            if face.size == 0:
                continue

            # Recognition
            name, conf = self.recognize_face(face)
            if i == 0:
                detected_name = name if conf > 50 else "Unknown"

            # Blendshape detection
            smile_score = 0.0
            mouth_open = 0.0
            blink = 0.0

            try:
                rgb = cv2.cvtColor(cv2.resize(face, (256, 256)), cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                r = self.face_landmarker.detect(mp_img)

                if r.face_blendshapes:
                    b = r.face_blendshapes[0]

                    def get_bs(name_):
                        for item in b:
                            if item.category_name == name_:
                                return item.score
                        return 0.0

                    smile_L = get_bs("mouthSmileLeft")
                    smile_R = get_bs("mouthSmileRight")
                    mouth_open = get_bs("jawOpen")
                    blink_L = get_bs("eyeBlinkLeft")
                    blink_R = get_bs("eyeBlinkRight")

                    smile_score = (smile_L + smile_R) / 2.0
                    blink = (blink_L + blink_R) / 2.0

            except Exception:
                pass

            if i == 0:
                global_smile = smile_score
                global_mouth_open = mouth_open
                global_blink = blink

            # Draw face box and overlays
            color = (0, 255, 0) if conf > 50 else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label1 = f"{name} {conf:.1f}%"
            label2 = f"Smile:{smile_score:.2f} Blink:{blink:.2f}"
            label3 = f"Mouth Open:{mouth_open:.2f}"

            text_y1 = max(20, y1 - 35)
            text_y2 = max(40, y1 - 15)
            text_y3 = min(frame.shape[0] - 10, y2 + 20)

            cv2.putText(
                frame,
                label1,
                (x1, text_y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )
            cv2.putText(
                frame,
                label2,
                (x1, text_y2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (0, 255, 255),
                2
            )
            cv2.putText(
                frame,
                label3,
                (x1, text_y3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.50,
                (255, 255, 0),
                2
            )

        # Blink logic
        if global_blink > self.blink_threshold and not self.blink_closed:
            self.blink_counter += 1
            self.blink_closed = True
        elif global_blink <= self.blink_threshold:
            self.blink_closed = False

        if self.blink_count is not None:
            self.blink_count.setText(str(self.blink_counter))

        # Hands
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hres = self.hand_detector.detect(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            )

            if hres.hand_landmarks:
                h, w, _ = frame.shape
                for lm in hres.hand_landmarks:
                    fingers = 0
                    wrist = np.array([lm[0].x, lm[0].y, lm[0].z])

                    for a, b, c in [(2, 3, 4), (6, 7, 8), (10, 11, 12), (14, 15, 16), (18, 19, 20)]:
                        ang = finger_angle(lm[a], lm[b], lm[c])
                        tip = np.array([lm[c].x, lm[c].y, lm[c].z])
                        pip = np.array([lm[b].x, lm[b].y, lm[b].z])

                        if ang > 160 and np.linalg.norm(tip - wrist) > np.linalg.norm(pip - wrist):
                            fingers += 1

                    total_fingers += fingers

                    points = []
                    for lm_i in lm:
                        cx = int(lm_i.x * w)
                        cy = int(lm_i.y * h)
                        points.append((cx, cy))
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)

                    for c0, c1 in HAND_CONNECTIONS:
                        cv2.line(frame, points[c0], points[c1], (0, 255, 0), 2)

                    wx = int(lm[0].x * w)
                    wy = int(lm[0].y * h)
                    cv2.putText(
                        frame,
                        f"Fingers: {fingers}",
                        (wx, wy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2
                    )
        except Exception:
            pass

        # Update UI labels
        if self.detected_person is not None:
            self.detected_person.setText(detected_name)
        if self.detected_finger_number is not None:
            self.detected_finger_number.setText(str(total_fingers))

        # Logs
        if detected_name != self.last_logged_person:
            self.log(f"Person: {detected_name}")
            self.last_logged_person = detected_name

        if total_fingers != self.last_logged_fingers:
            self.log(f"Fingers: {total_fingers}")
            self.last_logged_fingers = total_fingers

        if self.blink_counter != self.last_logged_blinks:
            self.log(f"Blinks: {self.blink_counter}")
            self.last_logged_blinks = self.blink_counter

        # Top HUD overlay
        cv2.putText(
            frame,
            f"Smile: {global_smile:.2f} | Mouth Open: {global_mouth_open:.2f} | Blink: {global_blink:.2f} | Blinks Count: {self.blink_counter}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

        # Display in graphics view
        rgb_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qimg = QtGui.QImage(
            rgb_display.data,
            frame.shape[1],
            frame.shape[0],
            3 * frame.shape[1],
            QtGui.QImage.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qimg)

        if self.pixmap_item is None:
            self.pixmap_item = self.scene.addPixmap(pix)
            self.graphics_view.fitInView(self.scene.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        else:
            self.pixmap_item.setPixmap(pix)

        self.scene.setSceneRect(QtCore.QRectF(pix.rect()))

    # -------------------------------------------------
    def recognize_face(self, face):
        if self.svm_model is None:
            return "NA", 0.0

        try:
            f = cv2.resize(cv2.cvtColor(face, cv2.COLOR_BGR2RGB), (160, 160))
            t = torch.tensor(f / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                emb = self.resnet(t).cpu().numpy().reshape(1, -1)

            probs = self.svm_model.predict_proba(emb)[0]
            i = np.argmax(probs)
            return self.svm_model.classes_[i], probs[i] * 100
        except Exception:
            return "Unknown", 0.0

    def closeEvent(self, e):
        self.stop_camera()
        self.hand_detector.close()
        self.face_landmarker.close()
        e.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = FaceTrainerApp()
    win.show()
    sys.exit(app.exec_())