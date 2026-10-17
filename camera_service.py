import os
import time
import threading
from datetime import datetime
import cv2

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

CAPTURE_FOLDER = os.path.join("static", "captures")
os.makedirs(CAPTURE_FOLDER, exist_ok=True)


class CameraManager:
    """Manages explicit webcam access, AI person detection, and Raised Crossed-Hands SOS gesture."""

    def __init__(self):
        self.cap = None
        self.is_running = False
        self.latest_frame = None
        self.lock = threading.Lock()

        # Milestone 5 & 6: AI Person + Crossed-Hands Gesture State
        self.ai_enabled = False
        self.person_count = 0
        self.max_confidence = 0.0
        self.last_detection_time = "Never"
        self.frame_counter = 0
        self.cached_boxes = []

        # Milestone 6: Raised Crossed-Hands ("X") Gesture Tracking
        self.gesture_hold_frames = 0
        self.gesture_detected = False
        self.gesture_confidence = 0.0
        self.gesture_status_text = "No SOS Gesture"
        self.cached_landmarks = []  # Stores wrist/elbow/shoulder lines for drawing

        # 1. OpenCV HOG + Cascade for Person Presence
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.close_range_detector = cv2.CascadeClassifier(cascade_path)

        # 2. MediaPipe Pose for Raised Crossed-Hands Gesture
        self.pose_detector = None
        if MEDIAPIPE_AVAILABLE:
            self.mp_pose = mp.solutions.pose
            self.pose_detector = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=0,
                min_detection_confidence=0.55,
                min_tracking_confidence=0.55
            )

    def start_camera(self) -> tuple[bool, str]:
        """Explicitly opens the default laptop webcam (index 0)."""
        with self.lock:
            if self.is_running and self.cap is not None and self.cap.isOpened():
                return True, "Camera is already active."

            if os.name == "nt":
                self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            else:
                self.cap = cv2.VideoCapture(0)

            if not self.cap or not self.cap.isOpened():
                self.cap = None
                self.is_running = False
                return False, "Could not access webcam. Check if another app is using it."

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.is_running = True
            return True, "Camera started by explicit user command."

    def stop_camera(self) -> tuple[bool, str]:
        """Explicitly releases the webcam hardware and resets AI telemetry."""
        with self.lock:
            self.is_running = False
            self.ai_enabled = False
            self.latest_frame = None
            self.person_count = 0
            self.max_confidence = 0.0
            self.cached_boxes = []
            self.gesture_hold_frames = 0
            self.gesture_detected = False
            self.gesture_confidence = 0.0
            self.gesture_status_text = "No SOS Gesture"
            self.cached_landmarks = []
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return True, "Camera stopped and hardware released."

    def set_ai_detection(self, enabled: bool) -> tuple[bool, str]:
        """Toggles AI Person & Gesture Detection on or off while camera is active."""
        with self.lock:
            if not self.is_running:
                self.ai_enabled = False
                return False, "Start the camera first before enabling AI Detection."

            self.ai_enabled = bool(enabled)
            if not self.ai_enabled:
                self.person_count = 0
                self.max_confidence = 0.0
                self.cached_boxes = []
                self.gesture_hold_frames = 0
                self.gesture_detected = False
                self.gesture_confidence = 0.0
                self.gesture_status_text = "Standby (AI Off)"
                self.cached_landmarks = []
                return True, "AI Person & Gesture Detection disabled."
            return True, "AI Person & Crossed-Hands Gesture Detection enabled."

    def get_status(self) -> dict:
        """Returns whether the webcam and AI module are currently active."""
        if not self.is_running:
            label = "INACTIVE (Hardware Locked)"
        elif self.ai_enabled:
            label = f"ACTIVE + AI ON ({self.person_count} Person)"
        else:
            label = "ACTIVE (Live Preview On)"

        return {
            "active": self.is_running,
            "ai_enabled": self.ai_enabled,
            "state_label": label
        }

    def get_ai_telemetry(self) -> dict:
        """Returns live AI person count, confidence, and Raised Crossed-Hands gesture state."""
        return {
            "camera_active": self.is_running,
            "ai_enabled": self.ai_enabled,
            "person_count": self.person_count if self.ai_enabled else 0,
            "max_confidence": round(self.max_confidence, 1) if self.ai_enabled else 0.0,
            "last_detection_time": self.last_detection_time,
            "gesture_detected": bool(self.gesture_detected and self.ai_enabled),
            "gesture_confidence": round(self.gesture_confidence, 1) if self.ai_enabled else 0.0,
            "gesture_status": self.gesture_status_text if self.ai_enabled else "Standby (AI Off)",
            "mediapipe_ready": MEDIAPIPE_AVAILABLE,
            "classification": (
                "CROSSED-HANDS SOS GESTURE DETECTED (Awaiting Human Review)"
                if (self.ai_enabled and self.gesture_detected)
                else (
                    "NEUTRAL HUMAN PRESENCE (No Threat Implied)"
                    if (self.ai_enabled and self.person_count > 0)
                    else "Monitoring Standby"
                )
            )
        }

    def _evaluate_crossed_hands_gesture(self, frame):
        """Uses MediaPipe Pose to detect two hands raised & crossed in an 'X' above shoulders."""
        if not MEDIAPIPE_AVAILABLE or self.pose_detector is None:
            self.gesture_status_text = "Install mediapipe for X-Gesture"
            return

        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose_detector.process(rgb_frame)
        self.cached_landmarks = []

        if not results.pose_landmarks:
            self.gesture_hold_frames = max(0, self.gesture_hold_frames - 1)
            self.gesture_detected = False
            self.gesture_status_text = "No Upper-Body Pose"
            return

        lm = results.pose_landmarks.landmark
        # Landmark indices: 11=Left Shoulder, 12=Right Shoulder, 13=Left Elbow, 14=Right Elbow, 15=Left Wrist, 16=Right Wrist
        l_sh, r_sh = lm[11], lm[12]
        l_el, r_el = lm[13], lm[14]
        l_wr, r_wr = lm[15], lm[16]

        # Ensure person detection count reflects visible upper body
        if l_sh.visibility > 0.5 or r_sh.visibility > 0.5:
            self.person_count = max(self.person_count, 1)
            pose_conf = round(max(l_sh.visibility, r_sh.visibility) * 95.0, 1)
            self.max_confidence = max(self.max_confidence, pose_conf)

        # Save arm skeleton lines for visual feedback
        pts = [
            ((int(l_sh.x * w), int(l_sh.y * h)), (int(l_el.x * w), int(l_el.y * h))),
            ((int(l_el.x * w), int(l_el.y * h)), (int(l_wr.x * w), int(l_wr.y * h))),
            ((int(r_sh.x * w), int(r_sh.y * h)), (int(r_el.x * w), int(r_el.y * h))),
            ((int(r_el.x * w), int(r_el.y * h)), (int(r_wr.x * w), int(r_wr.y * h)))
        ]
        self.cached_landmarks = pts

        # Check visibility of both wrists
        if l_wr.visibility < 0.45 or r_wr.visibility < 0.45:
            self.gesture_hold_frames = max(0, self.gesture_hold_frames - 1)
            self.gesture_detected = False
            self.gesture_status_text = "Hands Lowered / Not Visible"
            return

        shoulder_top_y = min(l_sh.y, r_sh.y)
        shoulder_width = abs(l_sh.x - r_sh.x) + 0.05

        # Rule 1: Both wrists must be raised high (above shoulder level)
        hands_raised_up = (l_wr.y < shoulder_top_y + 0.04) and (r_wr.y < shoulder_top_y + 0.04)

        # Rule 2: Arms crossed ("X" shape) — wrists cross horizontally or meet closely above head
        wrist_x_gap = abs(l_wr.x - r_wr.x)
        # In anatomical landmarks, left shoulder x > right shoulder x in front-facing view.
        # When arms cross, (l_wr.x - r_wr.x) reverses sign compared to (l_sh.x - r_sh.x), OR wrists overlap tightly.
        shoulder_sign = 1 if (l_sh.x - r_sh.x) >= 0 else -1
        wrist_sign = 1 if (l_wr.x - r_wr.x) >= 0 else -1
        wrists_crossed = (shoulder_sign != wrist_sign) or (wrist_x_gap < shoulder_width * 0.55)

        if hands_raised_up and wrists_crossed:
            self.gesture_hold_frames = min(15, self.gesture_hold_frames + 2)
            raw_conf = min(98.0, 75.0 + (self.gesture_hold_frames * 1.5))
            self.gesture_confidence = round(raw_conf, 1)

            if self.gesture_hold_frames >= 6:
                self.gesture_detected = True
                self.gesture_status_text = f"RAISED X-GESTURE ACTIVE ({self.gesture_confidence}%)"
            else:
                self.gesture_status_text = f"Holding Raised Cross... ({int(self.gesture_hold_frames * 100 / 6)}%)"
        else:
            self.gesture_hold_frames = max(0, self.gesture_hold_frames - 1)
            if self.gesture_hold_frames == 0:
                self.gesture_detected = False
                self.gesture_confidence = 0.0
                if hands_raised_up:
                    self.gesture_status_text = "Hands Raised (Cross Arms for SOS)"
                else:
                    self.gesture_status_text = "Monitoring (No SOS Gesture)"

    def _run_person_and_gesture_detection(self, frame):
        """Runs hybrid Person Detection + Raised Crossed-Hands Gesture recognition."""
        self.frame_counter += 1
        if self.frame_counter % 3 == 0:
            detected = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # A. Full-body HOG Detection
            rects, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.08)
            for (x, y, w, h), weight in zip(rects, weights):
                conf = min(98.0, max(55.0, float(weight) * 65.0))
                if conf >= 58.0:
                    detected.append((int(x), int(y), int(w), int(h), round(conf, 1), "Full-Body"))

            # B. Close-range presence detection
            if len(detected) == 0 and not self.close_range_detector.empty():
                close_rects, _, level_weights = self.close_range_detector.detectMultiScale3(
                    gray, scaleFactor=1.15, minNeighbors=5, minSize=(60, 60), outputRejectLevels=True
                )
                for (x, y, w, h), weight in zip(close_rects, level_weights):
                    conf = min(96.0, max(65.0, 60.0 + float(weight) * 8.0))
                    pad_x, pad_y = int(w * 0.35), int(h * 0.45)
                    bx, by = max(0, int(x - pad_x)), max(0, int(y - pad_y * 0.3))
                    bw = min(frame.shape[1] - bx, int(w + pad_x * 2))
                    bh = min(frame.shape[0] - by, int(h + pad_y * 1.8))
                    detected.append((bx, by, bw, bh, round(conf, 1), "Close-Range"))

            self.cached_boxes = detected
            self.person_count = len(detected)
            if detected:
                self.max_confidence = max(item[4] for item in detected)
                self.last_detection_time = datetime.now().strftime("%H:%M:%S")
            else:
                self.max_confidence = 0.0

            # C. Evaluate Raised Crossed-Hands ("X") SOS Gesture
            self._evaluate_crossed_hands_gesture(frame)

        # Draw cached person bounding boxes
        for (x, y, w, h, conf, mode_tag) in self.cached_boxes:
            box_color = (245, 158, 11) if self.gesture_detected else (56, 189, 248)
            cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
            label_str = f"Person ({mode_tag}) | {conf}% [Neutral]"
            cv2.rectangle(frame, (x, max(0, y - 24)), (x + min(w, 290), y), (15, 23, 42), -1)
            cv2.putText(frame, label_str, (x + 5, max(15, y - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, box_color, 1, cv2.LINE_AA)

        # Draw arm skeleton lines for gesture transparency
        for pt1, pt2 in self.cached_landmarks:
            line_color = (34, 197, 94) if not self.gesture_detected else (239, 68, 68)
            cv2.line(frame, pt1, pt2, line_color, 3)
            cv2.circle(frame, pt2, 5, (56, 189, 248), -1)

        # Banner overlay when Raised Crossed-Hands SOS Gesture is detected
        if self.gesture_detected:
            cv2.rectangle(frame, (10, 42), (630, 82), (66, 32, 6), -1)
            cv2.putText(
                frame,
                f"RAISED CROSS-HANDS SOS GESTURE ({self.gesture_confidence}%) - CHECK CONFIRMATION",
                (18, 67),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (253, 230, 138),
                2,
                cv2.LINE_AA
            )

    def generate_mjpeg_stream(self):
        """Generator that yields JPEG frames for the live browser preview while active."""
        while self.is_running:
            with self.lock:
                if self.cap is None or not self.cap.isOpened():
                    break
                success, frame = self.cap.read()
                if not success:
                    break

                self.latest_frame = frame.copy()
                preview_frame = frame.copy()

                if self.ai_enabled:
                    self._run_person_and_gesture_detection(preview_frame)

                mode_str = "AI + GESTURE ON" if self.ai_enabled else "LIVE PREVIEW"
                now_str = datetime.now().strftime(f"{mode_str} | %Y-%m-%d %H:%M:%S")
                cv2.putText(
                    preview_frame,
                    now_str,
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (34, 197, 94) if self.ai_enabled else (56, 189, 248),
                    2,
                    cv2.LINE_AA
                )

                ret, buffer = cv2.imencode(".jpg", preview_frame)
                if not ret:
                    continue
                frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
            time.sleep(0.04)

    def capture_photo(self) -> tuple[bool, str, str]:
        """Saves a single JPEG image when explicitly triggered by the user."""
        with self.lock:
            if not self.is_running or self.cap is None:
                return False, "", "Camera is not active. Please click 'Start Camera' first."

            frame_to_save = self.latest_frame
            if frame_to_save is None:
                success, frame_to_save = self.cap.read()
                if not success or frame_to_save is None:
                    return False, "", "Failed to read frame from webcam."

            timestamp_slug = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
            filename = f"capture_{timestamp_slug}.jpg"
            filepath = os.path.join(CAPTURE_FOLDER, filename)

            stamped_image = frame_to_save.copy()
            if self.ai_enabled:
                for (x, y, w, h, conf, mode_tag) in self.cached_boxes:
                    cv2.rectangle(stamped_image, (x, y), (x + w, y + h), (56, 189, 248), 2)
                    cv2.putText(
                        stamped_image,
                        f"Person {conf}% [Neutral]",
                        (x + 4, max(15, y - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (56, 189, 248),
                        2,
                        cv2.LINE_AA
                    )

            stamp_text = datetime.now().strftime("HS-AI CAPTURE | %Y-%m-%d %H:%M:%S")
            cv2.putText(
                stamped_image,
                stamp_text,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (34, 197, 94),
                2,
                cv2.LINE_AA
            )

            saved = cv2.imwrite(filepath, stamped_image)
            if not saved:
                return False, "", "Failed to write image file to disk."

            return True, filename, "Photo captured and saved locally."


camera_manager = CameraManager()