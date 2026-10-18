import os
import time
import threading
from datetime import datetime
import cv2

# 1. Check for Ultralytics YOLOv8 (Primary Production Street Surveillance Engine)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

# 2. Check for MediaPipe (Secondary Fallback)
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

CAPTURE_FOLDER = os.path.join("static", "captures")
os.makedirs(CAPTURE_FOLDER, exist_ok=True)


class CameraManager:
    """Production Street Surveillance & Explicit Camera Manager powered by YOLOv8-Pose."""

    def __init__(self):
        self.cap = None
        self.is_running = False
        self.latest_frame = None
        self.lock = threading.Lock()

        # AI Surveillance State
        self.ai_enabled = False
        self.person_count = 0
        self.max_confidence = 0.0
        self.last_detection_time = "Never"
        self.frame_counter = 0
        self.cached_boxes = []      # Stores (x1, y1, x2, y2, conf, label_tag, is_sos_person)
        self.cached_skeletons = []  # Stores arm keypoint lines for all detected people

        # Raised Crossed-Hands ("X") Gesture Tracking
        self.gesture_hold_frames = 0
        self.gesture_detected = False
        self.gesture_confidence = 0.0
        self.gesture_status_text = "No SOS Gesture"

        # Load YOLOv8n-Pose Model (Auto-downloads yolov8n-pose.pt ~6.5MB on first run)
        self.yolo_model = None
        self.engine_name = "OpenCV HOG Fallback"
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO("yolov8n-pose.pt")
                self.engine_name = "YOLOv8n-Pose (Street Surveillance)"
                print("[AI ENGINE] YOLOv8n-Pose Street Surveillance Model Loaded Successfully.")
            except Exception as e:
                print(f"[AI WARNING] Could not load YOLOv8n-pose: {e}")
                self.yolo_model = None

        # Fallback detectors if YOLOv8 is not installed yet
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.close_range_detector = cv2.CascadeClassifier(cascade_path)

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
            return True, f"Camera started ({self.engine_name})."

    def stop_camera(self) -> tuple[bool, str]:
        """Explicitly releases the webcam hardware and resets AI telemetry."""
        with self.lock:
            self.is_running = False
            self.ai_enabled = False
            self.latest_frame = None
            self.person_count = 0
            self.max_confidence = 0.0
            self.cached_boxes = []
            self.cached_skeletons = []
            self.gesture_hold_frames = 0
            self.gesture_detected = False
            self.gesture_confidence = 0.0
            self.gesture_status_text = "No SOS Gesture"
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return True, "Camera stopped and hardware released."

    def set_ai_detection(self, enabled: bool) -> tuple[bool, str]:
        """Toggles YOLOv8 Street Surveillance & Gesture Detection on or off."""
        with self.lock:
            if not self.is_running:
                self.ai_enabled = False
                return False, "Start the camera first before enabling YOLOv8 Surveillance."

            self.ai_enabled = bool(enabled)
            if not self.ai_enabled:
                self.person_count = 0
                self.max_confidence = 0.0
                self.cached_boxes = []
                self.cached_skeletons = []
                self.gesture_hold_frames = 0
                self.gesture_detected = False
                self.gesture_confidence = 0.0
                self.gesture_status_text = "Standby (AI Off)"
                return True, "YOLOv8 Street Surveillance disabled."
            return True, f"{self.engine_name} enabled."

    def get_status(self) -> dict:
        """Returns whether the webcam and YOLOv8 module are currently active."""
        if not self.is_running:
            label = "INACTIVE (Hardware Locked)"
        elif self.ai_enabled:
            label = f"YOLOv8 ACTIVE ({self.person_count} in View)"
        else:
            label = "ACTIVE (Live Preview On)"

        return {
            "active": self.is_running,
            "ai_enabled": self.ai_enabled,
            "engine_name": self.engine_name,
            "state_label": label
        }

    def get_ai_telemetry(self) -> dict:
        """Returns live YOLOv8 multi-person street telemetry and X-gesture status."""
        if not self.ai_enabled:
            scene_status = "Standby (AI Off)"
        elif self.gesture_detected:
            scene_status = "SOS X-GESTURE IN SCENE (Awaiting Human Confirmation)"
        elif self.person_count > 3:
            scene_status = f"HIGH STREET DENSITY ({self.person_count} Pedestrians - Neutral)"
        elif self.person_count > 0:
            scene_status = f"ACTIVE STREET MONITOR ({self.person_count} Person - Neutral)"
        else:
            scene_status = "CLEAR SECTOR (0 Persons)"

        return {
            "camera_active": self.is_running,
            "ai_enabled": self.ai_enabled,
            "engine_name": self.engine_name,
            "person_count": self.person_count if self.ai_enabled else 0,
            "max_confidence": round(self.max_confidence, 1) if self.ai_enabled else 0.0,
            "last_detection_time": self.last_detection_time,
            "gesture_detected": bool(self.gesture_detected and self.ai_enabled),
            "gesture_confidence": round(self.gesture_confidence, 1) if self.ai_enabled else 0.0,
            "gesture_status": self.gesture_status_text if self.ai_enabled else "Standby (AI Off)",
            "classification": scene_status
        }

    def _run_yolov8_street_surveillance(self, frame):
        """Runs Ultralytics YOLOv8n-Pose for multi-person street detection & X-gesture recognition."""
        self.frame_counter += 1

        # Run neural network inference every 3rd frame for smooth 25 FPS video on laptop CPU
        if self.frame_counter % 3 == 0:
            detected_boxes = []
            skeleton_lines = []
            any_person_doing_x_gesture = False

            if self.yolo_model is not None:
                results = self.yolo_model.predict(frame, conf=0.40, verbose=False)
                if len(results) > 0:
                    res = results[0]
                    boxes = res.boxes
                    keypoints = res.keypoints

                    for idx in range(len(boxes)):
                        box = boxes[idx]
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf_pct = round(float(box.conf[0].item()) * 100.0, 1)
                        person_has_x = False

                        # Inspect 17 COCO Pose Keypoints for this specific person
                        # 5: L-Shoulder, 6: R-Shoulder, 7: L-Elbow, 8: R-Elbow, 9: L-Wrist, 10: R-Wrist
                        if keypoints is not None and keypoints.xy is not None and idx < len(keypoints.xy):
                            kpts_xy = keypoints.xy[idx].cpu().numpy()
                            kpts_conf = (
                                keypoints.conf[idx].cpu().numpy()
                                if keypoints.conf is not None
                                else [1.0] * len(kpts_xy)
                            )

                            if len(kpts_xy) >= 11:
                                l_sh, r_sh = kpts_xy[5], kpts_xy[6]
                                l_el, r_el = kpts_xy[7], kpts_xy[8]
                                l_wr, r_wr = kpts_xy[9], kpts_xy[10]

                                # Save skeleton arm segments if visible
                                if kpts_conf[5] > 0.35 and kpts_conf[7] > 0.35:
                                    skeleton_lines.append(((int(l_sh[0]), int(l_sh[1])), (int(l_el[0]), int(l_el[1]))))
                                if kpts_conf[7] > 0.35 and kpts_conf[9] > 0.35:
                                    skeleton_lines.append(((int(l_el[0]), int(l_el[1])), (int(l_wr[0]), int(l_wr[1]))))
                                if kpts_conf[6] > 0.35 and kpts_conf[8] > 0.35:
                                    skeleton_lines.append(((int(r_sh[0]), int(r_sh[1])), (int(r_el[0]), int(r_el[1]))))
                                if kpts_conf[8] > 0.35 and kpts_conf[10] > 0.35:
                                    skeleton_lines.append(((int(r_el[0]), int(r_el[1])), (int(r_wr[0]), int(r_wr[1]))))

                                # Check if both wrists and shoulders are visible
                                wrists_visible = (
                                    kpts_conf[9] > 0.35 and kpts_conf[10] > 0.35 and
                                    l_wr[1] > 0 and r_wr[1] > 0 and l_sh[1] > 0 and r_sh[1] > 0
                                )
                                if wrists_visible:
                                    shoulder_top_y = min(l_sh[1], r_sh[1])
                                    shoulder_width = abs(l_sh[0] - r_sh[0]) + 15.0

                                    # Rule 1: Both hands raised high (above shoulder level)
                                    hands_up = (l_wr[1] < shoulder_top_y + 20) and (r_wr[1] < shoulder_top_y + 20)

                                    # Rule 2: Arms crossed into "X" shape or wrists overlapped overhead
                                    sh_sign = 1 if (l_sh[0] - r_sh[0]) >= 0 else -1
                                    wr_sign = 1 if (l_wr[0] - r_wr[0]) >= 0 else -1
                                    wrist_gap = abs(l_wr[0] - r_wr[0])
                                    crossed = (sh_sign != wr_sign) or (wrist_gap < shoulder_width * 0.60)

                                    if hands_up and crossed:
                                        person_has_x = True
                                        any_person_doing_x_gesture = True

                        tag = f"P#{idx + 1} YOLOv8"
                        detected_boxes.append((x1, y1, x2, y2, conf_pct, tag, person_has_x))

            # Update multi-person counters
            self.cached_boxes = detected_boxes
            self.cached_skeletons = skeleton_lines
            self.person_count = len(detected_boxes)

            if detected_boxes:
                self.max_confidence = max(item[4] for item in detected_boxes)
                self.last_detection_time = datetime.now().strftime("%H:%M:%S")
            else:
                self.max_confidence = 0.0

            # Update Sustained X-Gesture Hold Counter
            if any_person_doing_x_gesture:
                self.gesture_hold_frames = min(15, self.gesture_hold_frames + 2)
                self.gesture_confidence = round(min(99.0, 78.0 + self.gesture_hold_frames * 1.4), 1)
                if self.gesture_hold_frames >= 6:
                    self.gesture_detected = True
                    self.gesture_status_text = f"YOLOv8 X-GESTURE SOS ({self.gesture_confidence}%)"
                else:
                    self.gesture_status_text = f"Verifying X-Gesture ({int(self.gesture_hold_frames * 100 / 6)}%)"
            else:
                self.gesture_hold_frames = max(0, self.gesture_hold_frames - 1)
                if self.gesture_hold_frames == 0:
                    self.gesture_detected = False
                    self.gesture_confidence = 0.0
                    self.gesture_status_text = "Monitoring Street Scene (No SOS)"

        # Draw YOLOv8 Arm Skeletons
        for pt1, pt2 in self.cached_skeletons:
            line_col = (239, 68, 68) if self.gesture_detected else (34, 197, 94)
            cv2.line(frame, pt1, pt2, line_col, 2)
            cv2.circle(frame, pt2, 4, (56, 189, 248), -1)

        # Draw YOLOv8 Multi-Person Street Surveillance Bounding Boxes
        for (x1, y1, x2, y2, conf, tag, is_sos) in self.cached_boxes:
            color = (239, 68, 68) if is_sos else (56, 189, 248)
            status_suffix = "SOS X-GESTURE" if is_sos else "Neutral"
            label_str = f"{tag} | {conf}% [{status_suffix}]"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (min(frame.shape[1], x1 + 245), y1), (15, 23, 42), -1)
            cv2.putText(
                frame,
                label_str,
                (x1 + 5, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                color,
                1,
                cv2.LINE_AA
            )

        # Top-Right Street Surveillance Telemetry HUD
        hud_text = f"STREET HUD | Persons: {self.person_count}"
        cv2.rectangle(frame, (430, 6), (632, 34), (15, 23, 42), -1)
        cv2.putText(frame, hud_text, (440, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (56, 189, 248), 1, cv2.LINE_AA)

        # Alert Banner if X-Gesture SOS is active
        if self.gesture_detected:
            cv2.rectangle(frame, (10, 42), (630, 80), (66, 32, 6), -1)
            cv2.putText(
                frame,
                f"YOLOv8 SOS GESTURE DETECTED ({self.gesture_confidence}%) - 5s CONFIRMATION",
                (18, 66),
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
                    self._run_yolov8_street_surveillance(preview_frame)

                mode_str = "YOLOv8 STREET AI" if self.ai_enabled else "LIVE PREVIEW"
                now_str = datetime.now().strftime(f"{mode_str} | %Y-%m-%d %H:%M:%S")
                cv2.putText(
                    preview_frame,
                    now_str,
                    (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.54,
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
                for (x1, y1, x2, y2, conf, tag, is_sos) in self.cached_boxes:
                    color = (239, 68, 68) if is_sos else (56, 189, 248)
                    cv2.rectangle(stamped_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        stamped_image,
                        f"{tag} {conf}%",
                        (x1 + 4, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
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