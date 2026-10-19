import os
import time
import threading
from datetime import datetime
import cv2

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURE_FOLDER = os.path.join(BASE_DIR, "static", "captures")
os.makedirs(CAPTURE_FOLDER, exist_ok=True)

# COCO dataset vehicle class IDs in yolov8n.pt
VEHICLE_CLASSES = {
    1: "Bicycle",
    2: "Car",
    3: "Motorcycle/Bike",
    5: "Bus",
    7: "Truck"
}


def compute_iou(box_a, box_b) -> float:
    """Calculates Intersection-over-Union (overlap ratio) between two bounding boxes."""
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    inter_w = max(0, x_b - x_a)
    inter_h = max(0, y_b - y_a)
    inter_area = inter_w * inter_h
    if inter_area == 0:
        return 0.0

    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter_area / float(area_a + area_b - inter_area)


class CameraManager:
    """Dual YOLOv8 Street Surveillance: Human Pose + Vehicle Tracking & Accident Detection."""

    def __init__(self):
        self.cap = None
        self.is_running = False
        self.latest_frame = None
        self.lock = threading.Lock()

        # AI Surveillance State
        self.ai_enabled = False
        self.person_count = 0
        self.vehicle_count = 0
        self.max_confidence = 0.0
        self.last_detection_time = "Never"
        self.frame_counter = 0

        # Cached drawing primitives
        self.cached_person_boxes = []   # (x1, y1, x2, y2, conf, tag, is_sos)
        self.cached_vehicle_boxes = []  # (x1, y1, x2, y2, conf, v_type, is_accident)
        self.cached_skeletons = []

        # Human Raised Crossed-Hands ("X") Gesture Tracking
        self.gesture_hold_frames = 0
        self.gesture_detected = False
        self.gesture_confidence = 0.0
        self.gesture_status_text = "No SOS Gesture"

        # Vehicle Accident / Collision / Bike Tip-Over Tracking
        self.accident_hold_frames = 0
        self.accident_detected = False
        self.accident_confidence = 0.0
        self.accident_reason = "Normal Traffic / No Accident"
        self.simulated_accident_until = 0.0

        # Load Dual YOLOv8 Models (Pose for Humans + Nano Detector for Vehicles)
        self.pose_model = None
        self.vehicle_model = None
        self.engine_name = "OpenCV HOG + Cascade Fallback"

        # Built-in OpenCV Fallbacks (used if ultralytics is unavailable)
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.close_range_detector = cv2.CascadeClassifier(cascade_path)

        if YOLO_AVAILABLE:
            try:
                pose_path = os.path.join(BASE_DIR, "yolov8n-pose.pt")
                veh_path = os.path.join(BASE_DIR, "yolov8n.pt")
                self.pose_model = YOLO(pose_path)
                self.vehicle_model = YOLO(veh_path)
                self.engine_name = "Dual YOLOv8 (Human Pose + Vehicle Guard)"
                print("[AI ENGINE] Dual YOLOv8 (Human Pose + Vehicle Tracking) Loaded Successfully.")
            except Exception as e:
                print(f"[AI WARNING] Could not load YOLOv8 models, using OpenCV fallback: {e}")

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
        """Explicitly releases the webcam hardware and resets all AI telemetry."""
        with self.lock:
            self.is_running = False
            self.ai_enabled = False
            self.latest_frame = None
            self.person_count = 0
            self.vehicle_count = 0
            self.max_confidence = 0.0
            self.cached_person_boxes = []
            self.cached_vehicle_boxes = []
            self.cached_skeletons = []
            self.gesture_hold_frames = 0
            self.gesture_detected = False
            self.gesture_confidence = 0.0
            self.gesture_status_text = "No SOS Gesture"
            self.accident_hold_frames = 0
            self.accident_detected = False
            self.accident_confidence = 0.0
            self.accident_reason = "Normal Traffic / No Accident"
            self.simulated_accident_until = 0.0
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return True, "Camera stopped and hardware released."

    def set_ai_detection(self, enabled: bool) -> tuple[bool, str]:
        """Toggles Dual YOLOv8 Human + Vehicle Surveillance on or off."""
        with self.lock:
            if not self.is_running:
                self.ai_enabled = False
                return False, "Start the camera first before enabling YOLOv8 Surveillance."

            self.ai_enabled = bool(enabled)
            if not self.ai_enabled:
                self.person_count = 0
                self.vehicle_count = 0
                self.max_confidence = 0.0
                self.cached_person_boxes = []
                self.cached_vehicle_boxes = []
                self.cached_skeletons = []
                self.gesture_hold_frames = 0
                self.gesture_detected = False
                self.accident_hold_frames = 0
                self.accident_detected = False
                self.simulated_accident_until = 0.0
                return True, "Dual YOLOv8 Human & Vehicle Surveillance disabled."
            return True, f"{self.engine_name} enabled."

    def trigger_simulated_vehicle_accident(self) -> tuple[bool, str]:
        """Activates an 8-second simulated vehicle accident overlay for indoor viva demonstrations."""
        with self.lock:
            self.simulated_accident_until = time.time() + 8.0
            self.accident_detected = True
            self.accident_confidence = 94.5
            self.accident_reason = "Vehicle Collision / Bike Tip-Over Anomaly (Simulated Test)"
            return True, "Simulated vehicle accident anomaly active for 8 seconds."

    def get_status(self) -> dict:
        """Returns whether the webcam and YOLOv8 module are currently active."""
        if not self.is_running:
            label = "INACTIVE (Hardware Locked)"
        elif self.ai_enabled:
            label = f"YOLOv8 ON ({self.person_count}P | {self.vehicle_count}V)"
        else:
            label = "ACTIVE (Live Preview On)"

        return {
            "active": self.is_running,
            "ai_enabled": self.ai_enabled,
            "engine_name": self.engine_name,
            "state_label": label
        }

    def get_ai_telemetry(self) -> dict:
        """Returns live human count, vehicle count, X-gesture status, and vehicle accident status."""
        sim_active = time.time() < self.simulated_accident_until
        acc_active = (self.ai_enabled and self.accident_detected) or sim_active
        acc_reason = (
            "Vehicle Collision / Bike Tip-Over Anomaly"
            if acc_active
            else (self.accident_reason if self.ai_enabled else "Standby (AI Off)")
        )
        acc_conf = 94.5 if sim_active else (round(self.accident_confidence, 1) if self.ai_enabled else 0.0)

        if acc_active:
            scene_status = f"ALERT: {acc_reason} (Awaiting 5s Human Review)"
        elif self.ai_enabled and self.gesture_detected:
            scene_status = "ALERT: HUMAN X-GESTURE SOS (Awaiting 5s Human Review)"
        elif self.ai_enabled:
            scene_status = f"Monitoring ({self.person_count} Human, {self.vehicle_count} Vehicle — Normal)"
        else:
            scene_status = "Standby (AI Off)"

        return {
            "camera_active": self.is_running,
            "ai_enabled": self.ai_enabled,
            "engine_name": self.engine_name,
            "person_count": self.person_count if self.ai_enabled else 0,
            "vehicle_count": (max(1, self.vehicle_count) if sim_active else (self.vehicle_count if self.ai_enabled else 0)),
            "max_confidence": (max(94.5, round(self.max_confidence, 1)) if sim_active else (round(self.max_confidence, 1) if self.ai_enabled else 0.0)),
            "last_detection_time": self.last_detection_time,
            "gesture_detected": bool(self.gesture_detected and self.ai_enabled),
            "gesture_confidence": round(self.gesture_confidence, 1) if self.ai_enabled else 0.0,
            "gesture_status": self.gesture_status_text if self.ai_enabled else "Standby (AI Off)",
            "accident_detected": bool(acc_active),
            "accident_confidence": acc_conf,
            "accident_status": acc_reason,
            "classification": scene_status
        }

    def _run_dual_yolov8_surveillance(self, frame):
        """Runs YOLOv8n-Pose (Humans + X-Gesture) AND YOLOv8n (Vehicles + Accident Detection)."""
        self.frame_counter += 1
        sim_active = time.time() < self.simulated_accident_until

        if self.frame_counter % 3 == 0:
            person_boxes = []
            vehicle_boxes = []
            skeleton_lines = []
            any_x_gesture = False
            any_fallen_person = False
            all_confs = []

            # ----------------------------------------------------
            # PART 1: YOLOv8n-Pose (or OpenCV Fallback) for Humans
            # ----------------------------------------------------
            if self.pose_model is not None:
                pose_results = self.pose_model.predict(frame, conf=0.40, verbose=False)
                if len(pose_results) > 0:
                    res = pose_results[0]
                    boxes = res.boxes
                    keypoints = res.keypoints

                    for idx in range(len(boxes)):
                        box = boxes[idx]
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        conf_pct = round(float(box.conf[0].item()) * 100.0, 1)
                        all_confs.append(conf_pct)
                        person_has_x = False

                        bw = max(1, x2 - x1)
                        bh = max(1, y2 - y1)
                        if bw > bh * 1.45:
                            any_fallen_person = True

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

                                if kpts_conf[5] > 0.35 and kpts_conf[7] > 0.35:
                                    skeleton_lines.append(((int(l_sh[0]), int(l_sh[1])), (int(l_el[0]), int(l_el[1]))))
                                if kpts_conf[7] > 0.35 and kpts_conf[9] > 0.35:
                                    skeleton_lines.append(((int(l_el[0]), int(l_el[1])), (int(l_wr[0]), int(l_wr[1]))))
                                if kpts_conf[6] > 0.35 and kpts_conf[8] > 0.35:
                                    skeleton_lines.append(((int(r_sh[0]), int(r_sh[1])), (int(r_el[0]), int(r_el[1]))))
                                if kpts_conf[8] > 0.35 and kpts_conf[10] > 0.35:
                                    skeleton_lines.append(((int(r_el[0]), int(r_el[1])), (int(r_wr[0]), int(r_wr[1]))))

                                wrists_visible = (
                                    kpts_conf[9] > 0.35 and kpts_conf[10] > 0.35 and
                                    l_wr[1] > 0 and r_wr[1] > 0 and l_sh[1] > 0 and r_sh[1] > 0
                                )
                                if wrists_visible:
                                    shoulder_top_y = min(l_sh[1], r_sh[1])
                                    shoulder_width = abs(l_sh[0] - r_sh[0]) + 15.0
                                    hands_up = (l_wr[1] < shoulder_top_y + 20) and (r_wr[1] < shoulder_top_y + 20)

                                    sh_sign = 1 if (l_sh[0] - r_sh[0]) >= 0 else -1
                                    wr_sign = 1 if (l_wr[0] - r_wr[0]) >= 0 else -1
                                    wrist_gap = abs(l_wr[0] - r_wr[0])
                                    crossed = (sh_sign != wr_sign) or (wrist_gap < shoulder_width * 0.60)

                                    if hands_up and crossed:
                                        person_has_x = True
                                        any_x_gesture = True

                        person_boxes.append((x1, y1, x2, y2, conf_pct, f"Human #{idx + 1}", person_has_x))
            else:
                # OpenCV Fallback if YOLOv8 is not installed
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects, weights = self.hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.08)
                for idx, ((x, y, w, h), weight) in enumerate(zip(rects, weights)):
                    conf = round(min(96.0, max(60.0, float(weight) * 65.0)), 1)
                    all_confs.append(conf)
                    person_boxes.append((int(x), int(y), int(x + w), int(y + h), conf, f"Human #{idx + 1}", False))

                if len(person_boxes) == 0 and not self.close_range_detector.empty():
                    c_rects = self.close_range_detector.detectMultiScale(gray, 1.15, 5, minSize=(60, 60))
                    for idx, (x, y, w, h) in enumerate(c_rects):
                        all_confs.append(82.0)
                        person_boxes.append((int(x), int(y), int(x + w), int(y + h), 82.0, f"Human #{idx + 1}", False))

            # ----------------------------------------------------
            # PART 2: YOLOv8n Vehicle Tracking + Accident Heuristics
            # ----------------------------------------------------
            raw_v_boxes = []
            if self.vehicle_model is not None:
                v_results = self.vehicle_model.predict(
                    frame,
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=0.38,
                    verbose=False
                )
                if len(v_results) > 0:
                    v_boxes = v_results[0].boxes
                    for idx in range(len(v_boxes)):
                        vbox = v_boxes[idx]
                        vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0].tolist())
                        v_conf = round(float(vbox.conf[0].item()) * 100.0, 1)
                        v_cls = int(vbox.cls[0].item())
                        v_name = VEHICLE_CLASSES.get(v_cls, "Vehicle")
                        all_confs.append(v_conf)
                        raw_v_boxes.append([vx1, vy1, vx2, vy2, v_conf, v_cls, f"{v_name} #{idx + 1}", False])

            frame_accident_found = False
            detected_accident_reason = "Normal Traffic Flow"

            for i in range(len(raw_v_boxes)):
                vx1, vy1, vx2, vy2, v_conf, v_cls, v_label, _ = raw_v_boxes[i]
                vw = max(1, vx2 - vx1)
                vh = max(1, vy2 - vy1)

                # Heuristic A: Motorcycle/Bike/Bicycle Tip-Over (flat horizontal ratio)
                if v_cls in (1, 3) and (vw > vh * 1.65):
                    raw_v_boxes[i][7] = True
                    frame_accident_found = True
                    detected_accident_reason = f"{v_label} Tip-Over / Fall Detected"

                # Heuristic B: Vehicle-to-Vehicle Collision Overlap
                for j in range(i + 1, len(raw_v_boxes)):
                    iou_v = compute_iou((vx1, vy1, vx2, vy2), raw_v_boxes[j][:4])
                    if iou_v > 0.42:
                        raw_v_boxes[i][7] = True
                        raw_v_boxes[j][7] = True
                        frame_accident_found = True
                        detected_accident_reason = f"Collision Overlap: {v_label} & {raw_v_boxes[j][6]}"

                # Heuristic C: Vehicle Overlap with Fallen Pedestrian
                for (px1, py1, px2, py2, _, p_tag, _) in person_boxes:
                    iou_p = compute_iou((vx1, vy1, vx2, vy2), (px1, py1, px2, py2))
                    if iou_p > 0.35 and any_fallen_person:
                        raw_v_boxes[i][7] = True
                        frame_accident_found = True
                        detected_accident_reason = f"Pedestrian-Vehicle Impact Alert ({v_label})"

            for item in raw_v_boxes:
                vehicle_boxes.append((item[0], item[1], item[2], item[3], item[4], item[6], item[7]))

            # If indoor Simulate Vehicle Crash button is active and no real vehicle is in room, add demo box
            if sim_active and len(vehicle_boxes) == 0:
                vehicle_boxes.append((140, 170, 500, 410, 94.5, "Simulated Bike/Car Crash", True))
                all_confs.append(94.5)

            # Save state
            self.cached_person_boxes = person_boxes
            self.cached_vehicle_boxes = vehicle_boxes
            self.cached_skeletons = skeleton_lines
            self.person_count = len(person_boxes)
            self.vehicle_count = len(vehicle_boxes)
            self.max_confidence = max(all_confs) if all_confs else 0.0
            if all_confs:
                self.last_detection_time = datetime.now().strftime("%H:%M:%S")

            # Update Human X-Gesture Hold Counter
            if any_x_gesture:
                self.gesture_hold_frames = min(15, self.gesture_hold_frames + 2)
                self.gesture_confidence = round(min(99.0, 78.0 + self.gesture_hold_frames * 1.4), 1)
                if self.gesture_hold_frames >= 6:
                    self.gesture_detected = True
                    self.gesture_status_text = f"HUMAN X-GESTURE SOS ({self.gesture_confidence}%)"
                else:
                    self.gesture_status_text = f"Verifying X-Gesture ({int(self.gesture_hold_frames * 100 / 6)}%)"
            else:
                self.gesture_hold_frames = max(0, self.gesture_hold_frames - 1)
                if self.gesture_hold_frames == 0:
                    self.gesture_detected = False
                    self.gesture_confidence = 0.0
                    self.gesture_status_text = "No Human SOS Gesture"

            # Update Vehicle Accident Hold Counter
            if sim_active:
                self.accident_detected = True
                self.accident_confidence = 94.5
                self.accident_reason = "Vehicle Crash / Bike Rollover (Simulated Viva Test)"
            elif frame_accident_found:
                self.accident_hold_frames = min(15, self.accident_hold_frames + 2)
                self.accident_confidence = round(min(97.0, 76.0 + self.accident_hold_frames * 1.5), 1)
                if self.accident_hold_frames >= 5:
                    self.accident_detected = True
                    self.accident_reason = detected_accident_reason
                else:
                    self.accident_reason = f"Checking Vehicle Anomaly... ({detected_accident_reason})"
            else:
                self.accident_hold_frames = max(0, self.accident_hold_frames - 1)
                if self.accident_hold_frames == 0:
                    self.accident_detected = False
                    self.accident_confidence = 0.0
                    self.accident_reason = (
                        f"Normal Traffic ({self.vehicle_count} Vehicle)"
                        if self.vehicle_count > 0
                        else "Clear Roadway (0 Vehicles)"
                    )

        # Draw Arm Skeletons
        for pt1, pt2 in self.cached_skeletons:
            line_col = (239, 68, 68) if self.gesture_detected else (34, 197, 94)
            cv2.line(frame, pt1, pt2, line_col, 2)
            cv2.circle(frame, pt2, 4, (56, 189, 248), -1)

        # Draw Human Bounding Boxes (Cyan for Neutral, Red for SOS)
        for (x1, y1, x2, y2, conf, tag, is_sos) in self.cached_person_boxes:
            color = (239, 68, 68) if is_sos else (56, 189, 248)
            status_suffix = "SOS X-GESTURE" if is_sos else "Safe"
            label_str = f"{tag} | {conf}% [{status_suffix}]"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 22)), (min(frame.shape[1], x1 + 235), y1), (15, 23, 42), -1)
            cv2.putText(frame, label_str, (x1 + 5, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # Draw Vehicle Bounding Boxes (Amber for Normal Vehicle, Red for Accident/Crash)
        for (vx1, vy1, vx2, vy2, vconf, vtag, is_acc) in self.cached_vehicle_boxes:
            vcolor = (239, 68, 68) if is_acc else (245, 158, 11)
            vsuffix = "ACCIDENT ALERT" if is_acc else "Tracked"
            vlabel = f"{vtag} | {vconf}% [{vsuffix}]"
            cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), vcolor, 2)
            cv2.rectangle(frame, (vx1, max(0, vy1 - 22)), (min(frame.shape[1], vx1 + 280), vy1), (15, 23, 42), -1)
            cv2.putText(frame, vlabel, (vx1 + 5, max(14, vy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, vcolor, 1, cv2.LINE_AA)

        # Top-Right Street HUD (Humans + Vehicles)
        hud_text = f"HUD | Humans: {self.person_count} | Vehicles: {self.vehicle_count}"
        cv2.rectangle(frame, (365, 6), (634, 34), (15, 23, 42), -1)
        cv2.putText(frame, hud_text, (375, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (56, 189, 248), 1, cv2.LINE_AA)

        # Alert Banner if Human Gesture OR Vehicle Accident is detected
        if self.gesture_detected or self.accident_detected or sim_active:
            banner_msg = (
                f"VEHICLE ACCIDENT DETECTED: {self.accident_reason[:38]}"
                if (self.accident_detected or sim_active)
                else f"HUMAN SOS X-GESTURE DETECTED ({self.gesture_confidence}%)"
            )
            cv2.rectangle(frame, (10, 42), (630, 80), (127, 29, 29), -1)
            cv2.putText(
                frame,
                banner_msg,
                (18, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (254, 202, 202),
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

                if self.ai_enabled or (time.time() < self.simulated_accident_until):
                    self._run_dual_yolov8_surveillance(preview_frame)

                mode_str = "YOLOv8 HUMAN+VEHICLE" if self.ai_enabled else "LIVE PREVIEW"
                now_str = datetime.now().strftime(f"{mode_str} | %Y-%m-%d %H:%M:%S")
                cv2.putText(
                    preview_frame,
                    now_str,
                    (12, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
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
            if self.ai_enabled or (time.time() < self.simulated_accident_until):
                for (x1, y1, x2, y2, conf, tag, is_sos) in self.cached_person_boxes:
                    color = (239, 68, 68) if is_sos else (56, 189, 248)
                    cv2.rectangle(stamped_image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(stamped_image, f"{tag} {conf}%", (x1 + 4, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

                for (vx1, vy1, vx2, vy2, vconf, vtag, is_acc) in self.cached_vehicle_boxes:
                    vcolor = (239, 68, 68) if is_acc else (245, 158, 11)
                    cv2.rectangle(stamped_image, (vx1, vy1), (vx2, vy2), vcolor, 2)
                    cv2.putText(stamped_image, f"{vtag} {vconf}%", (vx1 + 4, max(15, vy1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, vcolor, 2, cv2.LINE_AA)

            stamp_text = datetime.now().strftime("HS-AI CAPTURE | %Y-%m-%d %H:%M:%S")
            cv2.putText(stamped_image, stamp_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 197, 94), 2, cv2.LINE_AA)

            saved = cv2.imwrite(filepath, stamped_image)
            if not saved:
                return False, "", "Failed to write image file to disk."

            return True, filename, "Photo captured and saved locally."


camera_manager = CameraManager()