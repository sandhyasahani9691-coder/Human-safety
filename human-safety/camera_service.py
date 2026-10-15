import os
import time
import threading
from datetime import datetime
import cv2

CAPTURE_FOLDER = os.path.join("static", "captures")
os.makedirs(CAPTURE_FOLDER, exist_ok=True)


class CameraManager:
    """Manages explicit start/stop webcam access and frame capture via OpenCV."""

    def __init__(self):
        self.cap = None
        self.is_running = False
        self.latest_frame = None
        self.lock = threading.Lock()

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
        """Explicitly releases the webcam hardware and turns off the indicator light."""
        with self.lock:
            self.is_running = False
            self.latest_frame = None
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return True, "Camera stopped and hardware released."

    def get_status(self) -> dict:
        """Returns whether the webcam is currently active or locked."""
        return {
            "active": self.is_running,
            "state_label": "ACTIVE (Live Preview On)" if self.is_running else "INACTIVE (Hardware Locked)"
        }

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
                now_str = datetime.now().strftime("LIVE PREVIEW | %Y-%m-%d %H:%M:%S")
                cv2.putText(
                    preview_frame,
                    now_str,
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (56, 189, 248),
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