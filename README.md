# Human & Vehicle Safety AI — Class XII Group Project (Session 2027–28)

A privacy-first, Python-powered web interface exploring how **Computer Vision (Dual YOLOv8)**, **Speech Recognition**, and **Structured SQLite Logging** can support both **personal human safety** and **road/vehicle accident response**.

## Key Features
1. **Responsive Safety Dashboard (`/`):** Live telemetry, manual SOS reporting, and explicit Voice-Triggered SOS (`"Emergency SOS"`, `"Help me"`, `"Accident Alert"`).
2. **Dual YOLOv8 Street Surveillance (`/camera`):**
   - **Human Pose (`yolov8n-pose.pt`):** Multi-person tracking + overhead **Raised Crossed-Hands ("X") SOS Gesture** recognition and road fall detection.
   - **Vehicle Guard (`yolov8n.pt`):** Tracks `Car`, `Motorcycle/Bike`, `Bicycle`, `Bus`, and `Truck` classes and evaluates two-wheeler tip-overs and collision overlaps.
3. **5-Second Human Confirmation Cancel Timer:** Prevents false alarms by giving the user a 5-second window to cancel before logging to SQLite.
4. **Photo Evidence Gallery (`/photos`):** Explicit user-triggered photo snapshots stamped with YOLOv8 bounding boxes, search, label editing, and synchronized disk + SQLite deletion.
5. **Consent-Based Location & Trusted Contacts (`/contacts`):** Explicit GPS/Sector sharing and simulated emergency contact dispatch packets.
6. **Settings & CSV Export (`/settings`):** One-click export of SQLite incident logs to `.csv` for spreadsheet analysis.

## How to Run Locally (Windows 11)
```powershell
pip install -r requirements.txt
python app.py