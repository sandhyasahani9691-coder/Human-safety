import platform
from datetime import datetime
from flask import Flask, render_template, jsonify, request, Response
import database
from camera_service import camera_manager

app = Flask(__name__)

SESSION_START_TIME = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Ensure SQLite database and tables exist as soon as the app starts
database.init_db()


@app.route("/")
def dashboard():
    """Renders the main Human Safety AI Dashboard."""
    return render_template("dashboard.html", active_page="dashboard")


@app.route("/incidents")
def incidents_page():
    """Renders the Incident History & Management page."""
    search_query = request.args.get("q", "").strip()
    status_filter = request.args.get("status", "ALL").strip().upper()
    incidents = database.get_all_incidents(search_query, status_filter)
    return render_template(
        "incidents.html",
        active_page="incidents",
        incidents=incidents,
        search_query=search_query,
        status_filter=status_filter
    )


@app.route("/camera")
def camera_page():
    """Renders the Camera & Explicit Photo Capture page."""
    cam_status = camera_manager.get_status()
    recent_photos = database.get_recent_photos(limit=4)
    return render_template(
        "camera.html",
        active_page="camera",
        camera_active=cam_status["active"],
        recent_photos=recent_photos
    )


@app.route("/api/status", methods=["GET"])
def system_status():
    """Returns real-time backend health, camera state, and SQLite counts as JSON."""
    counts = database.get_incident_counts()
    cam_status = camera_manager.get_status()
    return jsonify({
        "status": "ONLINE",
        "backend": "Flask (Python " + platform.python_version() + ")",
        "database": f"SQLite ({counts['open']} Open | {counts.get('photos', 0)} Photos)",
        "camera_state": cam_status["state_label"],
        "camera_active": cam_status["active"],
        "microphone_state": "INACTIVE",
        "mode": "PROTOTYPE_SIMULATION",
        "session_started": SESSION_START_TIME,
        "simulated_incidents_total": counts["total"],
        "open_incidents": counts["open"],
        "photos_total": counts.get("photos", 0)
    })


@app.route("/api/camera/start", methods=["POST"])
def start_camera_route():
    """Explicitly starts the OpenCV webcam stream."""
    success, message = camera_manager.start_camera()
    if success:
        print("[CAMERA] Webcam started by explicit user request.")
        return jsonify({"success": True, "active": True, "message": message})
    return jsonify({"success": False, "active": False, "error": message}), 500


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera_route():
    """Explicitly stops the OpenCV webcam stream and releases hardware."""
    success, message = camera_manager.stop_camera()
    print("[CAMERA] Webcam stopped and hardware released.")
    return jsonify({"success": success, "active": False, "message": message})


@app.route("/api/camera/stream")
def camera_stream_route():
    """Streams live MJPEG frames only while the camera is explicitly active."""
    if not camera_manager.get_status()["active"]:
        return Response("Camera is inactive.", status=204)
    return Response(
        camera_manager.generate_mjpeg_stream(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/camera/capture", methods=["POST"])
def capture_photo_route():
    """Captures a single photo on explicit user click and logs metadata to SQLite."""
    data = request.get_json(silent=True) or {}
    label = data.get("label", "").strip()

    if not label:
        label = "Manual User Capture"
    if len(label) > 100:
        return jsonify({"success": False, "error": "Photo label must be 100 characters or less."}), 400

    success, filename, message = camera_manager.capture_photo()
    if not success:
        return jsonify({"success": False, "error": message}), 400

    record = database.create_photo_record(filename=filename, label=label)
    print(f"[PHOTO CAPTURE] Saved {record['photo_code']} -> static/captures/{filename} | Label: {label}")

    return jsonify({
        "success": True,
        "photo_code": record["photo_code"],
        "filename": record["filename"],
        "label": record["label"],
        "timestamp": record["timestamp"],
        "image_url": f"/static/captures/{record['filename']}",
        "message": message
    })


@app.route("/api/simulate-sos", methods=["POST"])
def trigger_simulated_sos():
    """Validates input and logs a simulated SOS incident into SQLite."""
    data = request.get_json(silent=True) or {}
    reason = data.get("reason", "").strip()
    incident_type = data.get("incident_type", "SIMULATED_SOS").strip()

    if not reason:
        reason = "Manual Dashboard SOS Trigger (No note entered)"
    if len(reason) > 150:
        return jsonify({"success": False, "error": "Reason must be 150 characters or less."}), 400

    allowed_types = ["SIMULATED_SOS", "SAFETY_CHECK", "SUSPICIOUS_ACTIVITY_LOG", "SYSTEM_TEST"]
    if incident_type not in allowed_types:
        incident_type = "SIMULATED_SOS"

    record = database.create_incident(incident_type=incident_type, reason=reason)
    print(f"[SQLITE LOG] Created {record['incident_code']} ({record['incident_type']}) at {record['timestamp']}")

    return jsonify({
        "success": True,
        "incident_id": record["incident_code"],
        "timestamp": record["timestamp"],
        "type": record["incident_type"],
        "reason": record["reason"],
        "message": "Saved permanently to local SQLite database (safety_records.db)."
    })


@app.route("/api/incidents/<int:incident_id>/status", methods=["POST"])
def change_incident_status(incident_id):
    """Toggles an incident status between OPEN and RESOLVED."""
    data = request.get_json(silent=True) or {}
    new_status = data.get("status", "").upper()

    if new_status not in ("OPEN", "RESOLVED"):
        return jsonify({"success": False, "error": "Invalid status value."}), 400

    updated = database.update_incident_status(incident_id, new_status)
    if updated:
        return jsonify({"success": True, "new_status": new_status})
    return jsonify({"success": False, "error": "Incident not found."}), 404


@app.route("/api/incidents/<int:incident_id>", methods=["DELETE"])
def remove_incident(incident_id):
    """Deletes a test incident record from SQLite."""
    deleted = database.delete_incident(incident_id)
    if deleted:
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Incident not found."}), 404


if __name__ == "__main__":
    print("=== Human Safety AI Backend Starting ===")
    print("Open http://127.0.0.1:5000 in your browser")
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=True)