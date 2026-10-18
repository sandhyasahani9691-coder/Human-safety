import os
import platform
from datetime import datetime
from flask import Flask, render_template, jsonify, request, Response
import database
from camera_service import camera_manager, CAPTURE_FOLDER

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
    """Renders the Camera, YOLOv8 Street Surveillance & Explicit Photo Capture page."""
    cam_status = camera_manager.get_status()
    recent_photos = database.get_recent_photos(limit=4)
    return render_template(
        "camera.html",
        active_page="camera",
        camera_active=cam_status["active"],
        ai_enabled=cam_status["ai_enabled"],
        engine_name=cam_status["engine_name"],
        recent_photos=recent_photos
    )


@app.route("/photos")
def photos_page():
    """Renders the Dedicated Photo Records Viewer & Gallery page."""
    search_query = request.args.get("q", "").strip()
    photos = database.get_all_photos(search_query)
    return render_template(
        "photos.html",
        active_page="photos",
        photos=photos,
        search_query=search_query
    )


@app.route("/contacts")
def contacts_page():
    """Renders the Emergency Contacts & Consent-Based Location Sharing page (Milestone 7)."""
    contacts = database.get_emergency_contacts()
    return render_template(
        "contacts.html",
        active_page="contacts",
        contacts=contacts
    )


@app.route("/api/status", methods=["GET"])
def system_status():
    """Returns real-time backend health, YOLOv8 state, and SQLite counts as JSON."""
    counts = database.get_incident_counts()
    cam_status = camera_manager.get_status()
    return jsonify({
        "status": "ONLINE",
        "backend": "Flask (Python " + platform.python_version() + ")",
        "database": f"SQLite ({counts['open']} Open | {counts.get('contacts', 0)} Contacts)",
        "camera_state": cam_status["state_label"],
        "camera_active": cam_status["active"],
        "microphone_state": "EXPLICIT BROWSER CONTROL",
        "mode": "PROTOTYPE_SIMULATION",
        "session_started": SESSION_START_TIME,
        "simulated_incidents_total": counts["total"],
        "open_incidents": counts["open"],
        "photos_total": counts.get("photos", 0),
        "contacts_total": counts.get("contacts", 0)
    })


# ==========================================
# CAMERA, YOLOv8 & PHOTO API ROUTES
# ==========================================

@app.route("/api/camera/start", methods=["POST"])
def start_camera_route():
    """Explicitly starts the OpenCV webcam stream."""
    success, message = camera_manager.start_camera()
    if success:
        print(f"[CAMERA] {message}")
        return jsonify({"success": True, "active": True, "message": message})
    return jsonify({"success": False, "active": False, "error": message}), 500


@app.route("/api/camera/stop", methods=["POST"])
def stop_camera_route():
    """Explicitly stops the OpenCV webcam stream and releases hardware."""
    success, message = camera_manager.stop_camera()
    print("[CAMERA] Webcam stopped and hardware released.")
    return jsonify({"success": success, "active": False, "ai_enabled": False, "message": message})


@app.route("/api/camera/ai-toggle", methods=["POST"])
def toggle_ai_detection_route():
    """Explicitly turns YOLOv8 Street Surveillance & Gesture Detection ON or OFF."""
    data = request.get_json(silent=True) or {}
    enable_flag = bool(data.get("enabled", False))

    success, message = camera_manager.set_ai_detection(enable_flag)
    if not success:
        return jsonify({"success": False, "error": message}), 400

    print(f"[YOLOv8 VISION] Street Surveillance set to {enable_flag}")
    return jsonify({
        "success": True,
        "ai_enabled": enable_flag,
        "message": message
    })


@app.route("/api/camera/ai-telemetry", methods=["GET"])
def ai_telemetry_route():
    """Returns real-time YOLOv8 person count, confidence score, and gesture status."""
    return jsonify(camera_manager.get_ai_telemetry())


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


@app.route("/api/photos/<int:photo_id>/label", methods=["POST"])
def update_photo_label_route(photo_id):
    """Updates the label of an existing photo record in SQLite."""
    data = request.get_json(silent=True) or {}
    new_label = data.get("label", "").strip()

    if not new_label:
        return jsonify({"success": False, "error": "Label cannot be empty."}), 400
    if len(new_label) > 100:
        return jsonify({"success": False, "error": "Label must be 100 characters or less."}), 400

    updated = database.update_photo_label(photo_id, new_label)
    if updated:
        return jsonify({"success": True, "label": new_label})
    return jsonify({"success": False, "error": "Photo record not found."}), 404


@app.route("/api/photos/<int:photo_id>", methods=["DELETE"])
def delete_photo_route(photo_id):
    """Deletes a photo record from SQLite AND deletes the local .jpg file safely."""
    filename = database.delete_photo_record(photo_id)
    if not filename:
        return jsonify({"success": False, "error": "Photo record not found."}), 404

    safe_filename = os.path.basename(filename)
    file_path = os.path.join(CAPTURE_FOLDER, safe_filename)
    if os.path.exists(file_path):
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"[WARNING] Could not delete file {file_path}: {e}")

    print(f"[PHOTO DELETE] Removed record #{photo_id} and file {safe_filename}")
    return jsonify({"success": True})


# ==========================================
# EMERGENCY CONTACTS API ROUTES (Milestone 7)
# ==========================================

@app.route("/api/contacts", methods=["POST"])
def add_contact_route():
    """Validates and saves a trusted emergency contact to SQLite."""
    data = request.get_json(silent=True) or {}
    name = data.get("name", "").strip()
    relation = data.get("relation", "").strip()
    phone = data.get("phone", "").strip()

    if not name or not phone:
        return jsonify({"success": False, "error": "Contact Name and Phone are required."}), 400
    if len(name) > 60 or len(relation) > 40 or len(phone) > 25:
        return jsonify({"success": False, "error": "Input exceeds maximum length."}), 400

    contact = database.add_emergency_contact(name, relation or "Trusted Contact", phone)
    print(f"[CONTACT ADDED] {contact['name']} ({contact['relation']}) -> {contact['phone']}")
    return jsonify({"success": True, "contact": contact})


@app.route("/api/contacts/<int:contact_id>", methods=["DELETE"])
def delete_contact_route(contact_id):
    """Deletes an emergency contact from SQLite."""
    deleted = database.delete_emergency_contact(contact_id)
    if deleted:
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "Contact not found."}), 404


# ==========================================
# INCIDENT & SIMULATED DISPATCH API ROUTES
# ==========================================

@app.route("/api/simulate-sos", methods=["POST"])
def trigger_simulated_sos():
    """Logs a simulated SOS incident with optional location consent and returns simulated dispatch summary."""
    data = request.get_json(silent=True) or {}
    reason = data.get("reason", "").strip()
    incident_type = data.get("incident_type", "SIMULATED_SOS").strip()
    location_info = data.get("location_info", "").strip()

    if not reason:
        reason = "Manual Dashboard SOS Trigger (No note entered)"
    if len(reason) > 150:
        return jsonify({"success": False, "error": "Reason must be 150 characters or less."}), 400

    if not location_info:
        location_info = "Location Not Shared (Privacy Lock)"
    elif len(location_info) > 120:
        location_info = location_info[:120]

    allowed_types = [
        "SIMULATED_SOS",
        "VOICE_SOS",
        "GESTURE_X_SOS",
        "SAFETY_CHECK",
        "SUSPICIOUS_ACTIVITY_LOG",
        "SYSTEM_TEST"
    ]
    if incident_type not in allowed_types:
        incident_type = "SIMULATED_SOS"

    record = database.create_incident(
        incident_type=incident_type,
        reason=reason,
        location_info=location_info
    )

    # Fetch trusted contacts to build a Simulated Dispatch Preview (no real SMS sent)
    contacts = database.get_emergency_contacts()
    recipient_names = [f"{c['name']} ({c['phone']})" for c in contacts]
    dispatch_summary = (
        f"Simulated alert prepared for {len(contacts)} contact(s): {', '.join(recipient_names)}"
        if contacts
        else "No emergency contacts configured yet (Add contacts in Menu #5)."
    )

    print(
        f"[SQLITE LOG] Created {record['incident_code']} ({record['incident_type']}) | "
        f"Loc: {location_info} | {dispatch_summary}"
    )

    return jsonify({
        "success": True,
        "incident_id": record["incident_code"],
        "timestamp": record["timestamp"],
        "type": record["incident_type"],
        "reason": record["reason"],
        "location_info": record["location_info"],
        "dispatch_summary": dispatch_summary,
        "message": "Saved to local SQLite database. Dispatch simulated locally."
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