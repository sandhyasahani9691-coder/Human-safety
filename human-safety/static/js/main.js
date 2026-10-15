document.addEventListener("DOMContentLoaded", () => {
    const refreshBtn = document.getElementById("refresh-status-btn");
    const simulateSosBtn = document.getElementById("simulate-sos-btn");
    const reasonInput = document.getElementById("sos-reason");
    const typeSelect = document.getElementById("sos-type");
    const feedbackBox = document.getElementById("sos-feedback");

    // 1. Fetch live backend telemetry (Dashboard)
    async function fetchSystemStatus() {
        const backendStatusEl = document.getElementById("backend-status");
        if (!backendStatusEl) return;

        try {
            const response = await fetch("/api/status");
            const data = await response.json();

            backendStatusEl.textContent = data.status;
            document.getElementById("backend-version").textContent = data.backend;
            document.getElementById("camera-status").textContent = data.camera_state;
            document.getElementById("db-status").textContent = data.database;
            document.getElementById("sos-count").textContent = data.simulated_incidents_total;
            document.getElementById("session-time").textContent = "Session started: " + data.session_started;
        } catch (error) {
            backendStatusEl.textContent = "OFFLINE";
            backendStatusEl.style.color = "#ef4444";
            console.error("Failed to fetch status from Python server:", error);
        }
    }

    // 2. Trigger Simulated SOS event & save to SQLite (Dashboard)
    if (simulateSosBtn) {
        simulateSosBtn.addEventListener("click", async () => {
            const reasonText = reasonInput.value.trim();
            const selectedType = typeSelect ? typeSelect.value : "SIMULATED_SOS";

            try {
                const response = await fetch("/api/simulate-sos", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        incident_type: selectedType,
                        reason: reasonText
                    })
                });
                const result = await response.json();

                if (result.success) {
                    feedbackBox.classList.remove("hidden");
                    feedbackBox.innerHTML = `
                        <strong>[${result.incident_id}] (${result.type}) Logged at ${result.timestamp}</strong><br>
                        Note: ${result.reason}<br>
                        <em>${result.message}</em> — <a href="/incidents" class="inline-link">View in Incident Logs</a>
                    `;
                    reasonInput.value = "";
                    fetchSystemStatus();
                } else {
                    alert("Validation Error: " + (result.error || "Unknown error"));
                }
            } catch (error) {
                alert("Could not reach Python backend. Make sure app.py is running in PowerShell.");
            }
        });
    }

    // 3. Handle Status Toggle (OPEN <-> RESOLVED) on Incident History page
    document.querySelectorAll(".toggle-status-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
            const incidentId = btn.getAttribute("data-id");
            const nextStatus = btn.getAttribute("data-next");

            const response = await fetch(`/api/incidents/${incidentId}/status`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ status: nextStatus })
            });
            const result = await response.json();
            if (result.success) {
                window.location.reload();
            } else {
                alert("Could not update status: " + result.error);
            }
        });
    });

    // 4. Handle Delete Incident button on Incident History page
    document.querySelectorAll(".delete-incident-btn").forEach((btn) => {
        btn.addEventListener("click", async () => {
            const incidentId = btn.getAttribute("data-id");
            const incidentCode = btn.getAttribute("data-code");

            if (!confirm(`Delete simulated record ${incidentCode} from local SQLite database?`)) {
                return;
            }

            const response = await fetch(`/api/incidents/${incidentId}`, {
                method: "DELETE"
            });
            const result = await response.json();
            if (result.success) {
                const row = document.getElementById(`row-${incidentId}`);
                if (row) row.remove();
            } else {
                alert("Could not delete record: " + result.error);
            }
        });
    });

    // ==========================================
    // 5. CAMERA & PHOTO CAPTURE CONTROLS (Milestone 3)
    // ==========================================
    const startCamBtn = document.getElementById("start-camera-btn");
    const stopCamBtn = document.getElementById("stop-camera-btn");
    const captureBtn = document.getElementById("capture-photo-btn");
    const streamImg = document.getElementById("camera-stream-img");
    const placeholder = document.getElementById("camera-placeholder");
    const camBadge = document.getElementById("cam-indicator-badge");
    const photoLabelInput = document.getElementById("photo-label");
    const captureFeedback = document.getElementById("capture-feedback");
    const recentPhotosGrid = document.getElementById("recent-photos-grid");

    if (startCamBtn && stopCamBtn) {
        startCamBtn.addEventListener("click", async () => {
            startCamBtn.disabled = true;
            startCamBtn.textContent = "Starting...";

            try {
                const response = await fetch("/api/camera/start", { method: "POST" });
                const result = await response.json();

                if (result.success) {
                    // Point image tag to MJPEG route with cache-busting timestamp
                    streamImg.src = "/api/camera/stream?t=" + new Date().getTime();
                    streamImg.classList.remove("hidden");
                    placeholder.classList.add("hidden");

                    startCamBtn.textContent = "Start Camera";
                    stopCamBtn.disabled = false;
                    captureBtn.disabled = false;

                    camBadge.textContent = "CAMERA ACTIVE (LIVE PREVIEW)";
                    camBadge.className = "status-badge badge-open";
                } else {
                    alert("Camera Error: " + (result.error || "Could not start camera."));
                    startCamBtn.disabled = false;
                    startCamBtn.textContent = "Start Camera";
                }
            } catch (err) {
                alert("Failed to communicate with Python server.");
                startCamBtn.disabled = false;
                startCamBtn.textContent = "Start Camera";
            }
        });

        stopCamBtn.addEventListener("click", async () => {
            stopCamBtn.disabled = true;
            try {
                await fetch("/api/camera/stop", { method: "POST" });
                streamImg.src = "";
                streamImg.classList.add("hidden");
                placeholder.classList.remove("hidden");

                startCamBtn.disabled = false;
                captureBtn.disabled = true;

                camBadge.textContent = "CAMERA LOCKED (OFF)";
                camBadge.className = "status-badge badge-resolved";
            } catch (err) {
                alert("Error stopping camera.");
                stopCamBtn.disabled = false;
            }
        });
    }

    if (captureBtn) {
        captureBtn.addEventListener("click", async () => {
            const labelText = photoLabelInput ? photoLabelInput.value.trim() : "";
            captureBtn.disabled = true;
            captureBtn.textContent = "Capturing...";

            try {
                const response = await fetch("/api/camera/capture", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ label: labelText })
                });
                const result = await response.json();

                if (result.success) {
                    captureFeedback.classList.remove("hidden");
                    captureFeedback.innerHTML = `
                        <strong>[${result.photo_code}] Saved Locally!</strong><br>
                        File: <code>static/captures/${result.filename}</code><br>
                        Label: ${result.label} (${result.timestamp})
                    `;
                    if (photoLabelInput) photoLabelInput.value = "";

                    // Remove "No photos" message if present and prepend the new photo card
                    const noPhotosMsg = document.getElementById("no-photos-msg");
                    if (noPhotosMsg) noPhotosMsg.remove();

                    if (recentPhotosGrid) {
                        const card = document.createElement("div");
                        card.className = "photo-card";
                        card.innerHTML = `
                            <img src="${result.image_url}" alt="${result.label}">
                            <div class="photo-card-body">
                                <span class="code-cell">${result.photo_code}</span>
                                <p class="photo-label-text">${result.label}</p>
                                <small class="time-cell">${result.timestamp}</small>
                            </div>
                        `;
                        recentPhotosGrid.prepend(card);
                    }
                } else {
                    alert("Capture Failed: " + (result.error || "Unknown error"));
                }
            } catch (err) {
                alert("Error communicating with backend during photo capture.");
            } finally {
                captureBtn.disabled = false;
                captureBtn.textContent = "Capture Photo Now";
            }
        });
    }

    if (refreshBtn) {
        refreshBtn.addEventListener("click", fetchSystemStatus);
    }

    fetchSystemStatus();
});