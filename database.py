import sqlite3
from datetime import datetime

DB_NAME = "safety_records.db"


def get_db_connection():
    """Creates a connection to the local SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates the required tables if they do not exist yet."""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Table 1: Simulated Incidents (Milestone 2)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            incident_code TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            incident_type TEXT NOT NULL,
            reason TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN'
        )
    """)

    # Table 2: Captured Photo Records (Milestone 3 & 4)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS photo_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            photo_code TEXT NOT NULL,
            filename TEXT NOT NULL,
            label TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ==========================================
# INCIDENT FUNCTIONS (Milestone 2)
# ==========================================

def create_incident(incident_type: str, reason: str) -> dict:
    """Inserts a new simulated safety incident and generates a readable code."""
    conn = get_db_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO incidents (incident_code, timestamp, incident_type, reason, status) VALUES (?, ?, ?, ?, ?)",
        ("PENDING", timestamp, incident_type, reason, "OPEN")
    )
    new_id = cursor.lastrowid
    incident_code = f"INC-{new_id:04d}"

    cursor.execute(
        "UPDATE incidents SET incident_code = ? WHERE id = ?",
        (incident_code, new_id)
    )
    conn.commit()
    conn.close()

    return {
        "id": new_id,
        "incident_code": incident_code,
        "timestamp": timestamp,
        "incident_type": incident_type,
        "reason": reason,
        "status": "OPEN"
    }


def get_all_incidents(search_query: str = "", status_filter: str = "ALL") -> list:
    """Fetches incidents from SQLite with optional keyword search and status filter."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM incidents WHERE 1=1"
    params = []

    if search_query:
        query += " AND (reason LIKE ? OR incident_code LIKE ? OR incident_type LIKE ?)"
        wildcard = f"%{search_query}%"
        params.extend([wildcard, wildcard, wildcard])

    if status_filter in ("OPEN", "RESOLVED"):
        query += " AND status = ?"
        params.append(status_filter)

    query += " ORDER BY id DESC"

    rows = cursor.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_incident_status(incident_id: int, new_status: str) -> bool:
    """Updates an incident's status between OPEN and RESOLVED."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE incidents SET status = ? WHERE id = ?",
        (new_status, incident_id)
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def delete_incident(incident_id: int) -> bool:
    """Deletes a specific incident record by its ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM incidents WHERE id = ?", (incident_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def get_incident_counts() -> dict:
    """Returns total, open incident, and photo counts for dashboard telemetry."""
    conn = get_db_connection()
    cursor = conn.cursor()
    total = cursor.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
    open_count = cursor.execute("SELECT COUNT(*) FROM incidents WHERE status = 'OPEN'").fetchone()[0]
    photo_count = cursor.execute("SELECT COUNT(*) FROM photo_records").fetchone()[0]
    conn.close()
    return {"total": total, "open": open_count, "photos": photo_count}


# ==========================================
# PHOTO RECORD FUNCTIONS (Milestone 3 & 4)
# ==========================================

def create_photo_record(filename: str, label: str) -> dict:
    """Saves metadata for a user-captured photo into SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute(
        "INSERT INTO photo_records (photo_code, filename, label, timestamp) VALUES (?, ?, ?, ?)",
        ("PENDING", filename, label, timestamp)
    )
    new_id = cursor.lastrowid
    photo_code = f"CAP-{new_id:04d}"

    cursor.execute(
        "UPDATE photo_records SET photo_code = ? WHERE id = ?",
        (photo_code, new_id)
    )
    conn.commit()
    conn.close()

    return {
        "id": new_id,
        "photo_code": photo_code,
        "filename": filename,
        "label": label,
        "timestamp": timestamp
    }


def get_recent_photos(limit: int = 4) -> list:
    """Returns the most recently captured photo records for the camera page."""
    conn = get_db_connection()
    cursor = conn.cursor()
    rows = cursor.execute(
        "SELECT * FROM photo_records ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_all_photos(search_query: str = "") -> list:
    """Fetches all photo records with optional keyword search by ID, label, or timestamp."""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT * FROM photo_records WHERE 1=1"
    params = []

    if search_query:
        query += " AND (photo_code LIKE ? OR label LIKE ? OR timestamp LIKE ?)"
        wildcard = f"%{search_query}%"
        params.extend([wildcard, wildcard, wildcard])

    query += " ORDER BY id DESC"
    rows = cursor.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_photo_label(photo_id: int, new_label: str) -> bool:
    """Updates the user-entered label for a specific photo record."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE photo_records SET label = ? WHERE id = ?",
        (new_label, photo_id)
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def delete_photo_record(photo_id: int) -> str | None:
    """Deletes a photo record from SQLite and returns its filename so the file can be removed."""
    conn = get_db_connection()
    cursor = conn.cursor()

    row = cursor.execute(
        "SELECT filename FROM photo_records WHERE id = ?",
        (photo_id,)
    ).fetchone()

    if not row:
        conn.close()
        return None

    filename = row["filename"]
    cursor.execute("DELETE FROM photo_records WHERE id = ?", (photo_id,))
    conn.commit()
    conn.close()
    return filename