from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import face_recognition
import mysql.connector
import numpy as np
import base64
import pickle
import os
from datetime import datetime, date, time as dtime
from io import BytesIO
from PIL import Image

app = Flask(__name__, static_folder=".")
CORS(app)


DB_CONFIG = {
    "host"    : "localhost",
    "user"    : "root",
    "password": "",          
    "database": "faceattend"
}

ENCODINGS_FILE  = "encodings.pkl"
LATE_THRESHOLD  = dtime(8, 15)   # scans after 08:15 AM marked late

# ── Load face encodings once at startup ──────────────────────
def load_encodings():
    if not os.path.exists(ENCODINGS_FILE):
        print("⚠️  encodings.pkl not found. Run encode_faces.py first.")
        return [], []
    with open(ENCODINGS_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"✅  Loaded {len(data['names'])} encoding(s): {data['names']}")
    return data["encodings"], data["names"]

known_encodings, known_names = load_encodings()

# ── DB helper ─────────────────────────────────────────────────
def get_db():
    return mysql.connector.connect(**DB_CONFIG)

# ── Serve frontend files ──────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(".", filename)

# ─────────────────────────────────────────────────────────────
#  MAIN RECOGNITION ENDPOINT
# ─────────────────────────────────────────────────────────────
@app.route("/recognize", methods=["POST"])
def recognize():
    payload = request.get_json()
    if not payload or "image" not in payload:
        return jsonify({"name": None, "face_box": None}), 400

    # Decode image
    try:
        img_bytes = base64.b64decode(payload["image"])
        pil_image = Image.open(BytesIO(img_bytes)).convert("RGB")
        frame     = np.array(pil_image)
    except Exception as e:
        return jsonify({"name": "Unknown", "error": str(e)}), 400

    # Detect faces
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    if not face_locations:
        return jsonify({"name": None, "face_box": None})

    # Use first detected face
    face_encoding = face_encodings[0]
    loc           = face_locations[0]   # top, right, bottom, left

    face_box = {
        "top"   : loc[0],
        "left"  : loc[3],
        "width" : loc[1] - loc[3],
        "height": loc[2] - loc[0]
    }

    # Match against known encodings
    matches   = face_recognition.compare_faces(
        known_encodings, face_encoding, tolerance=0.5
    )
    distances = face_recognition.face_distance(known_encodings, face_encoding)

    name         = "Unknown"
    student_info = {}

    if len(distances) > 0 and any(matches):
        best_idx = int(np.argmin(distances))
        if matches[best_idx]:
            name = known_names[best_idx]

            try:
                db     = get_db()
                cursor = db.cursor(dictionary=True)

                # Get student record
                cursor.execute(
                    "SELECT * FROM students WHERE name = %s", (name,)
                )
                student = cursor.fetchone()

                if student:
                    # Determine scan type (in or out)
                    scan_type = detect_scan_type(cursor, student["id"])

                    # Record the scan
                    record_scan(
                        cursor     = cursor,
                        db         = db,
                        student_id = student["id"],
                        scan_type  = scan_type,
                        latitude   = payload.get("latitude",  ""),
                        longitude  = payload.get("longitude", ""),
                        location   = payload.get("location",  "")
                    )

                    student_info = {
                        "name"        : student["name"],
                        "index_number": student["index_number"],
                        "program"     : student["program"],
                        "level"       : student["level"],
                        "scan_type"   : scan_type
                    }

                cursor.close()
                db.close()

            except Exception as e:
                print(f"[DB ERROR] {e}")
                student_info = {"name": name, "scan_type": "in"}

    return jsonify({
        "name"        : student_info.get("name",         name),
        "index_number": student_info.get("index_number", "—"),
        "program"     : student_info.get("program",      "—"),
        "level"       : student_info.get("level",        "—"),
        "scan_type"   : student_info.get("scan_type",    None),
        "face_box"    : face_box
    })


# ─────────────────────────────────────────────────────────────
#  SCAN TYPE AUTO-DETECT
#  Last scan today = "in"  → next = "out"
#  Last scan today = "out" → next = "in"
#  No scan today           → "in"
# ─────────────────────────────────────────────────────────────
def detect_scan_type(cursor, student_id):
    cursor.execute("""
        SELECT scan_type FROM scans
        WHERE student_id = %s
          AND DATE(scanned_at) = CURDATE()
        ORDER BY scanned_at DESC
        LIMIT 1
    """, (student_id,))
    row = cursor.fetchone()
    if not row:
        return "in"
    return "out" if row["scan_type"] == "in" else "in"


# ─────────────────────────────────────────────────────────────
#  RECORD EVERY SCAN AS A NEW ROW
#  Unlimited scan-ins and scan-outs per student per day
# ─────────────────────────────────────────────────────────────
def record_scan(cursor, db, student_id, scan_type,
                latitude, longitude, location):
    now    = datetime.now()
    status = (
        "late"
        if scan_type == "in" and now.time() > LATE_THRESHOLD
        else "present"
    )
    cursor.execute("""
        INSERT INTO scans
          (student_id, scan_type, status, scanned_at, latitude, longitude, location)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
    """, (
        student_id,
        scan_type,
        status,
        now,
        str(latitude),
        str(longitude),
        location
    ))
    db.commit()


# ─────────────────────────────────────────────────────────────
#  API — GET TODAY'S SCANS (admin dashboard)
# ─────────────────────────────────────────────────────────────
@app.route("/api/scans/today", methods=["GET"])
def get_today_scans():
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                s.name, s.index_number, s.program, s.level,
                sc.scan_type, sc.status, sc.location,
                TIME_FORMAT(sc.scanned_at, '%H:%i:%s') AS scan_time,
                DATE(sc.scanned_at) AS scan_date,
                sc.id AS scan_id
            FROM scans sc
            JOIN students s ON sc.student_id = s.id
            WHERE DATE(sc.scanned_at) = CURDATE()
            ORDER BY sc.scanned_at ASC
        """)
        rows = cursor.fetchall()
        for row in rows:
            if row.get("scan_date"):
                row["scan_date"] = str(row["scan_date"])
        cursor.close()
        db.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
#  API — GET SCANS BY DATE
# ─────────────────────────────────────────────────────────────
@app.route("/api/scans/<date_str>", methods=["GET"])
def get_scans_by_date(date_str):
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                s.name, s.index_number, s.program, s.level,
                sc.scan_type, sc.status, sc.location,
                TIME_FORMAT(sc.scanned_at, '%H:%i:%s') AS scan_time,
                DATE(sc.scanned_at) AS scan_date,
                sc.id AS scan_id
            FROM scans sc
            JOIN students s ON sc.student_id = s.id
            WHERE DATE(sc.scanned_at) = %s
            ORDER BY sc.scanned_at ASC
        """, (date_str,))
        rows = cursor.fetchall()
        for row in rows:
            if row.get("scan_date"):
                row["scan_date"] = str(row["scan_date"])
        cursor.close()
        db.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
#  API — GET A STUDENT'S FULL SCAN HISTORY
# ─────────────────────────────────────────────────────────────
@app.route("/api/student/<index_number>/scans", methods=["GET"])
def get_student_scans(index_number):
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("""
            SELECT
                s.name, s.index_number, s.program, s.level,
                sc.scan_type, sc.status, sc.location,
                TIME_FORMAT(sc.scanned_at, '%H:%i:%s') AS scan_time,
                DATE(sc.scanned_at) AS scan_date
            FROM scans sc
            JOIN students s ON sc.student_id = s.id
            WHERE s.index_number = %s
            ORDER BY sc.scanned_at DESC
        """, (index_number,))
        rows = cursor.fetchall()
        for row in rows:
            if row.get("scan_date"):
                row["scan_date"] = str(row["scan_date"])
        cursor.close()
        db.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
#  API — GET ALL STUDENTS
# ─────────────────────────────────────────────────────────────
@app.route("/api/students", methods=["GET"])
def get_students():
    try:
        db     = get_db()
        cursor = db.cursor(dictionary=True)
        cursor.execute("SELECT * FROM students ORDER BY name")
        rows = cursor.fetchall()
        for row in rows:
            if row.get("created_at"):
                row["created_at"] = str(row["created_at"])
        cursor.close()
        db.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
#  API — ADD STUDENT
# ─────────────────────────────────────────────────────────────
@app.route("/api/students", methods=["POST"])
def add_student():
    d = request.get_json()
    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute("""
            INSERT INTO students (name, index_number, program, level)
            VALUES (%s, %s, %s, %s)
        """, (d["name"], d["index_number"], d["program"], d["level"]))
        db.commit()
        new_id = cursor.lastrowid
        cursor.close()
        db.close()
        return jsonify({"success": True, "id": new_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
#  API — DELETE STUDENT
# ─────────────────────────────────────────────────────────────
@app.route("/api/students/<int:student_id>", methods=["DELETE"])
def delete_student(student_id):
    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute(
            "DELETE FROM scans WHERE student_id = %s", (student_id,)
        )
        cursor.execute(
            "DELETE FROM students WHERE id = %s", (student_id,)
        )
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────
#  API — DELETE A SINGLE SCAN ENTRY
# ─────────────────────────────────────────────────────────────
@app.route("/api/scans/<int:scan_id>", methods=["DELETE"])
def delete_scan(scan_id):
    try:
        db     = get_db()
        cursor = db.cursor()
        cursor.execute("DELETE FROM scans WHERE id = %s", (scan_id,))
        db.commit()
        cursor.close()
        db.close()
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚀  FaceAttend backend running")
    print("  📡  http://localhost:5000")
    print("="*55 + "\n")
    app.run(debug=True, host="0.0.0.0", port=5000)