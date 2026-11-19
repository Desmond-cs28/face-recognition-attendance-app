"""Palm University College face-attendance server."""

import base64
import os
from datetime import timedelta

from flask import Flask, abort, jsonify, request, send_from_directory, session

from constants import (
    ADMIN_PASSWORD_ENV,
    ADMIN_USERNAME,
    ASSET_FILES,
    DATA_URL_SEPARATOR,
    DEFAULT_ADMIN_PASSWORD,
    DEFAULT_APP_PORT,
    EMPTY_TEXT,
    HTML_TYPE,
    HttpStatus,
    MAX_IMAGE_BYTES,
    SECRET_FILE,
    SESSION_HOURS,
    SESSION_KEY,
)
from database import connect, init_db
from seed import import_known_faces
from service import (
    RegistryError,
    add_student,
    attendance_today,
    check_location,
    edit_student,
    now_accra,
    recognize,
    reload_registry,
    remove_student,
    student_cards,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "frontend"))

app = Flask(__name__)
app.permanent_session_lifetime = timedelta(hours=SESSION_HOURS)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
_BOOTED = False


def _secret():
    path = os.path.join(BASE_DIR, SECRET_FILE)
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    token = os.urandom(32).hex()
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(token)
    return token


app.secret_key = _secret()


def boot_once():
    global _BOOTED
    if _BOOTED:
        return
    init_db()
    with connect() as connection:
        import_known_faces(connection, now_accra().isoformat(timespec="seconds"))
        reload_registry(connection)
    _BOOTED = True


def admin_password():
    return os.environ.get(ADMIN_PASSWORD_ENV, DEFAULT_ADMIN_PASSWORD)


def signed_in():
    return session.get(SESSION_KEY) == ADMIN_USERNAME


def fail(message, status=HttpStatus.BAD_REQUEST):
    return jsonify({"message": message}), status


def require_admin():
    if signed_in():
        return None
    return fail("Sign in to the registry office first.", HttpStatus.UNAUTHORIZED)


def _page(filename):
    response = send_from_directory(FRONTEND_DIR, filename)
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Type"] = HTML_TYPE
    return response


def _upload_bytes(required):
    upload = request.files.get("photo")
    if upload is None or upload.filename == EMPTY_TEXT:
        if required:
            raise RegistryError("Choose a photograph of the student.")
        return None
    data = upload.read()
    if len(data) > MAX_IMAGE_BYTES:
        raise RegistryError("That photograph is too large.")
    if not data:
        raise RegistryError("That photograph is empty.")
    return data


def _scan_bytes(payload):
    image = payload.get("image") or EMPTY_TEXT
    if DATA_URL_SEPARATOR in image:
        image = image.split(DATA_URL_SEPARATOR, 1)[1]
    try:
        raw = base64.b64decode(image, validate=False)
    except Exception as error:
        raise RegistryError("The camera frame could not be read.") from error
    if raw == b"" or len(raw) > MAX_IMAGE_BYTES:
        raise RegistryError("The camera frame could not be read.")
    return raw


@app.before_request
def _ensure_registry():
    boot_once()


@app.get("/")
def student_page():
    return _page("index.html")


@app.get("/admin")
def admin_page():
    return _page("admin.html")


@app.get("/assets/<path:filename>")
def assets(filename):
    if filename not in ASSET_FILES:
        abort(HttpStatus.NOT_FOUND)
    return send_from_directory(FRONTEND_DIR, filename)


@app.get("/api/campus")
def campus():
    return jsonify(check_location(None, None, None))


@app.post("/api/location/check")
def location_check():
    payload = request.get_json(silent=True) or {}
    return jsonify(
        check_location(
            payload.get("latitude"),
            payload.get("longitude"),
            payload.get("accuracy_metres"),
        )
    )


@app.post("/api/recognize")
def recognize_face():
    payload = request.get_json(silent=True) or {}
    try:
        image = _scan_bytes(payload)
        with connect() as connection:
            result = recognize(
                connection,
                image,
                payload.get("latitude"),
                payload.get("longitude"),
                payload.get("accuracy_metres"),
            )
    except RegistryError as error:
        return fail(str(error))
    return jsonify(result)


@app.post("/api/admin/login")
def login():
    payload = request.get_json(silent=True) or {}
    username = (payload.get("username") or EMPTY_TEXT).strip()
    password = payload.get("password") or EMPTY_TEXT
    if username == ADMIN_USERNAME and password == admin_password():
        session[SESSION_KEY] = ADMIN_USERNAME
        session.permanent = True
        return jsonify({"signed_in": True})
    return fail(
        "Those details do not match the registry office login.",
        HttpStatus.UNAUTHORIZED,
    )


@app.post("/api/admin/logout")
def logout():
    session.pop(SESSION_KEY, None)
    return jsonify({"signed_in": False})


@app.get("/api/admin/session")
def admin_session():
    return jsonify({"signed_in": signed_in()})


@app.get("/api/students")
def students():
    denied = require_admin()
    if denied is not None:
        return denied
    with connect() as connection:
        return jsonify(student_cards(connection))


@app.post("/api/students")
def create_student():
    denied = require_admin()
    if denied is not None:
        return denied
    try:
        image = _upload_bytes(required=True)
        with connect() as connection:
            add_student(connection, request.form, image)
    except RegistryError as error:
        return fail(str(error))
    return jsonify({"saved": True})


@app.post("/api/students/<int:student_id>")
def update_student(student_id):
    denied = require_admin()
    if denied is not None:
        return denied
    try:
        image = _upload_bytes(required=False)
        with connect() as connection:
            edit_student(connection, student_id, request.form, image)
    except RegistryError as error:
        return fail(str(error))
    return jsonify({"saved": True})


@app.delete("/api/students/<int:student_id>")
def destroy_student(student_id):
    denied = require_admin()
    if denied is not None:
        return denied
    try:
        with connect() as connection:
            remove_student(connection, student_id)
    except RegistryError as error:
        return fail(str(error))
    return jsonify({"saved": True})


@app.get("/api/attendance/today")
def today():
    denied = require_admin()
    if denied is not None:
        return denied
    with connect() as connection:
        return jsonify(attendance_today(connection))


if __name__ == "__main__":
    boot_once()
    port = int(os.environ.get("PORT", str(DEFAULT_APP_PORT)))
    print(f"Palm attendance ready on port {port}")
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
