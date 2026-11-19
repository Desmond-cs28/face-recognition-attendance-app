"""Registry, campus check, and attendance recording."""

import os
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from constants import (
    ACCRA_TIMEZONE,
    AttendanceStatus,
    EMPTY_TEXT,
    FaceOutcome,
    JPEG_DATA_PREFIX,
    LATE_HOUR_LABEL,
    LATE_THRESHOLD,
    LocationVerdict,
    MAX_INDEX_LENGTH,
    MAX_NAME_LENGTH,
    MAX_PROGRAM_LENGTH,
    PORTRAIT_EDGE,
    RecordState,
    STUDENT_PHOTOS_DIR,
    THUMB_EDGE,
    YEAR_VALUES,
)
from database import (
    delete_student,
    find_attendance,
    get_student,
    index_owner,
    insert_attendance,
    insert_student,
    list_attendance,
    list_students,
    update_student_details,
    update_student_photo,
)
from faces import (
    encode_portrait,
    encode_scan,
    encoding_bytes,
    encoding_from_blob,
    jpeg_data_url,
    match_student,
    save_jpeg,
)
from geo import location_payload, campus_verdict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PHOTO_DIR = os.path.join(BASE_DIR, STUDENT_PHOTOS_DIR)
ACCRA = ZoneInfo(ACCRA_TIMEZONE)

REGISTRY = {"rows": [], "encodings": []}


class RegistryError(Exception):
    pass


def now_accra():
    return datetime.now(ACCRA)


def details_complete(row):
    return bool(
        row["name"]
        and row["index_number"]
        and row["program"]
        and row["year_of_study"]
    )


def reload_registry(connection):
    rows = []
    encodings = []
    for row in list_students(connection):
        item = _student_dict(row)
        rows.append(item)
        encodings.append(encoding_from_blob(row["encoding"]))
    REGISTRY["rows"] = rows
    REGISTRY["encodings"] = encodings


def _student_dict(row):
    return {
        "id": row["id"],
        "name": row["name"],
        "index_number": row["index_number"],
        "program": row["program"],
        "year_of_study": row["year_of_study"],
        "photo_path": row["photo_path"],
        "source_filename": row["source_filename"],
        "created_at": row["created_at"],
        "details_complete": details_complete(row),
    }


def _year_label(year_of_study):
    if year_of_study == EMPTY_TEXT:
        return EMPTY_TEXT
    return f"Year {year_of_study}"


def public_student(row, edge):
    photo = None
    path = row["photo_path"]
    if path and os.path.isfile(path):
        photo = jpeg_data_url(path, edge, JPEG_DATA_PREFIX)
    return {
        "id": row["id"],
        "name": row["name"],
        "index_number": row["index_number"],
        "program": row["program"],
        "year_of_study": row["year_of_study"],
        "year_label": _year_label(row["year_of_study"]),
        "details_complete": row["details_complete"],
        "photo": photo,
    }


def student_cards(connection):
    reload_registry(connection)
    return [public_student(row, THUMB_EDGE) for row in REGISTRY["rows"]]


def check_location(latitude, longitude, accuracy_metres):
    verdict, distance = campus_verdict(latitude, longitude, accuracy_metres)
    return location_payload(verdict, distance)


def _clean_text(value, limit, label):
    text = (value or EMPTY_TEXT).strip()
    if text == EMPTY_TEXT:
        raise RegistryError(f"Enter the {label}.")
    if len(text) > limit:
        raise RegistryError(f"The {label} is too long.")
    return text


def _clean_year(value):
    year = (value or EMPTY_TEXT).strip()
    if year not in YEAR_VALUES:
        raise RegistryError("Choose a year of study from 1 to 4.")
    return year


def _require_fields(payload):
    name = _clean_text(payload.get("name"), MAX_NAME_LENGTH, "student name")
    index_number = _clean_text(
        payload.get("index_number"),
        MAX_INDEX_LENGTH,
        "index number",
    )
    program = _clean_text(payload.get("program"), MAX_PROGRAM_LENGTH, "programme")
    year_of_study = _clean_year(payload.get("year_of_study"))
    return {
        "name": name,
        "index_number": index_number,
        "program": program,
        "year_of_study": year_of_study,
    }


def _store_photo(image_bytes):
    os.makedirs(PHOTO_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(PHOTO_DIR, filename)
    save_jpeg(image_bytes, path)
    return path


def _portrait_encoding(image_bytes):
    encoding, problem = encode_portrait(image_bytes)
    if problem == FaceOutcome.NO_FACE:
        raise RegistryError(
            "No face was found in that photograph. Use a clear, front-facing photo of one student."
        )
    if problem == FaceOutcome.MANY_FACES:
        raise RegistryError(
            "That photograph has more than one face. Use a photo of one student."
        )
    if encoding is None:
        raise RegistryError("That photograph could not be read.")
    return encoding


def add_student(connection, payload, image_bytes):
    fields = _require_fields(payload)
    if index_owner(connection, fields["index_number"], None) is not None:
        raise RegistryError("That index number is already on the registry.")
    encoding = _portrait_encoding(image_bytes)
    path = _store_photo(image_bytes)
    student_id = insert_student(
        connection,
        {
            **fields,
            "photo_path": path,
            "source_filename": os.path.basename(path),
            "encoding": encoding_bytes(encoding),
            "created_at": now_accra().isoformat(timespec="seconds"),
        },
    )
    reload_registry(connection)
    return student_id


def edit_student(connection, student_id, payload, image_bytes):
    current = get_student(connection, student_id)
    if current is None:
        raise RegistryError("That student is no longer on the registry.")
    fields = _require_fields(payload)
    if index_owner(connection, fields["index_number"], student_id) is not None:
        raise RegistryError("That index number is already on the registry.")
    update_student_details(connection, student_id, fields)
    if image_bytes:
        encoding = _portrait_encoding(image_bytes)
        path = _store_photo(image_bytes)
        update_student_photo(connection, student_id, path, encoding_bytes(encoding))
        _remove_upload(current["photo_path"])
    reload_registry(connection)


def remove_student(connection, student_id):
    current = get_student(connection, student_id)
    if current is None:
        raise RegistryError("That student is no longer on the registry.")
    delete_student(connection, student_id)
    _remove_upload(current["photo_path"])
    reload_registry(connection)


def _remove_upload(path):
    if not path:
        return
    folder = os.path.abspath(PHOTO_DIR)
    target = os.path.abspath(path)
    if target.startswith(folder) and os.path.isfile(target):
        os.remove(target)


def _status_label(status):
    if status == AttendanceStatus.LATE:
        return "Late"
    return "Present"


def _status_for(moment):
    if moment.time() > LATE_THRESHOLD:
        return AttendanceStatus.LATE
    return AttendanceStatus.PRESENT


def _clock(moment):
    return moment.strftime("%H:%M")


def attendance_today(connection):
    moment = now_accra()
    rows = list_attendance(connection, moment.date().isoformat())
    items = []
    for row in rows:
        scanned = datetime.fromisoformat(row["scanned_at"])
        items.append(
            {
                "name": row["name"],
                "index_number": row["index_number"],
                "program": row["program"],
                "year_label": _year_label(row["year_of_study"]),
                "status": row["status"],
                "mark": _status_label(row["status"]),
                "time": _clock(scanned),
                "distance_label": f"{int(round(row['distance_metres']))} m",
            }
        )
    return {"date": moment.date().isoformat(), "items": items}


def recognize(connection, image_bytes, latitude, longitude, accuracy_metres):
    location = check_location(latitude, longitude, accuracy_metres)
    encoding, problem = encode_scan(image_bytes)
    if problem == FaceOutcome.NO_FACE or encoding is None:
        return {
            "outcome": FaceOutcome.NO_FACE,
            "message": "No face was found. Look at the camera and try again.",
            "location": location,
            "student": None,
            "attendance": None,
        }
    if not REGISTRY["rows"]:
        reload_registry(connection)
    row = match_student(encoding, REGISTRY["rows"], REGISTRY["encodings"])
    if row is None:
        return {
            "outcome": FaceOutcome.UNKNOWN,
            "message": "This face is not in the registry yet.",
            "location": location,
            "student": None,
            "attendance": None,
        }
    student = public_student(row, PORTRAIT_EDGE)
    if not row["details_complete"]:
        return {
            "outcome": FaceOutcome.INCOMPLETE,
            "message": (
                "This face is on file. Add the index number, programme, and "
                "year in the registry before attendance can be recorded."
            ),
            "location": location,
            "student": student,
            "attendance": {"recorded": False, "record_state": RecordState.INCOMPLETE},
        }
    if location["verdict"] != LocationVerdict.ON_CAMPUS:
        return {
            "outcome": FaceOutcome.MATCHED,
            "message": location["message"],
            "location": location,
            "student": student,
            "attendance": {"recorded": False, "record_state": RecordState.BLOCKED},
        }
    return _record(connection, row, student, location, latitude, longitude)


def _record(connection, row, student, location, latitude, longitude):
    moment = now_accra()
    day = moment.date().isoformat()
    existing = find_attendance(connection, row["id"], day)
    if existing is not None:
        scanned = datetime.fromisoformat(existing["scanned_at"])
        return {
            "outcome": FaceOutcome.MATCHED,
            "message": f"Attendance for today is already on the record at {_clock(scanned)}.",
            "location": location,
            "student": student,
            "attendance": {
                "recorded": True,
                "record_state": RecordState.ALREADY,
                "status": existing["status"],
                "time": _clock(scanned),
            },
        }
    status = _status_for(moment)
    insert_attendance(
        connection,
        {
            "student_id": row["id"],
            "status": status,
            "scanned_at": moment.isoformat(timespec="seconds"),
            "attendance_date": day,
            "latitude": float(latitude),
            "longitude": float(longitude),
            "distance_metres": location["distance_metres"],
        },
    )
    if status == AttendanceStatus.LATE:
        message = f"Attendance recorded. Arrival is after {LATE_HOUR_LABEL}, so this mark is late."
    else:
        message = "Attendance recorded."
    return {
        "outcome": FaceOutcome.MATCHED,
        "message": message,
        "location": location,
        "student": student,
        "attendance": {
            "recorded": True,
            "record_state": RecordState.RECORDED,
            "status": status,
            "time": _clock(moment),
        },
    }
