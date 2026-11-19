"""Named values for the Palm University College attendance app."""

from datetime import time as clock_time


class FaceOutcome:
    MATCHED = "matched"
    UNKNOWN = "unknown"
    NO_FACE = "no_face"
    MANY_FACES = "many_faces"
    INCOMPLETE = "incomplete"


class LocationVerdict:
    ON_CAMPUS = "on_campus"
    OFF_CAMPUS = "off_campus"
    IMPRECISE = "imprecise"
    MISSING = "missing"


class AttendanceStatus:
    PRESENT = "present"
    LATE = "late"


class RecordState:
    RECORDED = "recorded"
    ALREADY = "already_recorded"
    BLOCKED = "blocked"
    INCOMPLETE = "incomplete"


class HttpStatus:
    OK = 200
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    NOT_FOUND = 404


CAMPUS_NAME = "Palm University College"
CAMPUS_ADDRESS = "1 University Avenue Annex, Manya Jorpanya, Shai Hills"
CAMPUS_LATITUDE = 5.920471
CAMPUS_LONGITUDE = 0.031308
CAMPUS_RADIUS_METRES = 500

EARTH_RADIUS_METRES = 6_371_000
METRES_PER_KILOMETRE = 1000
MAX_LATITUDE = 90
MAX_LONGITUDE = 180
MIN_ACCURACY_METRES = 0

LATE_THRESHOLD = clock_time(8, 15)
LATE_HOUR_LABEL = "08:15"
ACCRA_TIMEZONE = "Africa/Accra"

FACE_MATCH_TOLERANCE = 0.5
FACE_COUNT_NONE = 0
MAX_FACES_IN_PORTRAIT = 1
FACE_ENCODING_LENGTH = 128
MAX_FRAME_EDGE = 900
THUMB_EDGE = 160
PORTRAIT_EDGE = 480
JPEG_QUALITY = 82

EMPTY_TEXT = ""
EMPTY_REGISTRY = 0
YEAR_VALUES = ("1", "2", "3", "4")
MAX_NAME_LENGTH = 80
MAX_PROGRAM_LENGTH = 80
MAX_INDEX_LENGTH = 32
MAX_IMAGE_BYTES = 8 * 1024 * 1024
PHOTO_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

DATABASE_FILE = "attendance.db"
KNOWN_FACES_DIR = "known_faces"
STUDENT_PHOTOS_DIR = "student_photos"
SECRET_FILE = ".session_secret"

DEFAULT_APP_PORT = 4721
SESSION_KEY = "admin_signed_in"
SESSION_HOURS = 12
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD_ENV = "ADMIN_PASSWORD"
DEFAULT_ADMIN_PASSWORD = "palm2024"

ASSET_FILES = ("base.css", "scan.css", "scan.js", "admin.css", "admin.js")
HTML_TYPE = "text/html; charset=utf-8"
JPEG_DATA_PREFIX = "data:image/jpeg;base64,"
DATA_URL_SEPARATOR = ","
