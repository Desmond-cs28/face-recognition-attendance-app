"""Detect a face and compare it with the registry encodings."""

from io import BytesIO

import face_recognition
import numpy as np
from PIL import Image

from constants import (
    FACE_COUNT_NONE,
    FACE_ENCODING_LENGTH,
    FACE_MATCH_TOLERANCE,
    FaceOutcome,
    JPEG_QUALITY,
    MAX_FACES_IN_PORTRAIT,
    MAX_FRAME_EDGE,
    PORTRAIT_EDGE,
    THUMB_EDGE,
)


def _open_image(image_bytes):
    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    image.thumbnail((MAX_FRAME_EDGE, MAX_FRAME_EDGE))
    return image


def _largest(locations):
    def area(location):
        top, right, bottom, left = location
        return (bottom - top) * (right - left)

    return max(locations, key=area)


def _encoding_from_image(image, require_single):
    frame = np.array(image)
    locations = face_recognition.face_locations(frame)
    count = len(locations)
    if count == FACE_COUNT_NONE:
        return None, FaceOutcome.NO_FACE
    if require_single and count > MAX_FACES_IN_PORTRAIT:
        return None, FaceOutcome.MANY_FACES
    chosen = _largest(locations)
    encodings = face_recognition.face_encodings(frame, [chosen])
    if not encodings:
        return None, FaceOutcome.NO_FACE
    encoding = np.asarray(encodings[0], dtype=np.float64)
    if encoding.shape[0] != FACE_ENCODING_LENGTH:
        return None, FaceOutcome.NO_FACE
    return encoding, None


def encode_scan(image_bytes):
    return _encoding_from_image(_open_image(image_bytes), require_single=False)


def encode_portrait(image_bytes):
    return _encoding_from_image(_open_image(image_bytes), require_single=True)


def encoding_bytes(encoding):
    return np.asarray(encoding, dtype=np.float64).tobytes()


def encoding_from_blob(blob):
    values = np.frombuffer(blob, dtype=np.float64)
    return np.array(values, copy=True)


def match_student(unknown, rows, encodings):
    if not encodings:
        return None
    matches = face_recognition.compare_faces(
        encodings,
        unknown,
        tolerance=FACE_MATCH_TOLERANCE,
    )
    distances = face_recognition.face_distance(encodings, unknown)
    best_index = int(np.argmin(distances))
    if not matches[best_index]:
        return None
    return rows[best_index]


def jpeg_data_url(path, edge, prefix):
    image = Image.open(path).convert("RGB")
    image.thumbnail((edge, edge))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY)
    import base64

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return prefix + encoded


def save_jpeg(image_bytes, path):
    image = _open_image(image_bytes)
    image.thumbnail((PORTRAIT_EDGE * 2, PORTRAIT_EDGE * 2))
    image.save(path, format="JPEG", quality=JPEG_QUALITY)


def thumb_edge():
    return THUMB_EDGE


def portrait_edge():
    return PORTRAIT_EDGE
