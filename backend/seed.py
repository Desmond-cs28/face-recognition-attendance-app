"""Load the photographs already stored in known_faces on a first run."""

import os

from constants import KNOWN_FACES_DIR, PHOTO_EXTENSIONS
from database import find_by_source, insert_student, registry_is_empty
from faces import encode_portrait, encoding_bytes
from names import display_name


def _photo_files(folder):
    names = []
    for filename in sorted(os.listdir(folder)):
        lower = filename.lower()
        if lower.endswith(PHOTO_EXTENSIONS):
            names.append(filename)
    return names


def import_known_faces(connection, now_iso):
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), KNOWN_FACES_DIR)
    if not os.path.isdir(folder):
        return []
    if not registry_is_empty(connection):
        return []
    added = []
    for filename in _photo_files(folder):
        if find_by_source(connection, filename) is not None:
            continue
        path = os.path.join(folder, filename)
        with open(path, "rb") as handle:
            encoding, problem = encode_portrait(handle.read())
        if problem is not None or encoding is None:
            continue
        insert_student(
            connection,
            {
                "name": display_name(filename),
                "index_number": "",
                "program": "",
                "year_of_study": "",
                "photo_path": path,
                "source_filename": filename,
                "encoding": encoding_bytes(encoding),
                "created_at": now_iso,
            },
        )
        added.append(display_name(filename))
    return added
