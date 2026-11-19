"""SQLite registry of students and the attendance they record on campus."""

import os
import sqlite3
from contextlib import contextmanager

from constants import (
    DATABASE_FILE,
    EMPTY_REGISTRY,
    EMPTY_TEXT,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(BASE_DIR, DATABASE_FILE)

SCHEMA = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    index_number TEXT NOT NULL DEFAULT '',
    program TEXT NOT NULL DEFAULT '',
    year_of_study TEXT NOT NULL DEFAULT '',
    photo_path TEXT NOT NULL,
    source_filename TEXT UNIQUE,
    encoding BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY,
    student_id INTEGER NOT NULL,
    status TEXT NOT NULL,
    scanned_at TEXT NOT NULL,
    attendance_date TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    distance_metres REAL NOT NULL,
    UNIQUE (student_id, attendance_date),
    FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
);
"""


@contextmanager
def connect():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db():
    with connect() as connection:
        connection.executescript(SCHEMA)


def count_students(connection):
    row = connection.execute("SELECT COUNT(*) AS total FROM students").fetchone()
    return row["total"]


def registry_is_empty(connection):
    return count_students(connection) == EMPTY_REGISTRY


def insert_student(connection, fields):
    cursor = connection.execute(
        """
        INSERT INTO students (
            name, index_number, program, year_of_study,
            photo_path, source_filename, encoding, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fields["name"],
            fields["index_number"],
            fields["program"],
            fields["year_of_study"],
            fields["photo_path"],
            fields["source_filename"],
            fields["encoding"],
            fields["created_at"],
        ),
    )
    return cursor.lastrowid


def update_student_details(connection, student_id, fields):
    connection.execute(
        """
        UPDATE students
        SET name = ?, index_number = ?, program = ?, year_of_study = ?
        WHERE id = ?
        """,
        (
            fields["name"],
            fields["index_number"],
            fields["program"],
            fields["year_of_study"],
            student_id,
        ),
    )


def update_student_photo(connection, student_id, photo_path, encoding):
    connection.execute(
        "UPDATE students SET photo_path = ?, encoding = ? WHERE id = ?",
        (photo_path, encoding, student_id),
    )


def delete_student(connection, student_id):
    connection.execute("DELETE FROM students WHERE id = ?", (student_id,))


def get_student(connection, student_id):
    return connection.execute(
        "SELECT * FROM students WHERE id = ?",
        (student_id,),
    ).fetchone()


def list_students(connection):
    return connection.execute(
        "SELECT * FROM students ORDER BY name COLLATE NOCASE"
    ).fetchall()


def find_by_source(connection, source_filename):
    return connection.execute(
        "SELECT id FROM students WHERE source_filename = ?",
        (source_filename,),
    ).fetchone()


def index_owner(connection, index_number, ignore_id):
    if index_number == EMPTY_TEXT:
        return None
    row = connection.execute(
        "SELECT id FROM students WHERE index_number = ?",
        (index_number,),
    ).fetchone()
    if row is None:
        return None
    if ignore_id is not None and row["id"] == ignore_id:
        return None
    return row


def find_attendance(connection, student_id, attendance_date):
    return connection.execute(
        """
        SELECT * FROM attendance
        WHERE student_id = ? AND attendance_date = ?
        """,
        (student_id, attendance_date),
    ).fetchone()


def insert_attendance(connection, fields):
    cursor = connection.execute(
        """
        INSERT INTO attendance (
            student_id, status, scanned_at, attendance_date,
            latitude, longitude, distance_metres
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            fields["student_id"],
            fields["status"],
            fields["scanned_at"],
            fields["attendance_date"],
            fields["latitude"],
            fields["longitude"],
            fields["distance_metres"],
        ),
    )
    return cursor.lastrowid


def list_attendance(connection, attendance_date):
    return connection.execute(
        """
        SELECT
            a.scanned_at, a.status, a.distance_metres,
            s.name, s.index_number, s.program, s.year_of_study
        FROM attendance a
        JOIN students s ON s.id = a.student_id
        WHERE a.attendance_date = ?
        ORDER BY a.scanned_at
        """,
        (attendance_date,),
    ).fetchall()
