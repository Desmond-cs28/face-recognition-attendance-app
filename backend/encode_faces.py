"""Enrol the photographs in known_faces. The admin page is the usual way."""

from database import connect, init_db
from seed import import_known_faces
from service import now_accra, reload_registry


def main():
    init_db()
    with connect() as connection:
        added = import_known_faces(connection, now_accra().isoformat(timespec="seconds"))
        reload_registry(connection)
    if added:
        print("Enrolled:", ", ".join(added))
        print("Open the registry page and add each index number, programme, and year.")
        return
    print("The registry already has students. Add photographs from the registry page.")


if __name__ == "__main__":
    main()
