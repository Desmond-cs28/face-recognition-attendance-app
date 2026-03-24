import mysql.connector
from datetime import datetime


def connect_db():

    return mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="face_attendance_db"
    )


def save_attendance(name, latitude, longitude, ip):

    db = connect_db()

    cursor = db.cursor()

    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    query = """
    INSERT INTO attendance (name, date, time, latitude, longitude, ip_address)
    VALUES (%s,%s,%s,%s,%s,%s)
    """

    values = (name, date, time, latitude, longitude, ip)

    cursor.execute(query, values)

    db.commit()

    cursor.close()
    db.close()