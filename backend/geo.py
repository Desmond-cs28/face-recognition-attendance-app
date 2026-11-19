"""Campus check for Palm University College, Manya Jorpanya."""

import math

from constants import (
    CAMPUS_ADDRESS,
    CAMPUS_LATITUDE,
    CAMPUS_LONGITUDE,
    CAMPUS_NAME,
    CAMPUS_RADIUS_METRES,
    EARTH_RADIUS_METRES,
    LocationVerdict,
    MAX_LATITUDE,
    MAX_LONGITUDE,
    METRES_PER_KILOMETRE,
    MIN_ACCURACY_METRES,
)


def haversine_metres(latitude, longitude, other_latitude, other_longitude):
    lat_one = math.radians(latitude)
    lat_two = math.radians(other_latitude)
    lat_delta = math.radians(other_latitude - latitude)
    lon_delta = math.radians(other_longitude - longitude)
    chord = (
        math.sin(lat_delta / 2) ** 2
        + math.cos(lat_one) * math.cos(lat_two) * math.sin(lon_delta / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METRES * math.asin(math.sqrt(chord))


def distance_label(metres):
    if metres < METRES_PER_KILOMETRE:
        return f"{int(round(metres))} m"
    kilometres = metres / METRES_PER_KILOMETRE
    return f"{kilometres:.1f} km"


def _coordinates(latitude, longitude, accuracy_metres):
    try:
        lat = float(latitude)
        lon = float(longitude)
        accuracy = float(accuracy_metres)
    except (TypeError, ValueError):
        return None
    if abs(lat) > MAX_LATITUDE or abs(lon) > MAX_LONGITUDE:
        return None
    if accuracy < MIN_ACCURACY_METRES:
        return None
    return lat, lon, accuracy


def campus_verdict(latitude, longitude, accuracy_metres):
    parsed = _coordinates(latitude, longitude, accuracy_metres)
    if parsed is None:
        return LocationVerdict.MISSING, None
    lat, lon, accuracy = parsed
    distance = haversine_metres(lat, lon, CAMPUS_LATITUDE, CAMPUS_LONGITUDE)
    if accuracy > CAMPUS_RADIUS_METRES:
        return LocationVerdict.IMPRECISE, distance
    outer_edge = distance + accuracy
    if outer_edge <= CAMPUS_RADIUS_METRES:
        return LocationVerdict.ON_CAMPUS, distance
    return LocationVerdict.OFF_CAMPUS, distance


def location_payload(verdict, distance):
    label = None if distance is None else distance_label(distance)
    message = _message(verdict, label)
    return {
        "verdict": verdict,
        "distance_metres": None if distance is None else round(distance),
        "distance_label": label,
        "campus_name": CAMPUS_NAME,
        "address": CAMPUS_ADDRESS,
        "radius_metres": CAMPUS_RADIUS_METRES,
        "message": message,
    }


def _message(verdict, label):
    if verdict == LocationVerdict.ON_CAMPUS:
        return f"You are on the {CAMPUS_NAME} campus ({label} from the registry)."
    if verdict == LocationVerdict.OFF_CAMPUS:
        return (
            f"You are {label} from campus. Attendance is recorded at "
            f"{CAMPUS_ADDRESS}."
        )
    if verdict == LocationVerdict.IMPRECISE:
        return (
            "This location fix is too broad to confirm the campus. "
            "Turn on precise location and try the check again."
        )
    return "Location is still missing. Allow location access and check again."
