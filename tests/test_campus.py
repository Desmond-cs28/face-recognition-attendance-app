"""Campus geofence and filename names."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from constants import CAMPUS_LATITUDE, CAMPUS_LONGITUDE, LocationVerdict
from geo import campus_verdict
from names import display_name


PRECISE_ACCURACY = 20
BROAD_ACCURACY = 5000
ACCRA_LATITUDE = 5.6037
ACCRA_LONGITUDE = -0.1870


def test_campus_centre_is_on_campus():
    verdict, distance = campus_verdict(CAMPUS_LATITUDE, CAMPUS_LONGITUDE, PRECISE_ACCURACY)
    assert verdict == LocationVerdict.ON_CAMPUS
    assert distance < PRECISE_ACCURACY


def test_accra_centre_is_off_campus():
    off_campus_metres = 10_000
    verdict, distance = campus_verdict(ACCRA_LATITUDE, ACCRA_LONGITUDE, PRECISE_ACCURACY)
    assert verdict == LocationVerdict.OFF_CAMPUS
    assert distance > off_campus_metres


def test_a_broad_fix_cannot_confirm_campus():
    verdict, _distance = campus_verdict(CAMPUS_LATITUDE, CAMPUS_LONGITUDE, BROAD_ACCURACY)
    assert verdict == LocationVerdict.IMPRECISE


def test_missing_coordinates():
    verdict, distance = campus_verdict(None, None, None)
    assert verdict == LocationVerdict.MISSING
    assert distance is None


def test_display_name_cleans_the_photo_filename():
    assert display_name("Comfort _Akorfa.jpg") == "Comfort Akorfa"
    assert display_name("Desmond_Zambogunaa.jpg") == "Desmond Zambogunaa"
    assert display_name("Edna-Boateng.jpeg") == "Edna Boateng"
