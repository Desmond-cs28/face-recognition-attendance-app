"""Turn a photograph filename into a student name."""

import os

from constants import EMPTY_TEXT


def display_name(filename):
    stem = os.path.splitext(os.path.basename(filename))[0]
    spaced = stem.replace("_", " ").replace("-", " ")
    parts = [part for part in spaced.split(" ") if part != EMPTY_TEXT]
    return " ".join(parts)
