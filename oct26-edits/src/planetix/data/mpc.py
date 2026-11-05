"""Parser for MPC orbit files in the MPCORB fixed-width format (e.g. Distant.txt).

Column layout follows the MPC export format documentation
(https://minorplanetcenter.net/iau/info/MPOrbitFormat.html). Columns are
1-based and inclusive in that document; the slices below are 0-based.

The previous code split each line with a float regex. That breaks on packed
provisional designations such as ``K12VB3P`` (2012 VP113), whose digits are
picked up as extra numbers and shift every orbital element by one or more
positions. Reading fixed columns avoids that.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from planetix.angles import varpi_deg

# (name, start, stop) as 0-based half-open slices
_FIELDS = [
    ("packed", 0, 7),
    ("H", 8, 13),
    ("G", 14, 19),
    ("epoch_packed", 20, 25),
    ("M", 26, 35),
    ("omega", 37, 46),
    ("Omega", 48, 57),
    ("i", 59, 68),
    ("e", 70, 79),
    ("n", 80, 91),
    ("a", 92, 103),
    ("U", 105, 106),
    ("reference", 107, 116),
    ("n_obs", 117, 122),
    ("n_opp", 123, 126),
    ("arc", 127, 136),
    ("rms", 137, 141),
    ("computer", 150, 160),
    ("name", 166, 194),
    ("last_obs", 194, 202),
]

_FLOAT_FIELDS = ["H", "G", "M", "omega", "Omega", "i", "e", "n", "a", "rms"]
_INT_FIELDS = ["n_obs", "n_opp"]

_CENTURY = {"I": 1800, "J": 1900, "K": 2000}


def _unpack_digit(c: str) -> int:
    """MPC packed digit: 0-9, A-Z = 10-35, a-z = 36-61."""
    if c.isdigit():
        return int(c)
    if c.isupper():
        return ord(c) - ord("A") + 10
    return ord(c) - ord("a") + 36


def unpack_epoch(packed: str) -> float:
    """Convert a packed MPC epoch such as ``K25BL`` to a Julian date (TT, 0h)."""
    packed = packed.strip()
    if len(packed) != 5 or packed[0] not in _CENTURY:
        raise ValueError(f"Bad packed epoch {packed!r}")
    year = _CENTURY[packed[0]] + int(packed[1:3])
    month = _unpack_digit(packed[3])
    day = _unpack_digit(packed[4])
    return _jd_from_calendar(year, month, day)


def _jd_from_calendar(year: int, month: int, day: float) -> float:
    """Julian date at 0h TT of a Gregorian calendar date."""
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + b - 1524.5


def _arc_days(arc: str) -> float:
    """Observed arc in days from '1990-2026' (years) or '  36 days'.

    A year span is only approximate: '2025-2026' can mean anything from a few
    days to almost two years. Use ``_arc_days_min`` for quality cuts.
    """
    arc = arc.strip()
    m = re.fullmatch(r"(\d+)\s*days", arc)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(\d{4})-(\d{4})", arc)
    if m:
        return (int(m.group(2)) - int(m.group(1))) * 365.25
    return np.nan


def _arc_days_min(arc: str, last_obs: str) -> float:
    """Shortest arc consistent with the arc field and the last-observation date.

    For '  36 days' this is the arc itself. For a year span 'Y1-Y2' the first
    observation can be as late as 31 December Y1, so the arc is at least
    ``last_obs - (Y1-12-31)`` days.
    """
    arc = arc.strip()
    m = re.fullmatch(r"(\d+)\s*days", arc)
    if m:
        return float(m.group(1))
    m = re.fullmatch(r"(\d{4})-(\d{4})", arc)
    if m and re.fullmatch(r"\d{8}", last_obs.strip()):
        lo = last_obs.strip()
        last = _jd_from_calendar(int(lo[:4]), int(lo[4:6]), int(lo[6:]))
        return float(max(last - _jd_from_calendar(int(m.group(1)), 12, 31), 0.0))
    return np.nan


def read_mpcorb(path: str | Path) -> pd.DataFrame:
    """Read an MPCORB-format file into a DataFrame.

    Returns one row per orbit with angles in degrees and distances in AU,
    plus derived columns q, Q, varpi, epoch_jd, arc_days and arc_days_min. Header lines and
    lines too short to hold the orbital elements are skipped.
    """
    rows = []
    with open(path, encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if len(line) < 103 or not line[92:103].strip():
                continue
            rows.append({name: line[lo:hi].strip() for name, lo, hi in _FIELDS})
    if not rows:
        raise ValueError(f"No MPCORB records found in {path}")

    df = pd.DataFrame(rows)
    for col in _FLOAT_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in _INT_FIELDS:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    # U is 0-9 or a letter flag (E = assumed eccentricity, D = double designation, ...)
    df["U"] = df["U"].replace("", pd.NA)

    df["epoch_jd"] = df["epoch_packed"].map(unpack_epoch)
    df["arc_days"] = df["arc"].map(_arc_days)
    df["arc_days_min"] = [_arc_days_min(a, l) for a, l in zip(df["arc"], df["last_obs"])]
    number = df["name"].str.extract(r"^\((\d+)\)", expand=False)
    df["name"] = df["name"].str.replace(r"^\(\d+\)\s*", "", regex=True)
    # Numbered objects are identified by number, others by provisional designation.
    df["designation"] = number.fillna(df["name"])
    df["q"] = df["a"] * (1.0 - df["e"])
    df["Q"] = df["a"] * (1.0 + df["e"])
    df["varpi"] = varpi_deg(df["Omega"], df["omega"])
    return df.drop(columns=["epoch_packed", "arc"])
