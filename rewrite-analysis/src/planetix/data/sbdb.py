"""Client for the JPL Small-Body Database query API, with an on-disk cache.

API reference: https://ssd-api.jpl.nasa.gov/doc/sbdb_query.html

The result has the same columns as ``planetix.data.catalog.load_catalog`` so
either source can feed ``select_etnos``. JPL elements are heliocentric
osculating elements; MPC Distant.txt elements are also heliocentric, so the two
agree to rounding for the same epoch.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from planetix.angles import varpi_deg
from planetix.data.catalog import COLUMNS, DATA_DIR

API_URL = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

# SBDB field -> our column
_FIELD_MAP = {
    "pdes": "designation",
    "name": "name",
    "full_name": "full_name",
    "a": "a",
    "e": "e",
    "i": "i",
    "om": "Omega",
    "w": "omega",
    "q": "q",
    "ad": "Q",
    "ma": "M",
    "H": "H",
    "epoch": "epoch_jd",
    "n_obs_used": "n_obs",
    "data_arc": "arc_days",
    "condition_code": "U",
}


def build_query(a_min: float = 150.0, q_min: float = 30.0) -> dict:
    """Query parameters for all small bodies with a > a_min and q > q_min (AU)."""
    constraint = {"AND": [f"a|GT|{a_min}", f"q|GT|{q_min}"]}
    return {
        "fields": ",".join(_FIELD_MAP),
        "sb-cdata": json.dumps(constraint, separators=(",", ":")),
        "full-prec": "true",
    }


def fetch(
    a_min: float = 150.0,
    q_min: float = 30.0,
    cache_dir: str | Path | None = None,
    refresh: bool = False,
    timeout: float = 60.0,
    opener=urllib.request.urlopen,
) -> pd.DataFrame:
    """Download (or read from cache) every orbit with a > a_min and q > q_min.

    The raw JSON response is cached under ``cache_dir`` keyed by the query, with
    the download time recorded, so analyses are reproducible offline. Pass
    ``refresh=True`` to re-download.
    """
    params = build_query(a_min, q_min)
    cache_dir = Path(cache_dir or DATA_DIR / "raw" / "sbdb")
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    cache_file = cache_dir / f"sbdb_{key}.json"

    if cache_file.exists() and not refresh:
        payload = json.loads(cache_file.read_text())
    else:
        url = f"{API_URL}?{urllib.parse.urlencode(params)}"
        with opener(url, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
        if "fields" not in payload:
            raise RuntimeError(f"Unexpected SBDB response: {str(payload)[:300]}")
        payload["_retrieved_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        payload["_query"] = params
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload))

    return parse_response(payload)


def parse_response(payload: dict) -> pd.DataFrame:
    """Turn an SBDB query JSON response into a catalogue DataFrame."""
    df = pd.DataFrame(payload.get("data", []), columns=payload["fields"])
    df = df.rename(columns=_FIELD_MAP)
    for col in ["a", "e", "i", "Omega", "omega", "q", "Q", "M", "H", "epoch_jd", "arc_days"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["n_obs"] = pd.to_numeric(df["n_obs"], errors="coerce").astype("Int64")
    # SBDB does not report oppositions; approximate from the arc for a common schema.
    df["n_opp"] = pd.array(np.where(df["arc_days"] >= 365, 2, 1), dtype="Int64")
    df["U"] = df["U"].astype("string").str.strip()
    df["name"] = df["name"].fillna(df["full_name"].str.strip())
    df["varpi"] = varpi_deg(df["Omega"], df["omega"])
    df["arc_days_min"] = df["arc_days"]  # SBDB reports the exact arc
    return df[COLUMNS].reset_index(drop=True)
