"""The canonical ETNO sample: one place that defines which objects are analysed.

Every analysis should start from ``load_etnos()`` so that all results use the
same objects, the same orbit snapshot and the same, explicit cuts.

Columns (angles in degrees, distances in AU):
    designation, name, a, e, i, Omega, omega, varpi, q, Q, M, H,
    epoch_jd, n_obs, n_opp, arc_days, arc_days_min, U

``arc_days`` is the catalogue's arc (for MPC year spans, whole years times
365.25). ``arc_days_min`` is the shortest arc the catalogue allows, after
applying exact arcs from ``ARC_OVERRIDES``; the arc cut uses it.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from planetix.data.mpc import read_mpcorb

DATA_DIR = Path(os.environ.get("PLANETIX_DATA", Path(__file__).resolve().parents[3] / "data"))

# Bundled snapshot of the MPC Distant.txt export (orbits computed up to 2026-01-22).
DEFAULT_SNAPSHOT = DATA_DIR / "raw" / "mpc_distant_2026-01.dat"

# Exact observed arcs (first and last observation, from the MPC observation
# API) for orbits whose year-span arc field cannot tell whether the one-year
# cut is met. An entry applies only while the catalogue's last-observation
# date matches, so it goes stale on its own once the orbit is updated.
ARC_OVERRIDES = DATA_DIR / "raw" / "mpc_exact_arcs.csv"

# Named orbital cuts. Bounds are strict lower limits, in AU.
CUTS = {
    # Trujillo & Sheppard (2014, Nature 507, 471): the original ETNO definition.
    "ts14": {"a_min": 150.0, "q_min": 30.0},
    # Batygin & Brown (2016, AJ 151, 22): the clustered sample that motivated Planet Nine.
    "bb16": {"a_min": 250.0, "q_min": 30.0},
    # Perihelia beyond Neptune's direct reach: the "a > 250, q > 40" filter of the
    # original repo scripts (scripts 21, 25, 28), kept so old results can be compared.
    "detached": {"a_min": 250.0, "q_min": 40.0},
}

COLUMNS = [
    "designation", "name", "a", "e", "i", "Omega", "omega", "varpi", "q", "Q",
    "M", "H", "epoch_jd", "n_obs", "n_opp", "arc_days", "arc_days_min", "U",
]


def load_catalog(path: str | Path | None = None) -> pd.DataFrame:
    """Load every orbit in an MPCORB-format file (default: the bundled snapshot)."""
    df = read_mpcorb(path or DEFAULT_SNAPSHOT)
    df = apply_arc_overrides(df)
    return df[COLUMNS].reset_index(drop=True)


def apply_arc_overrides(df: pd.DataFrame, path: str | Path | None = None) -> pd.DataFrame:
    """Replace ``arc_days_min`` with the exact arc where it is known.

    ``path`` is a CSV with designation, first_obs, last_obs (ISO dates). A row
    is used only when its last_obs equals the catalogue's last-observation date.
    """
    path = Path(path or ARC_OVERRIDES)
    if not path.exists() or "last_obs" not in df:
        return df
    ov = pd.read_csv(path, dtype=str, comment="#")
    first = pd.to_datetime(ov["first_obs"])
    last = pd.to_datetime(ov["last_obs"])
    exact = dict(zip(zip(ov["designation"], last.dt.strftime("%Y%m%d")), (last - first).dt.days.astype(float)))
    keys = zip(df["designation"], df["last_obs"])
    df = df.copy()
    df["arc_days_min"] = [exact.get(k, v) for k, v in zip(keys, df["arc_days_min"])]
    return df


def select_etnos(
    df: pd.DataFrame,
    cuts: str | dict = "bb16",
    min_arc_days: float = 365.0,
    reliable_only: bool = True,
) -> pd.DataFrame:
    """Apply orbital and orbit-quality cuts.

    Parameters
    ----------
    cuts
        A key of ``CUTS`` or a dict with ``a_min`` and ``q_min`` (AU).
    min_arc_days
        Drop orbits observed over a shorter arc. One year removes
        single-opposition orbits whose elements are poorly constrained.
        Uses ``arc_days_min`` when present, so an MPC year span such as
        '2025-2026' passes only if the arc is known to be long enough.
    reliable_only
        Drop orbits whose MPC uncertainty flag is not a digit 0-9 (for example
        ``E``, where the eccentricity was assumed rather than fitted).
    """
    c = CUTS[cuts] if isinstance(cuts, str) else cuts
    keep = (df["a"] > c["a_min"]) & (df["q"] > c["q_min"]) & (df["e"] < 1.0)
    if min_arc_days:
        arc = df["arc_days_min"] if "arc_days_min" in df else df["arc_days"]
        keep &= arc >= min_arc_days
    if reliable_only:
        keep &= df["U"].astype("string").str.fullmatch(r"\d").fillna(False).astype(bool)
    return df[keep].sort_values("a").reset_index(drop=True)


def load_etnos(
    cuts: str | dict = "bb16",
    min_arc_days: float = 365.0,
    reliable_only: bool = True,
    path: str | Path | None = None,
) -> pd.DataFrame:
    """Load the canonical ETNO sample. See ``select_etnos`` for the cuts."""
    return select_etnos(load_catalog(path), cuts, min_arc_days, reliable_only)
