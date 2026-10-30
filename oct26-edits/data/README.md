# Data

## raw/mpc_distant_2026-01.dat
A snapshot of the Minor Planet Center's `Distant.txt` export (MPCORB fixed-width format): 6,536 orbits of
objects beyond Jupiter, with orbit solutions computed up to 2026-01-22 and most epochs at 2025-11-21 (JD 2461000.5).
It is the newer of the two `distant_extended.dat` copies in the original repo (`Benjamins code/`); the copy
in `Planet_IX/` is an older download of the same file.

Live sources (`ssd-api.jpl.nasa.gov`, `minorplanetcenter.net`) can be refreshed with
`python scripts/build_etnos.py --sbdb`, which caches the raw JSON under `raw/sbdb/` with its download time.

## processed/etnos_<cut>_<source>.csv
Written by `scripts/build_etnos.py`. Load them in code with `planetix.data.load_etnos(cut)` rather than
reading the CSVs, so every analysis uses the same cuts.

| cut | definition | before quality cuts | after |
|---|---|---|---|
| `ts14` | a > 150 AU, q > 30 AU (Trujillo & Sheppard 2014) | 71 | 62 |
| `bb16` | a > 250 AU, q > 30 AU (Batygin & Brown 2016), **default** | 35 | 31 |
| `detached` | a > 250 AU, q > 40 AU (filter used by the original scripts) | 20 | 19 |

Quality cuts: observed arc of at least 365 days, and an MPC uncertainty parameter U of 0 to 9 (drops
letter flags such as `E`, assumed eccentricity).

Columns: designation, name, a, e, i, Omega, omega, varpi, q, Q, M, H, epoch_jd, n_obs, n_opp, arc_days, U.
Angles are in degrees, distances in AU, varpi = (Omega + omega) mod 360.
