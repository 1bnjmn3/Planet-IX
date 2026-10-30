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

## Outputs written by other workstreams
- `processed/nbody_forced_poles.csv`, `processed/secular_vs_nbody.csv`, `processed/inversion/`: dynamics
  scripts (`scripts/nbody_forced_poles.py`, `scripts/validate_secular.py`, `scripts/invert_planet.py`).

## external/ossos_v11/
Survey characterizations from the OSSOS v11 release, copied unchanged from
[github.com/ijiraq/jwst-tno-followup](https://github.com/ijiraq/jwst-tno-followup) at commit
`ff79267dee915c1e30190bdac98d709388987bc8` (2026), under the EUPL-1.2 licence (`LICENCE.txt`).
For each survey, the folder holds `pointings.list` (sky polygons, epochs, filling factors), one `.eff`
efficiency file per block (detection efficiency vs. magnitude and rate, plus the characterisation limit)
and `<survey>.detections` (the characterised discoveries with their orbit fits). The format is described
in `README.formats`, and the OSSOS release notes are in `__README__.txt`.

| survey | blocks | filter | faintest characterised V | detections | with a > 250, q > 30 | bb16 objects detectable* |
|---|---|---|---|---|---|---|
| OSSOS_v11 (Bannister et al. 2018) | 11 | r | 25.7 | 844 | 3 (o5m52, o5p060, o5t52) | 31 / 31 |
| CFEPS_g (Petit et al. 2011) | 43 | g | 24.0 | 184 | 0 | 25 / 31 |
| CFEPS_r (L3h block) | 1 | r | 24.0 | 14 | 0 | 25 / 31 |
| CFEPS_presurvey | 1 | R | 24.4 | 12 | 0 | 25 / 31 |
| HiLat (Petit et al. 2017) | 80 | r | 25.1 | 21 | 0 | 21 / 31 |
| MA (Alexandersen et al. 2016) | 6 | r | 24.9 | 77 | 0 | 31 / 31 |

\* The number of the 31 `bb16` objects for which at least one of 50,000 orientation clones (with that
object's a, e, i and H) would be detected by the survey. Every survey can see a > 250 AU objects, but only
near perihelion and only in its own fields. The per-object detection probabilities are 10^-6 to 10^-4,
which is why only OSSOS found any. A bias model built from these surveys is valid only for the 9 objects
with a > 150 AU that they actually found (see `planetix.bias.surveys`). For the full sample,
`planetix.bias.footprint` uses the MPC discovery footprint instead.

Copied data files only (no code). Known quirk: one Dec in `CFEPS_g/pointings.list` reads `-21:57:38:0`,
and the parser reads it as -21:57:38.
