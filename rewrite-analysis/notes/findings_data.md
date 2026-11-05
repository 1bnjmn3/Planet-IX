# Data-layer findings (for the paper)

All numbers from the MPC Distant snapshot bundled in `data/raw/` (6,536 orbits).

1. **The original MPC parser misreads most of the file.** `Planet_IX/planetx_helpers.py::load_mpc_distant`
   pulls every number out of each line with a regex and takes elements by position. Packed designations
   and epochs contain digits (e.g. `K12VB3P` = 2012 VP113, `K25BL` = epoch 2025-11-21), which shift the
   positions. Result: of 6,536 orbits, 5,027 are rejected by its own sanity checks (77% silently dropped),
   and 214 of the 1,509 it keeps have wrong a, e or i. It finds 9 objects with a > 250 AU, q > 30 AU; the
   correct number is 35. Every provisional-designation ETNO (2012 VP113, 2013 SY99, 2017 OF201, ...) is lost. This parser feeds `Planet_IX/2026-01-22-file-1.ipynb`.
2. **No single definition of the sample.** The original scripts use at least three datasets
   (`live_mpc_data.csv` 71 rows, `smart_dataset.csv` 39, `processed_etnos.csv` 38) with different, undocumented
   cuts (smart_dataset is roughly a > 150, q > 38). The rebuild defines named cuts in one place.
3. **Old smart_dataset vs. canonical sample:** all 39 objects are in the snapshot; 3 (2019 EE6, 2020 KW54,
   2020 MQ53) fail the one-year arc quality cut.
4. **Duplicated raw data:** the two `distant_extended.dat` copies differ (different orbit-solution dates for
   some objects); the newer is used.
