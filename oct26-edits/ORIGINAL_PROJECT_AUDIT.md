# What was wrong with the original Planet IX project

A frank audit of the original code and claims, as of commit `4a23a3d` on `main`.

## Bottom line

The original repository says it found a secular "warp" in the outer Solar System and a
**1.65 Earth-mass "Planet Y"** that causes it. Neither claim holds up:

- **The 1.65 Earth masses is a product of two made-up constants.** Multiply either constant by 10 and
  the mass changes by a factor of 10. The fit's own error bar is ±16.4 Earth masses, and the fitted curve
  matches the data no better than a flat line.
- **The "warp" was measured as the peak of an inclination histogram.** That is the typical inclination of
  the sample, not a tilted plane. When the mean orbital plane is measured properly, it is tilted 4.6°, and
  random orbit orientations give that much tilt 21% of the time.
- **The one significant clustering result came from a parser bug.** The notebook's p = 0.006 comes from 20
  objects that survived a parser that drops 77% of the file. With all 71 objects the same test gives
  p = 0.12. With the 38-object cut used elsewhere it gives p = 0.41.
- **The "confidence" numbers are always high.** The 99–100% bootstrap "probability the warp is real" is
  99% even when the node angles are replaced with random numbers.

The honest result from the original data is a null one. There is no statistically significant alignment
of perihelia or orbital planes in 20–71 known distant TNOs, and no constraint on any planet's mass.

Every number below comes from rerunning the original scripts on the data files committed in the repo,
plus the short checks in [`audit/`](audit/) (see "How to reproduce" at the end).

---

## 1. The "Planet Y" mass (scripts 23, 25, 33–36)

### 1.1 The torque constants are invented, and the mass depends on them exactly

`Benjamins code/25WarpInverter.py:44-46` (copied from `23SecularAnalytic.py:41-42`):

```python
torque_giants = 1.0 / (particle_a**3.5) * 1e8   # "Scaling factor for inner system rigidity"
torque_p9 = (planet_m / m_sun) * (particle_a**2) / (planet_a**3) * 1e10
```

`1e8` and `1e10` have no physical derivation. In this model only the product of mass and the
`1e10 / 1e8` ratio can be identified, so the "measured" mass is set by whatever ratio was typed in:

| giant constant | planet constant | fitted inclination | fitted mass |
|---|---|---|---|
| 1e8 | 1e10 (original) | 17.11° | **1.65 M⊕** |
| 1e8 | 1e9 | 17.11° | 16.5 M⊕ |
| 1e8 | 1e11 | 17.11° | 0.165 M⊕ |
| 1e9 | 1e10 | 17.11° | 16.5 M⊕ |

The power laws are also wrong. The real Laplace–Lagrange nodal precession driven by an outer planet
scales as a^1.5 / a_p^3, not a^2 / a_p^3. And the ratio of the code's two torques overstates the outer
planet's pull by **180× at 250 AU, 228× at 400 AU and 279× at 600 AU** compared with the textbook ratio
(m_p a^5 / (a_p^3 Σ m_j a_j^2), using the giant planets' Σ m_j a_j² = 38,023 M⊕ AU²). If you refit with the
physical ratio you get 747 ± 18,000 M⊕, which tells you nothing.

### 1.2 The fit doesn't constrain anything

- `curve_fit`'s own covariance gives **mass = 1.65 ± 16.4 M⊕ (1σ)**. That range includes zero and the
  20 M⊕ upper bound.
- Above about 250 AU the model is essentially flat (torque ratio 61 at 250 AU, 813 at 400 AU), so the
  curve just returns the mean inclination. Residual sum of squares is 36.86 for the model and 36.93 for a
  constant line at the mean. The two-parameter "physics" adds nothing.
- N = 8. The comment at line 14 says "ISOLATE THE 9 'WARP' OBJECTS", but the cut finds 8.

### 1.3 It's circular: select by inclination, then fit inclination

`25WarpInverter.py:18-22` keeps only objects with **14° < i < 22°** and then fits their inclination.
The answer has to land inside the window. To check, I replaced the 8 inclinations with uniform random
values in 14–22° (no planet at all) and refit 2,000 times. The fitted inclination fell inside 14–22°
**100%** of the time (median 18.2°). The 17.1° "detection" is the midpoint of the selection window.

### 1.4 The 8 "warp" objects don't share a plane

An orbital plane needs inclination *and* node. The 8 objects selected in script 25 have nodes of
35°, 109°, 131°, 158°, 218°, 219°, 322° and 329°. They are spread all around the sky. They only share
a similar tilt, which most distant TNOs do.

### 1.5 Swapping models until the answer looks right

- `33Inc65Solver.py` uses the same constants with a 65° planet. The largest mass it finds anywhere in
  100–2000 AU is **0.041 M⊕**, and it prints that a 1 M⊕ and a 5 M⊕ planet are both "impossible".
- `35PlanetOptimizer.py` then switches to a different "J2 balance" formula (`34PlanetYWarpCheck.py`) for
  Planet Y only, which gives 1 M⊕ at **110 AU**. It keeps the old formula for Planet 9, which needs to be
  at **225,795 AU** (`35PlanetOptimizer.py:28-44`). `36FinalOptimizer.py` labels these "Plausibility: HIGH"
  and "LOW". So the two hypotheses were judged with two incompatible models, and the "target" both are
  fitted to (15.7°) is not a warp (see §2.1).
- `34PlanetYWarpCheck.py:42` uses a linear weighted average of inclinations, which is a small-angle
  approximation, with a 65° planet.
- The literature values the "optimizers" are tuned to reproduce are hard-coded (1 M⊕ for Planet Y, 5 M⊕
  for Planet 9: `35PlanetOptimizer.py:58-63`). The scripts solve for a distance that makes the assumed mass
  work, then present it as a result.

## 2. The "warp" itself (scripts 18, 21, 29, 30)

### 2.1 A histogram peak is not a warp

`29NakedEyeHistogram.py:23-29` takes the peak of a KDE of inclinations (**15.7°**) and calls it the warp
angle. Script 30 uses this as the planet's orbital inclination. But the peak of an inclination histogram is
just the typical inclination of the population. It ignores the node completely, so it can't detect a tilted
plane.

- The peak is not precise: bootstrap 95% range **10.6°–19.0°**.
- A plain scattered disk with random nodes and a sin(i)·Gaussian(σ = 15°) inclination distribution, with
  the same N = 39, has a median peak of **15.4°** (16–84%: 12.4°–19.3°). The "warp" is what you'd see with
  no warp at all.

The correct measurement is the mean angular-momentum (pole) vector. For the same 39 objects it gives a mean
plane tilted **4.6°** (bootstrap 16–84%: 2.7°–7.8°) at node 132°. Keeping the real inclinations and
scrambling the nodes gives a tilt this large or larger **20.5%** of the time. (That null even ignores survey
bias, which makes it generous to the warp hypothesis.)

### 2.2 The "planes" are clustering artefacts

- `21WarpStressTest.py:41-54`: GMM model selection over 1–5 components, with full covariance in 3 dimensions,
  on **N = 20** objects. BIC keeps falling up to the last value tried (196 → 154 → 156 → 124 → 68), so it
  picks 5, the edge of the search. That's 49 free parameters for 20 points. The "Planet 9 plane"
  (i = 17.9°, Ω = 114.2°) is **9 objects**, and the "Planet Y plane" (i = 11.2°, Ω = 212.3°) is **4 objects**.
  One "plane" is a single object. When I scrambled the nodes in 50 catalogs, BIC chose 5 components in **50/50**.
  This procedure always "detects" 5 planes.
- `18WarpSignature.py:38`: the DBSCAN `eps = 0.5` is arbitrary. Across eps = 0.3–0.7 the "primary warp"
  moves from (16.0°, 112°) to (17.9°, 147°), and its size from 11 to 47 objects. At eps = 0.5 the
  "cluster" holds 42 of the 71 objects, with nodes from 91° to 219°.
- `18WarpSignature.py:63-64` averages node angles arithmetically. That's wrong for circular data (it only
  shifts 154.3° to 153.9° here, but it would fail for any cluster that crosses 0°).

### 2.3 The final sky map mixes numbers from different analyses and draws the Galaxy in the wrong place

- `30FinalSkyTrack.py:7-8` combines i = 15.7° (the histogram peak, script 29, `smart_dataset.csv`) with
  Ω = 114° (a 9-object GMM cluster, script 21, `live_mpc_data.csv`). Those are two different samples and
  two different methods. The comment says it uses this node because it "was the strongest signal".
- `30FinalSkyTrack.py:35` passes the Galactic plane's tilt *relative to the celestial equator* (62.87°, node
  RA 282.85°) to a function that treats its inputs as *ecliptic* elements and then rotates by the obliquity.
  The plotted "Galactic plane" is off from the real one by up to **72°** (median 42°).
- The "Prime Observation Window" bands (`:48-49`) are hand-picked RA ranges, and the title says "Planet Y"
  while the node is the one assigned to "Planet 9".
- `20TreasureMap` (no `.py` extension) is an earlier copy of this script that uses a third set of numbers
  (16.8°, 154.3°) from script 18.

## 3. Statistics

### 3.1 Results were carried forward by hand, not computed

Numbers from one run were pasted into later scripts as constants, so the pipeline never checks itself:

| constant | where it came from | hard-coded in |
|---|---|---|
| i = 17.9°, Ω = 114.2° / 11.2°, 212.3° | 9- and 4-object GMM clusters (21) | `22ModelFaceoff.py:8-11`, `23SecularAnalytic.py:6-7`, `26PoleMapper.py:54-55` |
| i = 16.8°, Ω = 154.3° | DBSCAN eps = 0.5 (18) | `19NBodyCheck.py:8-9`, `20TreasureMap` |
| 15.7° | KDE peak (29) | `30FinalSkyTrack.py:7`, `32`, `33`–`36` |
| "camp" angles, "top anomalies" | earlier logs | `07FinalVisual.py:16-17`, `03Visualization.py:47` |

### 3.2 No correction for trying many analyses

The repo runs dozens of tests on the same 20–71 objects. They use different cuts (a > 100, 150, 230, 250
AU; q > 30, 35, 38, 40 AU; a Tisserand band), different methods (KS, Rayleigh, autoencoder, random forest,
DBSCAN, GMM, PCA, mutual information) and different targets (ϖ, then i, then Ω). `15LimitOptimizer.py:78-103`
alone scans 21 a-cuts × 3 filters = 63 configurations. When the original ϖ-clustering tests came back null
(§3.3), the project switched to a new statistic (the "warp") without accounting for the earlier tries.
Nothing anywhere corrects for multiple comparisons.

### 3.3 The project's own clean tests found nothing, and that wasn't the headline

On the 38-object sample (a > 230, q > 30):
- `04Statistics.py`: KS p = **0.195**.
- `05CircularStatistics.py`: Rayleigh p = **0.405**; Monte Carlo p = **0.406**.

These are the correct kind of test, and they say ϖ is consistent with uniform. (`04` also applies a
linear KS test to a circular angle, so the result depends on where 0° is. Kuiper's test is the circular
version.)

### 3.4 "Bayesian bootstrap confidence" isn't a probability that the warp is real

`24BayesianBootstrap.py:64,73` and `28ExpandedWarpTest.py:98-102` count the fraction of bootstrap resamples
in which *any* GMM component has a mean inclination in 12°–22° (or 25°), then print it as "Probability that
the 17.9 deg Warp is REAL". It isn't Bayesian either (a Bayesian bootstrap uses Dirichlet weights). The mean
inclination of the whole N = 20 sample is 17.8°, so even a one-component fit passes. Results:

- real data: **100%**
- same inclinations with **random nodes** (no plane at all): **99%**

### 3.5 Other statistical errors

- `31UnbiasedPca.py:74-76` uses silhouette > 0.5 as a "validity check". The 0.553 winner is 2 clusters with
  N = 2 and N = 37. The 2 are high-inclination outliers (mean i = 118°). PCA and clustering just separate
  outliers from everything else.
- `32BiasEmulator.py:28-55` "rejects survey bias" by comparing the data with a made-up null: Rayleigh(σ = 10°)
  inclinations, "declination ≈ i·sin(phase)", and boxes in RA/Dec drawn by hand. Rejecting it (AD = 29.2)
  only shows the real distribution isn't Rayleigh(10°). It says nothing about a warp.
- `09HypothesisTest.py`, `10`, `12`, `14`, `15`: random-forest classifiers are trained on single objects
  whose only difference between classes is ϖ. The script then averages the per-object probability and reads
  it against fixed thresholds (0.4/0.6, 0.45/0.65). Those thresholds aren't calibrated. I fed 200 purely
  random-ϖ catalogs to the script-09 classifier, and they scored a median of **0.374** (5–95%: 0.30–0.45).
  The real data scored 0.414, which is inside the random range.
- `Planet_IX/2026-01-22-file-1.ipynb` reports **P(Planet-X imprint | real catalog) = 2.1 × 10⁻⁷⁹**. Across a
  strength sweep that probability flips between exactly 1.0000 and 0.0000. That's a saturated logistic
  regression extrapolating, not a measurement. The synthetic selection function (`planetx_helpers.py:133-140`)
  depends only on q and i, not on where on the sky an object is. So the null can't reproduce the
  sky-position bias that matters most for ϖ clustering.
- No script sets a random seed for its simulations (`19`, `22`, `02`, `09`, ...), so results change
  from run to run.

## 4. Simulations (scripts 19, 22)

- **Far too short.** For a test particle at a = 300 AU, a 5 M⊕ planet at 500 AU drives nodal precession with
  a period of about **1.9 Gyr**. `19NBodyCheck.py:55` integrates for 500 kyr, which is **0.026%** of one cycle.
  A secular warp can't appear in that time.
- **Wrong units.** `22ModelFaceoff.py` never sets `sim.units`, so REBOUND uses G = 1. Then
  `sim.integrate(250000)` covers 250,000/2π ≈ **39,800 years**, not the 250,000 years the comment says.
- **Wrong comparison.** `19NBodyCheck.py:86` measures distance in (i, Ω) with plain Euclidean distance and
  degrees, with no wrap-around at 360°. It also puts the planet's node 180° away from the target and then
  counts particles that are "captured" near the target.
- The initial particles (e = 0.1–0.5, a = 150–400) have perihelia of 75–360 AU, which doesn't resemble the
  observed population (q ≈ 30–80 AU).

## 5. Data

- **The notebook's parser loses 77% of orbits.** `Planet_IX/planetx_helpers.py:7-46` (`load_mpc_distant`)
  pulls every number out of each line with a regex and picks elements by position. Packed designations like
  `~00U1` contain digits, which shift the positions. Of 6,536 lines, **5,241** are misaligned, and only
  **1,509** rows survive its sanity checks. After the a > 150, q > 30 cut it keeps **20 of the 71** real
  objects, and one of those 20 has wrong elements. This is where the notebook's **R = 0.49, p = 0.006**
  ϖ-clustering result comes from. On the correctly parsed 71 objects, R = 0.17 and p = 0.12. (The data
  rebuild finds the same thing: [`notes/findings_data.md`](notes/findings_data.md).) The whitespace parser in
  `01DataPrep&Features.py` happens to give the right values for this file, but it would break on any line
  with a blank H or G field.
- **There are four datasets with undocumented, inconsistent cuts.** `processed_etnos.csv` (38 rows, a > 230,
  q > 30), `live_mpc_data.csv` (71 rows, a > 150, q > 30, from script 11), `smart_dataset.csv` (39 rows,
  a > 150, q > 38, from JPL), and the notebook's 20. The "Golden Path" in the README mixes them: steps 2–3 use
  `smart_dataset.csv`, but step 4 (script 25) reads `live_mpc_data.csv`.
- `27JPLSmartHarvester.py:42` computes the Tisserand parameter, and the header calls it the "Tisserand
  Filter", but it's never applied. The filter is just q > 38 (`:51`).
- `13NASAJPLHarvester.py:69` deduplicates MPC and JPL rows on exact (a, e, i, ϖ). Orbits from two sources
  with different epochs never match exactly, so duplicates survive. The MPC IDs stay packed (`E8209`) while
  the JPL ones are full names (`148209 (2000 CR105)`).
- **Duplicated raw data.** `distant_extended.dat` exists in two places (`Benjamins code/` and `Planet_IX/`).
  They aren't identical: 6,536 vs 6,534 lines, with different orbit solutions for some objects (e.g. Bienor,
  Orcus). Nothing records which one any result used.
- There's no observation-arc or orbit-quality cut. Short-arc orbits are treated the same as decades-long ones.

## 6. Code

- **Helpers are redefined, with conflicting signatures.** In `Planet_IX/planetx_helpers.py`, `ang_wrap_deg` is
  defined 3 times (lines 48, 350, 396), `compute_varpi_deg` 3 times with **two different signatures**
  (`(data)` at 71 and 353, `(Omega, omega)` at 399), and `_floats_from_line`, `load_mpc_distant`,
  `circ_mean_deg` and `circ_R` twice each. Python keeps the last definition, so after `import`,
  `compute_varpi_deg(data)` raises `TypeError`. The notebook repeats the same pattern across its cells.
  Most of the numbered scripts also copy their own loaders, filters and GMM loops.
- **No tests, no requirements file, no package.** Scripts `02`, `06` and `08` import `tensorflow`, which the
  README's install line doesn't include.
- Scripts call `exit()` and block on `plt.show()`, so they can't be imported or run in batch.
- Paths: `filefuser.py:5` hard-codes `~/Desktop/VS_Planet IX/Benjamins code`. `01DataPrep&Features.py` has a
  `&` in its filename. `20TreasureMap` has no extension.
- `.DS_Store` files are committed in two places.

## 7. Documentation

- `Benjamins code/readme.md:1` begins "Here is the comprehensive `README.md` for your project. You can save
  this file…". That's chatbot output pasted in verbatim.
- **None of the script names in the README exist.** It says `27_Smart_Harvester.py`, `31_Unbiased_PCA.py`,
  `29_Naked_Eye_Histogram.py`, `25_Warp_Inverter.py`, `30_Final_Sky_Track.py`, `11_Live_MPC.py`,
  `15_Autoencoder.py`, `17_Nuclear_Bias.py`, and so on. The real files are `27JPLSmartHarvester.py` etc., and
  "15_Autoencoder" is actually `02AutoencoderModel.py` (15 is the limit optimizer).
- It says "all 31 scripts". There are 35 numbered `.py` scripts plus `20TreasureMap`.
- The key values are blank where math formatting got stripped: "stable TNOs with  AU,  AU", "a probability
  spike at ****", "Derived Mass ()".
- It says `17_Nuclear_Bias.py` "proved the clustering was just survey bias". The real `17SurveyBiasMapper.py`
  only plots sky positions next to hand-drawn "survey window" boxes and computes nothing.
- It calls the output a "discovery", "detection", "Mathematical proof" (script 23, whose own comments say
  "toy model approximation") and "publication-ready map".
- About 22,000 lines of repo-dump text are committed (`full_codebase_numbered.txt` 10,167 lines,
  `text stuff/combined_files.txt` 11,761 lines, plus 17 `* copy.txt` files), along with three near-identical
  notebooks (`FullJupyterNotebook.ipynb`, `FullJupyterNotebook_updated.ipynb`, `Notebook.ipynb`).

## What survives

- The data source is real (MPC Distant / JPL SBDB), and scripts `04`/`05` used appropriate tests. Their null
  results are the defensible finding.
- The pole-vector maths in `26PoleMapper.py` and the Kepler solver in `17SurveyBiasMapper.py` are reasonable, but
  `26` labels ecliptic pole coordinates as "RA/Dec".
- The bootstrap and Monte Carlo ideas in the helpers are sound. They just ran on the wrong 20
  objects.

## How to reproduce

From the repository root, with `numpy scipy pandas scikit-learn` installed:

```bash
python oct26-edits/audit/audit_parse.py   # §5 parser loss, ϖ clustering by sample
python oct26-edits/audit/audit_warp.py    # §1 constants, error bars, circularity null
python oct26-edits/audit/audit_stats.py   # §2-3 KDE peak, mean plane, bootstrap, GMM, DBSCAN, PCA (a few minutes)
python oct26-edits/audit/audit_misc.py    # §2.3 Galactic plane, §4 timescales/units, §3.5 classifier null
```

The figures quoted from the original scripts (BIC values, 1.65 M⊕, 15.7°, silhouette 0.553, AD = 29.2,
KS/Rayleigh p-values, script 33–36 outputs) come from running them unchanged with `MPLBACKEND=Agg` in
`Benjamins code/`. Random-number results can move slightly between runs.
