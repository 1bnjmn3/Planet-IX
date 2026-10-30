# Survey-bias and ML findings (for the paper)

All numbers come from `python scripts/run_bias.py` (seeded; it takes about 6 minutes). The outputs are in `results/bias/`.
The data are the bundled MPC snapshot (6,536 orbits) and the OSSOS v11 survey characterisations in
`data/external/ossos_v11/` (142 survey blocks from OSSOS, CFEPS, HiLat and MA).

## Methods in one paragraph each

**Two independent selection models.** (1) *Empirical footprint* (after Brown 2017): every known TNO with
q > 30 AU (4,855 after removing the tested sample) is placed at its discovery position and magnitude. Discovery
dates come from the half-month letter of the provisional designation for 6,463 of the 6,536 objects; the other 73
fall back to the first observation implied by the arc. A simulated object at (lon, lat, V) gets a weight equal to
the number of those discoveries within 5 deg that were at least as faint. (2) *Characterised surveys*: the published
OSSOS-family pointings, epochs, filling factors and efficiency functions (detection vs. magnitude and rate,
characterisation limit, tracking fraction), re-implemented in numpy. This model is valid only for objects those
surveys found.

**Per-object selection functions.** Each object keeps its own a, e, i and H. Its (Omega, omega, M) are drawn
uniformly and weighted by the selection model. M is sampled only inside the window around perihelion where the
object can be bright enough to detect, and the weights are rescaled by that window's share of the orbit. This is
exact, and it made the survey model about 100× cheaper.

**Bias-aware null.** Under H0 the orientations are intrinsically uniform. Each object's (Omega, omega) is then
drawn from its own selection function, independently of the other objects. Statistics: the resultant length R of
varpi, Omega and omega, of the 3-D perihelion direction, and of the orbit pole. Also D, the distance of the mean
vector from the null's expected mean vector. D asks whether the sample is clustered *beyond* what selection
produces, in both direction and strength. `min_p` is the smallest p-value, calibrated on the same mock samples
(look-elsewhere corrected across the 10 statistics). Each test uses 20,000 mock samples.

## Validation (real data)

1. **Geometry.** The two-body positions from the published OSSOS orbits reproduce the reported RA/Dec of all
   1,152 characterised detections with a median error of 0.034 deg (99th percentile 0.108, maximum 0.119). That
   is far smaller than any field (each is at least 1 deg across).
2. **Survey model.** Evaluated at their real orbits and epochs, 94.8% of the characterised detections come out
   detectable (OSSOS 95.9%, MA 100%, HiLat 95.2%, CFEPS 88 to 93%). The misses have not been examined one by one. The likely causes are
   block edges (each block has one reference epoch, but detections come from images taken days earlier or later)
   and objects near the characterisation limit.
3. **The two models disagree for single objects.** For the 9 characterised ETNOs, the median total-variation
   distance between the footprint's and the survey model's S_j(varpi) is 0.20. The survey model's own distance from
   uniform is 0.19. For the 3 objects with a > 250 AU it is 0.35 to 0.40
   (`figures/selection_varpi_ossos_vs_footprint.png`). The Brown-style footprint averages over every survey, while
   any one object was found by one survey. So the footprint is a coarse proxy, and p-values against it should not
   be quoted to two significant figures.

## Clustering significance

| sample | N | null | R_varpi p | D_varpi p | R_pole p | min_p (corrected) |
|---|---|---|---|---|---|---|
| ts14 (a>150, q>30) | 62 | uniform | 0.012 | 0.011 | 0.001 | 0.004 |
| ts14 | 62 | footprint | 0.020 | 0.002 | 0.009 | 0.007 |
| **bb16 (a>250, q>30)** | 31 | uniform | 0.185 | 0.191 | 0.034 | 0.112 |
| bb16 | 31 | footprint | 0.216 | 0.075 | 0.059 | 0.244 |
| detached (a>250, q>40) | 19 | uniform | 0.367 | 0.359 | 0.137 | 0.379 |
| detached | 19 | footprint | 0.381 | 0.236 | 0.184 | 0.566 |
| OSSOS-found ETNOs | 9 | OSSOS surveys | 0.709 | 0.641 | 0.807 | 0.755 |

Findings:

4. **With today's sample, the a > 250 AU ETNOs are not significantly clustered in varpi, with or without bias
   correction.** R_varpi = 0.234 (31 objects) gives p = 0.19 against uniform and p = 0.22 against the footprint
   null. The Batygin & Brown (2016) signal came from 6 objects. It has not grown with the sample: 31 objects with
   at least a one-year arc now dilute it.
5. **The clustering that survives is in the broader a > 150 AU sample.** It is significant against both nulls
   (corrected p = 0.004 uniform, 0.007 footprint). It does not go away with selection. With the footprint null, D
   (p = 0.002) is *smaller* than R (p = 0.020), because the selection-expected mean varpi points away from the
   observed one. Selection, as this model sees it, works against the observed cluster rather than creating it.
   Caveat: the 150 to 250 AU objects are more strongly perturbed by Neptune, so this is not direct evidence for a
   distant planet.
6. **Orbital-plane ("warp") clustering is marginal and weakens with bias.** For bb16, R_pole p = 0.034 against
   uniform and 0.059 against the footprint. The old scripts' 15.7 deg "warp" was never tested against a null.
7. **Sensitivity (bb16).** Across smoothing radius 3/5/10 deg, count vs. binary weighting, and dropping Plutinos
   from the footprint, R_varpi p stays in 0.19 to 0.22 and D_varpi p in 0.07 to 0.16. The conclusion does not
   depend on these choices.
8. **Shankman et al. (2017)-style test.** Using only the 9 ETNOs that the characterised surveys found, and those
   surveys' real selection, nothing is significant (all p > 0.2). This agrees with Shankman et al. (2017) and
   Napier et al. (2021). The sample is small, so this is consistent with no clustering, not proof of its absence.
9. **Sky coverage.** None of the 31 bb16 objects was discovered within 10 deg of the Galactic plane, and only 1.4%
   of all reference discoveries were. Discoveries crowd into longitudes 0 to 60 deg and 120 to 250 deg
   (`figures/sky_discoveries.png`). This is the footprint the null corrects for.

## Inclinations (old script 32)

10. Fitting a hot-disk distribution sin(i)·exp(-i²/2σ²) to bb16 gives σ = 14.5 deg (95% CI 12.5 to 17.5) when
    selection is ignored. With the footprint's P_j(i) it gives σ = 19.3 deg (95% CI 15.5 to 25.0). Ecliptic-heavy
    surveys hide high-i objects, so the intrinsic distribution is hotter than the observed one. Both fits are
    adequate (PIT KS p = 0.89 and 0.29). A standard hot disk explains the inclinations, and no warp is needed.
11. Old script 32 fixed σ = 10 deg in advance and pushed it through an invented box mask. Without the mask, its
    Anderson-Darling verdict flips with that fixed choice: p ≤ 0.002 for σ = 6, 10 and 20, and p ≥ 0.25 for
    σ = 15. (scipy caps these p-values to the range 0.001 to 0.25.) Its "reject the null" result came from the
    assumed width, not from the data. The old mask also mixed RA/Dec with ecliptic latitude (`dec ≈ i·sin(phase)`).

## Sky map (old script 17)

12. The old Milky Way curve `60 sin(lon - 280)` is off by up to 20.9 deg in latitude. The real Galactic plane
    peaks at ecliptic latitude 60.2 deg near longitude 0.3 deg, not 10 deg, and crosses the ecliptic at 90 and
    270 deg. The "survey windows" were hand-placed boxes with no source. The old script also plotted positions at
    the orbit epoch rather than at discovery.

## Machine learning (old script 02)

13. **The old autoencoder's "Planet 9 candidates" were noise.** We re-ran its recipe on bb16 over 10 seeds:
    synthetic training data with a, e and i bootstrapped independently, uniform varpi, min-max scaling, and a
    12-2-12 network. The top-3 list changes between seeds (mean Jaccard overlap 0.73; 5 different objects appear
    in it). The anomaly score tracks e and a (Spearman 0.25 and 0.23) and q (0.19), but hardly the varpi offset
    from the mean (0.09). It flags objects whose a-e combination is unusual, because the independent bootstrap
    broke the real a-e-q correlation. It does not flag unusual perihelion directions. The script had no
    calibration, so any reconstruction error looked like a detection.
14. **Calibrated replacements.** An autoencoder trained on bias-null mocks gives bb16 p = 0.075 (uniform null) and
    p = 0.26 (footprint null); ts14 gives 0.68 and 0.25. A classifier two-sample test (logistic regression on
    cos/sin features, with a and e matched exactly between the real objects and the null mocks) gives bb16 AUC 0.45
    and 0.52, p = 0.48 and 0.14; ts14 AUC 0.61 and 0.60, p = 0.003 and 0.010. The ML tests agree with the classical
    statistics: a signal at a > 150 AU, and none at a > 250 AU. With N = 31, two of 31 per-object autoencoder
    p-values fall below 0.05, which is what chance predicts (1.6 expected).

## Limitations to state in the paper

- The footprint mixes survey effort with the real density of objects (Brown 2017 makes the same approximation),
  and it does not reproduce single-survey selection (finding 3). DES and Pan-STARRS have no public
  characterisations in this format, so the full sample cannot be put through a true survey simulator.
- The survey model ignores the direction-of-motion cut, magnitude noise and chip gaps beyond the filling factor.
- Phase-angle effects (≤ 0.3 mag beyond 30 AU) and object colours (we use a mean V - r = 0.28) are approximate.
- MPC and JPL could not be reached from the build environment, so the bundled January 2026 snapshot is the sample.

## References

Batygin & Brown 2016, AJ 151, 22 · Brown 2001, AJ 121, 2804 · Brown 2017, AJ 154, 65 · Brown & Batygin 2019,
AJ 157, 62 · Bannister et al. 2018, ApJS 236, 18 · Petit et al. 2011, AJ 142, 131 · Petit et al. 2017, AJ 153, 236 ·
Alexandersen et al. 2016, AJ 152, 111 · Shankman et al. 2017, AJ 154, 50 · Napier et al. 2021, PSJ 2, 59 ·
Lopez-Paz & Oquab 2017, ICLR.
