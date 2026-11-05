# Is this worth publishing? A referee-style evaluation, with iterations

Written 2026-10-30 against the skeleton in `paper/aas/ms.tex`. Each iteration states the paper as it
stood, the report a demanding referee at *The Planetary Science Journal* (PSJ) or *The Astronomical
Journal* (AJ) would write, the verdict, and what was changed in response. The bar is: would a
competent referee recommend acceptance after at most one round of revision?

---

## Iteration 0: the October mini-paper (`paper/planet9_bayes_factor.md`)

**Claim.** Bias-aware Bayes factor K = 1.7–2.5 for a planet against isotropy on 31 ETNOs with
a > 250 au; K = 17 on the objects known before 2016; ~120 ETNOs needed to reach K = 100.

**Referee report (summary).**

1. *Novelty.* The conclusion "the clustering is weak once bias is handled" is already published
   (Shankman et al. 2017; Napier et al. 2021; Bernardinelli et al. 2022). The newest analysis,
   Siraj, Chyba & Tremaine (2026, ApJ, arXiv:2604.25990), reports a conditional-likelihood
   significance that fell from 2.7σ to 1.9σ as their sample grew from 21 to 25 objects. A Bayes
   factor of ~2 that says the same thing adds little.
2. *The pre/post-2016 split is descriptive.* A drop from 17 to 2.5 could be ordinary sampling noise.
   The paper does not say how surprising it is if the planet is real.
3. *Selection.* Two thirds of the objects come from uncharacterised surveys. The empirical footprint
   is a stand-in, and the paper shows that swapping footprints moves ln K by ±1. That uncertainty is
   reported but not propagated.
4. *Planet model.* The anti-aligned von Mises shape is assumed, not derived.
5. *Forecast.* "About 120 objects" is not connected to any survey that will deliver them.

**Verdict: reject for PSJ/AJ (not enough new), acceptable as an RNAAS note.**

---

## Iteration 1: this thread's upgrades

Changes made (code in `scripts/forecast_and_history.py`, `scripts/nbody_population.py`,
`src/planetix/inference/lsst.py`, `src/planetix/dynamics/population.py`):

| referee point | analysis added | result |
|---|---|---|
| 2. Is the decline surprising? | **Calibrated out-of-sample (prequential) test.** Train the planet model on objects discovered up to year Y; compute the conditional Bayes factor K_new = p(x_new \| x_old, H1)/p(x_new \| H0) of the later objects; compare with 1,000 posterior-predictive skies (planet real, as the old objects imply) and 1,000 no-planet skies. | Y = 2015 (the year before Batygin & Brown 2016): ln K_new = −1.87, **p = 0.031** under the trained planet, typical of no-planet skies (25% are lower). Y = 2013: −4.89, p < 0.001. Y = 2014: −4.24, p = 0.001. Y = 2016: −2.37, p = 0.009. |
| 2. Full history | **Year-by-year evidence history** for both samples. | a > 250 au: ln K rose to **5.9 (K ≈ 360) with the 11 objects known by 2013**, then fell almost monotonically to 0.86 (K = 2.4). a > 150 au: peak ln K = 9.2 in 2016, 5.0 now. |
| 3. Propagate selection uncertainty | **Selection-marginalised Bayes factor.** Each selection model is weighted by how well it predicts the observed orientations under H0; continuous geometric interpolation between the two most different footprints. | Data strongly prefer the footprint built from a > 50 au objects. Marginalised **K = 5.6 (discrete), 4.2 (continuous)**. Still "substantial" at most on Jeffreys' scale; the conclusion does not depend on the selection choice. |
| 4. Assumed planet model | **N-body forward model**: REBOUND MERCURIUS, Sun + Neptune explicit + planet + 600 particles, 2 Gyr, with Jupiter, Saturn and Uranus as exact secular J2 rotations applied to every body. Three published planets (BB16, reference, BB21); a faster WHFast variant without Neptune as a cross-check. | Anti-alignment is reproduced: beyond 750 au the cluster centre lies 2–74° from Δϖ = 180° (2–40° for BB21 and the reference planet). But strong clustering (R = 0.5–0.7) appears only beyond about 750 au. Between 250 and 750 au, where 24 of the 31 objects lie, R is about 0.1–0.35. Used directly as H1 (orientation marginalised), the simulated planets give **K = 0.09–1.2**. |
| 5. Tie the forecast to a survey | **LSST selection model + information-rate forecast + OSSOS-anchored yield.** Exact per-object KL divergence on the grid; Occam terms calibrated on 160 injection runs; end-to-end LSST simulations as a check. | LSST sees 93% of orientation space per object vs 41% for today's survey mix, so each object is worth 26% more evidence. If the planet the data imply is real, K = 100 needs **47 LSST ETNOs for a median sky, 95 for 90% of skies** (63/133 with today's mix). OSSOS-anchored yield: **~230 ETNOs with a > 250 au in ten years** (68%: 134–409; 114–278 across population models). The decision should come **2–8 years into the survey**. |
| 1. Novelty | All of the above, plus a **pre-registered decision rule** for LSST (sample, pipeline version, thresholds, predicted K distributions) published before LSST ETNO orbits exist. | See ms.tex §5.4. |

**Referee report on iteration 1 (summary).**

*Strengths.* The prequential test is a genuinely new kind of evidence on this question: it asks
whether the hypothesis predicted data it had not seen, and answers with a calibrated p-value
rather than a qualitative "the signal weakened". It is robust to the split year. The LSST forecast
is the first ETNO-specific decision forecast I know of, and pre-registration is unusual and welcome
in this field. Code and data are public.

*Required revisions.*

1. **Discovery versus announcement dates.** The split uses provisional-designation years. Some
   objects designated before 2016 were announced after it (e.g. 2015 TG387 in 2018; 2017 OF201 in
   2025). Redo the test ordered by announcement date, and with the six objects Batygin & Brown
   (2016) actually used as the training set.
2. **Compare directly with Siraj, Chyba & Tremaine (2026).** Same question, different statistic.
   Show where the two agree and why the prequential test adds information.
3. **The selection-model weighting uses the ETNO orientations themselves.** Discuss whether the
   preferred footprint could be absorbing real clustering. (It raises K, so it works against the
   paper's conclusion, which helps.)
4. **LSST selection model.** The footprint/depth model is simple. Either use Sorcha with the
   baseline cadence, or show that the conclusions hold over the ±0.5 mag and footprint variants.
   (Partly done: the forecast barely changes between V_lim = 24.5 and 25.5, and a slow-mover
   distance cut at 100–150 au removes no detectable ETNOs because LSST finds them near perihelion.)
5. **Refresh the orbits** to the latest MPC/JPL data and archive the snapshot (Zenodo DOI).
6. **Verify every reference.** Several are marked unchecked in `refs.bib`.

*Optional.* An alternative H1 (self-gravitating disk, stellar flyby) would broaden the paper but is
not needed for the stated claim.

**Verdict: publishable in PSJ after minor-to-major revision.** The paper now has a distinct
contribution (calibrated out-of-sample test plus a pre-registered, survey-specific decision
forecast), is timely (LSST began its survey in mid-2026), and makes a falsifiable prediction. AJ is
an equally good fit. MNRAS would accept it too, but the AAS journals are where the Planet Nine
debate is published.

---

## Iteration 2: what was done in-session on the iteration-1 report

| item | status |
|---|---|
| 4. LSST depth and distance variants | Done: V_lim 24.0/24.5/25.0 yield 100/233/525; r < 100 au cut leaves the yield unchanged; deep-stack selection (V_lim 25.5) changes D by < 4%. |
| Split-year robustness | Done: 2013, 2014, 2015, 2016 all give p ≤ 0.031 under the trained planet. |
| 3. Selection weighting | Done: under the data-preferred footprint the out-of-sample p-values are 0.002 (2013), 0.007 (2014), 0.056 (2015) and 0.003 (2016). The result does not depend on the selection model. |
| LSST forecast check | Done: end-to-end simulated LSST catalogues give K ≥ 100 in 19%, 62% and 100% of skies at N = 31, 62 and 124, which matches the analytic forecast (median crossing near 50 objects, 90% near 95). |
| N-body forward model | Done (Nov 5). The earlier "not anti-aligned" caveat came mostly from a sign-wrapping bug in the Figure 9 plot, not from the dynamics. The original 1 Gyr runs were within 15–80° of anti-alignment. The fixed model adds explicit Neptune, J2 precession of the planet, and the planet's evolving orbit as the reference. The remaining limits are a single configuration per planet, 600 particles and 2 Gyr instead of 4. **Revision item:** an emulator over (m9, a9, e9, i9) as in Brown & Batygin (2021), so that the N-body H1 can carry a prior and be compared with the analytic posterior. |
| 1, 2, 5, 6 | Need network access or reading that this environment could not do (MPC/JPL/arXiv blocked). These are the author's pre-submission tasks. |

## Bottom line

**Yes, it is worth publishing, as a full PSJ paper rather than only a research note**, provided the
author completes the pre-submission tasks above: announcement-date ordering, the Siraj et al. 2026
comparison, a data refresh with a Zenodo archive, and reference checks. The N-body emulator can be done
during revision. The headline is not the
size of the Bayes factor. It is that the planet hypothesis, as the pre-2016 data defined it, failed
a calibrated out-of-sample test (p ≈ 0.01–0.03), and that LSST will settle the question within
roughly 2–8 years under a decision rule fixed in advance.

The quick RNAAS note on the evidence history is still a sensible first step: it establishes
priority for the prequential result while the full paper is revised.
