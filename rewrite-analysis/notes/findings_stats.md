# Statistics-core findings (for the paper)

All numbers come from `python scripts/stats_report.py` (seed 20261030; 100,000 Monte Carlo draws, 299
bootstrap/null replicates) on the canonical MPC snapshot samples from `planetix.data`:
**ts14** (a > 150, q > 30 AU, N = 62), **bb16** (a > 250, q > 30, N = 31) and **detached**
(a > 250, q > 40, N = 19). Raw output: `results/stats_report.json`.

Every null below is **uniform/isotropic** unless stated otherwise. It does not model survey selection, so a
small p-value means "not uniform", not "shepherded by a planet". The bias-aware null lives in `planetix.bias`.

## 1. Clustering of angles against a uniform null

| Sample | N | angle | R-bar | Rayleigh p | Kuiper p | mean (95% CI) |
|---|---|---|---|---|---|---|
| ts14 | 62 | varpi | 0.265 | 0.012 | 0.044 | 43° (5° to 86°) |
| ts14 | 62 | Omega | 0.324 | **0.0013** | **0.0003** | 138° (104° to 166°) |
| ts14 | 62 | omega | 0.273 | 0.009 | 0.011 | 297° (257° to 337°) |
| bb16 | 31 | varpi | 0.234 | 0.19 | 0.42 | 56° (354° to 143°) |
| bb16 | 31 | Omega | 0.269 | 0.11 | 0.20 | 121° (49° to 180°) |
| bb16 | 31 | omega | 0.184 | 0.35 | 0.48 | 297° (194° to 34°) |
| detached | 19 | varpi | 0.230 | 0.37 | 0.67 | 33° (272° to 146°) |
| detached | 19 | Omega | 0.256 | 0.29 | 0.46 | 112° (11° to 199°) |
| detached | 19 | omega | 0.285 | 0.22 | 0.40 | 341° (261° to 57°) |

The analytic Rayleigh p agrees with 100k-draw Monte Carlo to within 0.001 in every row. The 3D test on
perihelion direction vectors gives p = 0.0036 (ts14), 0.16 (bb16) and 0.41 (detached).

* The angles cluster only in the broad ts14 sample, which includes many 150–250 AU objects that Neptune
  still affects. In the a > 250 AU samples that motivated Planet Nine, no angle is distinguishable from
  uniform, even before survey bias is accounted for.
* **Cut-choice correction.** Scanning 15 (a_min, q_min) cuts (a ∈ {150, 200, 230, 250, 300},
  q ∈ {30, 35, 40}) on varpi, the best cut is a > 150, q > 30 with local p = 0.012. Putting the same search
  into the null gives a **global p = 0.064**. Picking the most favourable cut is therefore worth about a
  factor of 5 in p.
* **Power.** The fitted concentrations are kappa ≈ 0.4–0.7. A Rayleigh test has 80% power only for
  kappa ≥ 0.60 (N = 62), 0.85 (N = 31) and 1.1 (N = 19). The a > 250 AU samples are too small to rule
  out a cluster as weak as the one ts14 shows. This is a non-detection, not evidence for uniformity.
* varpi and Omega are not significantly correlated (Jammalamadaka r = 0.25, permutation p = 0.053 for ts14;
  p = 0.50 for bb16). The bands that old script 08 drew by eye are not needed.

## 2. The "Planet 9 probability" classifiers (old scripts 09, 12, 14)

Those scripts averaged a random forest's output after training it on a 50/50 mix of a simulated P9 world
and a uniform world. The result depends on that arbitrary prior, so it is not a probability. The worlds they
simulated are fully specified, so the direct test is a likelihood ratio, which here equals the Bayes factor
against uniform:

| Sample | unimodal world (script 09) log10 BF | bimodal world (scripts 12/14) log10 BF |
|---|---|---|
| ts14 | **−8.5** | −1.5 |
| bb16 | **−4.8** | −0.6 |
| detached | −3.4 | +0.3 |

The unimodal world (all objects, kappa = 2 at 60°) is decisively disfavoured relative to plain uniform
angles. The real clustering is far weaker than that model assumed. The bimodal world is at best "barely
worth mentioning" (|log10 BF| < 0.5) on the 19-object sample.

Fitting von Mises mixtures, which include a uniform background, gives these BIC values: ts14 227.9 / 230.9 / 240.4
for 0 / 1 / 2 clusters, and bb16 113.9 / 117.5 / 126.4. Uniform wins on BIC in every sample. Parametric-bootstrap
LRTs give one cluster vs none p = 0.057 (ts14), 0.16 (bb16) and 0.50 (detached), and two clusters vs one
p = 0.60, 0.78 and 0.34. **There is no evidence for a second ("aligned") cluster.**

## 3. PCA and clustering (old scripts 21, 28, 31)

* **Parallel-analysis PCA:** in every sample only PC1 beats the permutation null (p = 0.003). Its loadings
  are log10 a (+0.60) and e (+0.59), with inclination at only −0.23 (ts14). This is the a–e correlation that
  the perihelion cut q = a(1−e) > q_min itself imposes. It is a selection artefact, not a "warp". The old
  README claimed PC1 is "driven by inclination", and that is not reproduced.
* **The silhouette > 0.5 rule of script 31.** On its own 39-object `smart_dataset.csv` the best silhouette
  is 0.553 at k = 2. The two "clusters" are 2 objects (the retrograde 2019 EE6 and 2020 MQ53 at
  a = 17,456 AU) against the other 37. Running the same k-search on column-permuted data gives a
  95th-percentile silhouette of 0.544, so p = 0.05 even for that outlier split. With log a it drops to 0.23
  (p = 0.24). On the canonical samples the calibrated p is 0.007 (ts14, again the a–e axis), 0.13 (bb16)
  and 0.41 (detached).
* **"BIC finds N planes" (scripts 21, 28).** A Gaussian mixture on (i, sin Ω, cos Ω) chooses 5 components
  for every canonical sample. On *uniform* random nodes with N = 31 it also chooses 5 components in 37 of
  50 trials, because (sin, cos) points lie on a circle that a single Gaussian cannot fit. The detected
  "planes" are an artefact of the method. The silhouette test, calibrated by permutation, gives p = 0.39,
  0.19 and 0.36 in that space.

## 4. Fixes to the old statistics, in one place

* KS test on varpi/360 (script 04): its result depends on where 0° is placed. It is replaced by Kuiper's V.
* Rayleigh p (script 05): the old series approximation is replaced by Greenwood–Durand. MC p-values use
  (1+k)/(1+M), so they are never 0.
* kappa: Best–Fisher small-sample correction for N < 16.
* Bootstrap "confidence" that a GMM mean lands in a hand-picked 12–25° window (script 28) is not a
  significance and is dropped.
* Mixture-model LRTs use a parametric bootstrap, because Wilks' theorem fails at the kappa = 0 boundary.
* Bug in the first draft of the rebuild, caught and fixed: `GaussianMixture.sample` reuses its fixed seed,
  so every bootstrap replicate was identical. There is now a test for it.
