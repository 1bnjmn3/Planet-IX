# Dynamics findings (for the paper)

Code: `src/planetix/dynamics/`. Numbers below come from `scripts/validate_secular.py`,
`scripts/nbody_forced_poles.py` and `scripts/invert_planet.py`, whose outputs are in `data/processed/`.
Samples are the canonical cuts of `data/README.md` (`ts14`: 62 objects, `bb16`: 31). Angles are measured
from the invariable plane unless marked "ecl".

## 1. The original "secular" model had no physical units

Scripts 23, 25, 33, 35 and 36 balance a giant-planet "torque" `1e8 / a^3.5` against a planet "torque"
`1e10 m a^2 / a_p^3`. The two constants were chosen by hand, and the planet term has the wrong power
of `a`: as a precession rate it is `(3/4) n m (a/a_p)^3 ∝ a^1.5`, not `a^2` (the giant term's `a^-3.5`
is right). The rebuild uses Laplace-Lagrange coupling rates
with real masses (Murray & Dermott 1999, sec. 7.4), checked against the quadrupole limits and the J2
precession in tests.

Effect: the toy model makes a 5 Earth-mass planet at 500 AU 185x stronger than the giants at a = 250 AU;
the real ratio is 1.7. At 400 AU the toy says 2,460x, the real value is 75.

| claim | original script | original number | real units |
|---|---|---|---|
| forced inclination at a = 150 AU, planet 5 Me / 500 AU / i = 20° | 23 | 19.5° | 1.6° (circular), 0.7° (e = 0.8) |
| forced inclination at a = 250 AU, same planet | 23 | 20.0° | 12.8° (circular), 8.4° (e = 0.8) |
| "Planet Y" 1 Me / 150 AU / i = 10°, forced i at 400 AU | 23 | 10.0° | 4.4° |
| distance at which 1 Me at i = 65° tilts a = 400 AU orbits by 15.7° | 33 | 5,778 AU | 912 AU (e = 0), 693 AU (e = 0.85) |
| same for 5 Me | 33 | 9,881 AU | 1,434 AU / 1,019 AU |
| 5 Me at i = 20° to give 15.7° at 400 AU | 36 | 4,382 AU ("LOW plausibility") | 850 AU / 656 AU |
| inner "Planet Y", 1 Me at i = 65°, 15.7° at 400 AU | 34-36 | 110 AU | 128 AU |

Scripts 34-36's inner-perturber branch (giants' sum of m a^2 = 38,023 Me AU^2, which is correct) has the
right structure for an inner planet and is off mainly through the linear averaging at 65°. The outer-planet
branch is off by factors of 5-10 in distance, i.e. more than 100 in mass.

Caveat for the table: 400 AU orbits with e ≈ 0.85 reach the planet's orbit for most of these planets,
where secular theory itself fails (section 2). The real-unit numbers are what the secular model says,
not a prediction.

## 2. Where secular theory holds: REBOUND check

`nbody.py` integrates test particles with IAS15 under the Sun, the giants replaced by the J2 they induce
(REBOUNDx), and the planet. The time-averaged pole of each particle over four free precession cycles is its
forced pole. Planet 5 Me, 500 AU, i = 20°; masses and J2 boosted 10x to shorten the cycles
(`data/processed/secular_vs_nbody.csv`):

| a (AU) | e | Q (AU) | N-body i | LL circular | quadrupole with e |
|---|---|---|---|---|---|
| 150 | 0.2 | 180 | 1.67° | 1.64° | 1.57° |
| 200 | 0.3 | 260 | 6.22° | 6.15° | 6.07° |
| 250 | 0.3 | 325 | 12.68° | 12.54° | 12.73° |
| 300 | 0.2 | 360 | 15.55° | 16.99° | 17.23° |
| 200 | 0.7 | 340 | 5.40° | 6.15° | 4.32° |
| 250 | 0.7 | 425 | 10.25° | 12.54° | 10.49° |
| 300 | 0.85 | 555 | 6.60° | 16.99° | 12.94° |

Also: the giants-only nodal precession rate matches N-body to 3% at a = 200 AU, e = 0.7, once the
(1-e^2)^-2 eccentricity factor is included (circular theory is 3.7x too slow there).

So the secular model is good to ~0.2° for orbits that stay inside ~0.65 a_p, needs the eccentricity
factors for ETNO-like e, and fails by a factor of two once the orbit reaches the planet (last row). The
inversion therefore uses, for each trial a_p, only objects with Q < 0.8 a_p or q > 1.25 a_p.

**Consequence for the canonical sample:** for any planet with a_p ≤ 600 AU, *none* of the 31 bb16 objects
satisfy that condition (median Q = 702 AU). A secular (Laplace-plane) inversion of bb16 is not possible
for the Planet Nine parameters usually quoted; it needs N-body forcing, which is section 4.

Original script 19 integrated 100 random particles for 500 kyr. The secular period at these distances
is 40-1,000 Myr, so its run covered under 1% of one cycle; its "captured" count measured the initial
conditions, not P9's influence.

## 3. What the ETNO poles show, using every object

Script 25 kept only objects with 14° < i < 22° and then fitted their inclination. Rerun on the bundled
snapshot with its own cuts it reproduces the original result exactly (8 objects, planet i = 17.11°,
1.654 Me). On 300 mock skies with the same (a, e), inclinations redrawn from the full catalogue and no
plane at all, it returns planet i = 17.7° (5-95%: 16.3-19.4°), and the mass hits one of its fit bounds
(0.1 or 20 Me) in 67% of mocks. The inclination is set by the selection window and the mass is
unconstrained, so the 1.65 Me "Planet Y" is not a measurement.

The rebuild fits every object in the sample, never selecting on the angles being fitted, with a von
Mises-Fisher scatter of poles about their forced pole. Results (`data/processed/inversion/`):

**Mean orbit pole.** ts14 (62 objects): tilted 5.5° from the invariable pole toward node 33°
(ecl: i = 6.9°, node 134°), bootstrap 68% range 3.7-7.7°. Against mocks with no tilt and the same
scatter (κ = 16, i.e. ~14° rms), p = 0.010 (bootstrap) and p = 0.02 (likelihood ratio). bb16 alone
(31 objects): 5.2°, p = 0.13 / 0.07. Per a-bin the tilt is 6.1° (150-250 AU), 7.5° (250-400),
13.0° (400-700, 6 objects), 7.7° toward a different node (>700, 7 objects); no single bin is
significant (p = 0.08-0.20).

**Planet fit.** A planet (mass, a_p, i_p, node free) beats no planet with p = 0.04 (ts14) and 0.03 (bb16)
after calibrating against 100 no-planet mocks, which include the freedom to choose a_p. But a model with
just one common tilted plane (2 parameters, no a-dependence) does as well: on the ts14 subsets used at
a_p = 800-1500 AU the common plane scores higher than the best planet. The data detect, at about 2σ, a
mean plane tilted ~5-7° from the invariable plane; they do not detect a dependence on a that would
point to a specific planet.

**Mass limits.** Mass and inclination are degenerate: a heavier planet at lower inclination gives the same
tilt. With i_p free, no mass up to 50 Me is excluded at any a_p. If the tilt is taken to be caused by a planet,
the lower limits are about 5 Me (a_p = 500 and 1000 AU) and 7 Me (600-800 AU) from ts14. With i_p fixed at 30°, the 95% upper limits
are 17-36 Me for a_p = 400-1000 AU; at i_p = 20° only a_p = 400 AU (35 Me) is bounded. The
`upper_limits.csv` file has the full table.

## 4. Real-mass N-body forcing for ETNO-like orbits

PENDING: `scripts/nbody_forced_poles.py` (5 Me / 500 AU and 10 Me / 700 AU, e_p = 0.25, i_p = 20°,
q = 45 AU, 1 Gyr, no boost).

## 5. What is not modelled

- Survey selection. Surveys near the ecliptic favour some (i, node) combinations and could produce or hide
  a few-degree tilt; the bias workstream should test the ~5.5° tilt against its simulator before it is
  called a signal.
- The giants' plane is held fixed; in reality a distant planet slowly torques it as well.
- Mean-motion resonances and apsidal (varpi) confinement. These are non-secular and are not predicted
  by this model.
- Orbit-element uncertainties (small for the multi-opposition orbits kept by the quality cuts).
