# Second-paper triggers

This file is read every week by the Planet Nine data tracker. It lists concrete, checkable events
that would change the first paper's result enough to justify a second paper. Each trigger has an
ID, a threshold, how to check it, and what it unlocks. The numbers come from
`paper/results/*.json` (snapshot: MPC orbits to 2026-01-22; 31 ETNOs with a > 250 au, q > 30 au).

## Baseline the triggers are measured against

| quantity | value at the snapshot | source |
|---|---|---|
| ETNOs with a > 250 au, q > 30 au, arc ≥ 1 yr (bb16 sample) | 31 | `main.json` |
| ETNOs with a > 150 au, q > 30 au (ts14 sample) | 62 | `main.json` |
| Newest bb16 object | 2024 FX26 | `history.json` |
| Bayes factor, orbits + non-detection | K = 1.7 (ln K = 0.52) | `nondet.json` |
| Bayes factor history | peak ln K = 5.9 with the 11 objects known by 2013; 0.86 today | `history.json` |
| Out-of-sample test (train ≤ 2015, test 8 later objects) | ln K_new = −1.87; p = 0.03 under the trained planet | `predictive_2015.json` |
| LSST ETNOs needed for K = 100 if the implied planet is real | 47 (median sky), 95 (90% of skies) | `lsst_forecast.json` |
| Same with today's survey mix | 63 (median), 133 (90%) | `lsst_forecast.json` |
| OSSOS-anchored LSST 10-year yield, a > 250 au | ~230 (68%: 134–409), 114–278 across population models | `lsst_yield.json` |

## Triggers

Status values: `not met`, `watch` (partly met or ambiguous), `met`.

| ID | trigger | threshold | how to check | unlocks |
|---|---|---|---|---|
| T1 | New ETNOs with a > 250 au, q > 30 au in the public catalogue | ≥ 8 new objects beyond the 31 (i.e. N ≥ 39). This doubles the out-of-sample test set | MPC Distant.txt or JPL SBDB query (`a > 250`, `q > 30`, `data_arc ≥ 365`); MPECs; discovery papers on arXiv astro-ph.EP | Rerun of the prequential test with a second independent test set; a short update paper (RNAAS) |
| T2 | LSST-discovered ETNOs with a > 250 au and multi-opposition orbits | ≥ 47 (median decision point); ≥ 95 is decisive at 90% | Rubin data-release notes, Rubin/LSST Solar System Science Collaboration papers, MPC counts of objects with LSST station code X05 | The pre-registered LSST test (second paper, full) |
| T3 | Rubin data release containing Solar System (moving-object) catalogue products | Any public release (DP2, DR1, or an alert-stream SSObject table) | rubinobservatory.org news, dp*.lsst.io / dr*.lsst.io, community.lsst.org announcements | Replace the model LSST selection function with the real survey cadence |
| T4 | A public LSST survey simulator configured for the actual survey (pointing history plus detection efficiency), e.g. Sorcha with released OpSim/visit history | Release of the visit database or a characterised-survey paper | GitHub (dirac-institute/sorcha, lsst/rubin_sim), arXiv | Characterised selection for LSST ETNOs; removes the main systematic |
| T5 | Characterisation of the Sheppard–Trujillo or DES survey for ETNOs (pointings + efficiencies) made public | Any public release | arXiv astro-ph.EP, journal data supplements, Zenodo | Characterised selection for about two thirds of today's sample |
| T6 | A single new ETNO that moves the evidence a lot | Any new object with a > 400 au, or with q > 60 au, or with ϖ within 30° of 247° (the inferred planet direction) or of 67° (the cluster centre) | Discovery announcements; compute its contribution with `scripts/forecast_and_history.py` | Update note if |Δ ln K| > 1 |
| T7 | A direct detection claim, or a search that excludes much of the remaining planet parameter space | Detection claim, or a published search excluding ≥ 50% of the Brown & Batygin (2021) posterior not already excluded | arXiv, AAS/NASA news, Brown/Batygin group papers, Rubin announcements | Rewrite of the non-detection term or, for a detection, a new paper |
| T8 | A competing analysis on a larger sample | A paper reporting a clustering significance or Bayes factor on ≥ 35 ETNOs with a > 250 au, or any LSST-based clustering result | arXiv listings for "Planet Nine", "extreme trans-Neptunian", "apsidal clustering" | Response or joint analysis |

## Decision rule for starting the second paper

Start the second paper when **T2 is met**, or when **T1 and one of T3, T4, T5 are met**.
Write a short update (RNAAS) when T1, T6 or T8 alone is met.
A T7 detection claim overrides everything: report it immediately.

## Tracker log

The tracker appends one line per run, newest last:
`YYYY-MM-DD | N_bb16 (best estimate) | triggers met | triggers on watch | sources checked | note`.

<!-- tracker log starts -->
2026-10-30 | 31 | none | none | baseline written | first-paper snapshot
2026-11-02 | 31 (no new ETNO verified) | none | T1, T3, T6 | WebSearch summaries (Rubin news, dp2.lsst.io, community.lsst.org, arXiv via search); arxiv.org and minorplanetcenter.net blocked | Rubin reports two TNOs reaching ~1,000 au (orbits unverified); DP2 SSObject tables exist but public release unconfirmed
2026-11-05 | 32 (MPC: +2015 DR370; 2025 LS2 at 350 d counts as 33 under year-span arc rounding; JPL still 31) | T3 (DP2 has SSObject/SSSource), T6 threshold only (2015 DR370 a = 416 au, Δ ln K = −0.40, below the 1.0 action level) | T1 (16 Rubin-discovered a > 250 au objects awaiting 1-yr arcs), T4 (DP2 Visit table, no efficiencies) | MPC Distant.txt, JPL SBDB, MPC get-obs/get-orb, MPECs 2026-U70–V92, arXiv, dp2.lsst.io, rubin.community | details in paper/tracker/2026-11-05.md; decision rule not met
