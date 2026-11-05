# Literature review: statistical evidence for Planet Nine from ETNO clustering
Compiled 2026-10-30.

## 0. Verification status (read first)
- Direct page fetches of arxiv.org, export.arxiv.org, ar5iv, survey-strategy.lsst.io, crossref were BLOCKED (host not allowed in this session; proxy returned 403/ENOTFOUND). Only github.com (one issue page) could be fetched.
- Everything marked **[V]** below was seen in a WebSearch result (title/arXiv id/DOI/journal reference visible in the result or its summary). Search results are summaries, not the primary pages, so even [V] entries should get a final check against ADS before submission.
- **[M]** / UNVERIFIED = written from memory, NOT checked against any page this session. Do not cite without checking.
- Per-object announcement dates (section 5b) could NOT be verified; the table is mostly blank by design.

## 1. BibTeX

### 1a. Seen in search results [V] (existence and identifiers; field-level details partly from result summaries)
```bibtex
@article{Batygin2016,
  author  = {Batygin, Konstantin and Brown, Michael E.},
  title   = {Evidence for a Distant Giant Planet in the Solar System},
  journal = {The Astronomical Journal}, year = {2016}, volume = {151}, pages = {22},
  eprint  = {1601.05438}, archivePrefix = {arXiv},
  note    = {[V] title, arXiv id seen; AJ 151, 22 as given in the task; DOI 10.3847/0004-6256/151/2/22 UNVERIFIED}
}
@article{Brown2016,
  author  = {Brown, Michael E. and Batygin, Konstantin},
  title   = {Observational constraints on the orbit and location of Planet Nine in the outer solar system},
  journal = {The Astrophysical Journal Letters}, year = {2016}, volume = {824}, pages = {L23},
  eprint  = {1603.05712}, archivePrefix = {arXiv},
  note    = {[V] title, arXiv id seen; volume/page/DOI UNVERIFIED (memory)}
}
@article{Brown2017,
  author  = {Brown, Michael E.},
  title   = {Observational bias and the clustering of distant eccentric Kuiper belt objects},
  journal = {The Astronomical Journal}, year = {2017}, volume = {154}, pages = {65},
  eprint  = {1706.04175}, archivePrefix = {arXiv},
  note    = {[V] title, arXiv id seen; volume/page UNVERIFIED (memory)}
}
@article{Brown2021,
  author  = {Brown, Michael E. and Batygin, Konstantin},
  title   = {The Orbit of Planet Nine},
  journal = {The Astronomical Journal}, year = {2021}, volume = {162}, pages = {219},
  doi     = {10.3847/1538-3881/ac2056}, eprint = {2108.09868}, archivePrefix = {arXiv},
  note    = {[V] AJ 162, 219; DOI suffix ac2056 seen in IOP URL; abstract numbers seen}
}
@article{Napier2021,
  author  = {Napier, Kevin J. and Gerdes, David W. and others},
  title   = {No Evidence for Orbital Clustering in the Extreme Trans-Neptunian Objects},
  journal = {The Planetary Science Journal}, year = {2021}, volume = {2}, pages = {59},
  doi     = {10.3847/PSJ/abe53e}, eprint = {2102.05601}, archivePrefix = {arXiv},
  note    = {[V] PSJ 2, 59, DOI, arXiv id seen; FULL AUTHOR LIST UNVERIFIED (search said lists differ across sources)}
}
@article{Brown2024,
  author  = {Brown, Michael E. and Holman, Matthew J. and Batygin, Konstantin},
  title   = {A Pan-STARRS1 Search for Planet Nine},
  journal = {The Astronomical Journal}, year = {2024},
  note    = {[V] title seen (ResearchGate id 378809174); authors, volume, page, DOI, arXiv id UNVERIFIED}
}
@article{Batygin2024,
  title   = {Generation of Low-Inclination, Neptune-Crossing TNOs by Planet Nine},
  eprint  = {2404.11594}, archivePrefix = {arXiv}, year = {2024},
  note    = {[V] title, arXiv id seen (draft April 18 2024, AASTeX); authors/journal UNVERIFIED (presumably Batygin, Brown et al.)}
}
@article{Cheng2025,
  author  = {Cheng, Sihao and Li, Jiaxuan and Yang, Eritas},
  title   = {Discovery of a dwarf planet candidate in an extremely wide orbit: 2017 OF201},
  year    = {2025}, eprint = {2505.15806}, archivePrefix = {arXiv},
  note    = {[V] authors, title, id seen (search summary); journal status UNVERIFIED}
}
@article{Siraj2025,
  author  = {Siraj, Amir and Chyba, Christopher F. and Tremaine, Scott},
  title   = {Measuring the Mean Plane of the Distant Kuiper Belt},
  journal = {Monthly Notices of the Royal Astronomical Society: Letters}, year = {2025},
  doi     = {10.1093/mnrasl/slaf091}, eprint = {2508.14156}, archivePrefix = {arXiv},
  note    = {[V] arXiv id, authors, DOI per secondary source (Watchers); volume/page UNVERIFIED. Popularly "Planet Y". Search also showed the title "The Curiously Warped Mean Plane of the Kuiper Belt" (ouci.dntb.gov.ua), possibly a retitled version.}
}
@article{Siraj2026,
  author  = {Siraj, Amir and Chyba, Christopher F. and Tremaine, Scott},
  title   = {Measuring Apsidal Clustering},
  journal = {The Astrophysical Journal}, year = {2026}, volume = {1003},
  eprint  = {2604.25990}, archivePrefix = {arXiv},
  note    = {[V] arXiv id, authors, title, "accepted/published ApJ vol. 1003" from search summaries; page/DOI UNVERIFIED}
}
@article{Kurlander2025,
  author  = {Kurlander, Jacob A. and others},
  title   = {Predictions of the LSST Solar System Yield: Near-Earth Objects, Main Belt Asteroids, Jupiter Trojans, and Trans-Neptunian Objects},
  journal = {The Astronomical Journal}, year = {2025}, volume = {170}, pages = {99},
  doi     = {10.3847/1538-3881/add685}, eprint = {2506.02487}, archivePrefix = {arXiv},
  note    = {[V] DOI, AJ 170 99, ~37,000 TNO headline (search summaries). arXiv id 2506.02487 appeared in results for this query but was not confirmed to be this paper. Author list UNVERIFIED.}
}
@article{Merritt2025,
  author  = {Merritt, Samuel and others},
  title   = {Sorcha: a solar system survey simulator},
  year    = {2025}, eprint = {2506.02804}, archivePrefix = {arXiv},
  note    = {[V] arXiv id and first author per search summary; exact title/journal UNVERIFIED}
}
@article{Murtagh2026,
  author  = {Murtagh, and Schwamb, Megan E. and others},
  title   = {Predictions of the LSST Solar System Yield: Neptune Trojans},
  journal = {The Astronomical Journal}, year = {2026}, volume = {171}, pages = {90},
  eprint  = {2512.03892}, archivePrefix = {arXiv},
  note    = {[V] title, id, AJ 171 90 per summary; full authors UNVERIFIED}
}
@article{Schwamb2023,
  author  = {Schwamb, Megan E. and others},
  title   = {Tuning the Legacy Survey of Space and Time (LSST) Observing Strategy for Solar System Science},
  year    = {2023}, eprint = {2303.02355}, archivePrefix = {arXiv},
  note    = {[V] title, id seen. Journal (ApJS 266, 22?) and authorship UNVERIFIED. Defines ETNO as q>42, a>150 au.}
}
@article{Pfalzner2026,
  author = {Pfalzner, S.},
  title  = {Trans-Neptunian Object dynamics even better explained by a stellar flyby after 4.5 Gyr of evolution},
  year   = {2026}, eprint = {2609.03575}, archivePrefix = {arXiv},
  note   = {[V] via GitHub paper-watch issue; initials/journal UNVERIFIED}
}
@article{Bansal2026,
  author = {Bansal},
  title  = {Distant TNO Inclinations as a Constraint on Primordial Cluster Perturbations in the Presence of Planet Nine},
  year   = {2026}, eprint = {2607.15646}, archivePrefix = {arXiv},
  note   = {[V] via GitHub paper-watch issue only; full authors UNVERIFIED}
}
@article{Bougakov2026,
  author = {Bougakov},
  title  = {Galactic tides in the Solar System within a non-axisymmetric Milky Way model adjusted to Gaia data},
  year   = {2026}, eprint = {2609.02693}, archivePrefix = {arXiv},
  note   = {[V] via GitHub paper-watch issue only}
}
```

### 1b. Requested references NOT verified this session (UNVERIFIED; keys and fields from memory only; check on ADS)
Trujillo & Sheppard 2014 (Nature 507, 471); Batygin et al. 2019 (Phys. Rep. 805, 1); Brown & Batygin 2019 (AJ 157, 62); Brown & Batygin 2022 (Pan-STARRS search, if distinct from Brown+2024); Shankman et al. 2017 (AJ 154, 50); Bannister et al. 2018 (ApJS 236, 18); Kavelaars et al. 2009 / Petit et al. 2011 (survey simulator); Bernardinelli et al. 2020 (ApJS 247, 32) and 2022 (ApJS 258, 41); Sheppard & Trujillo 2016 (AJ 152, 221); Sheppard et al. 2019 (AJ 157, 139); Kaib et al. 2019; Madigan & McCourt 2016; Sefilian & Touma 2019; Zderic & Madigan 2020; Clement & Kaib 2020; Nesvorny et al. 2017/2023; Batygin & Morbidelli; Chen & Kipping 2017; Ivezic et al. 2019 (ApJ 873, 111); Lykawka & Ito 2023; Speagle 2020 (MNRAS 493, 3132); Skilling 2006; Rein & Liu 2012; Rein & Tamayo 2015; Tamayo et al. 2020; Kass & Raftery 1995 (JASA 90, 773); Dawid 1984 (JRSS A 147, 278); Jeffreys 1961. None of these was retrieved or seen in a search result.

Other [V]-level items seen in results that may be worth citing (title and arXiv id only; authors UNVERIFIED): 1805.05355 (discovery of a high-inclination ETNO; presumably Becker et al. 2018), 2109.13307 (distant planet and detached-belt inclinations), 1605.02473 (Beust 2016 A&A, secular vs resonant clustering), 1603.06520 (Finding Planet Nine: a Monte Carlo approach), 2009.07653 (LSST impact on Solar System science), 2506.12854 (AKARI far-IR P9 search), 2504.05473 (parallax-based P9 search), 2110.13117 (ZTF P9 search), 2005.05326 (orbital precession constraints on P9), 2004.01198 (Apsidal clustering following the inclination instability; ApJL, DOI 10.3847/2041-8213/ab91a0), 2106.09739 (A Lopsided Outer Solar System).

## 2. Annotated list of 2023-2026 papers

- **Siraj, Chyba & Tremaine 2026, "Measuring Apsidal Clustering" (arXiv:2604.25990, ApJ) [V].** Introduces a conditional-likelihood method for apsidal clustering that is insensitive to uneven survey footprints. Expands the sample of long-term-stable distant TNOs from 21 to 25 and reports that the significance fell from 2.7 sigma to 1.9 sigma, with the clustering direction poorly constrained (~half the sky). Per the results section, the intermediate sample (SCT25) strengthened the evidence relative to the BB21 sample, while the latest additions (SCT26) weakened it below the BB21 level. **Overlap: this is the closest competitor. It covers our contribution (2) in spirit (evidence weakens as discoveries accumulate) and partly (1) (a likelihood, not a Bayes factor; p/sigma framing, not planet vs isotropy).** Needs a full-text read: which sample each sigma refers to, whether they model the planet hypothesis explicitly, and whether they discuss Rubin.
- **Siraj, Chyba & Tremaine 2025 (arXiv:2508.14156, MNRAS Letters) [V].** Measures the mean plane of the non-resonant Kuiper belt at 50-400 au with a bias-independent method; finds a warp at ~80-200 au attributed to a hypothetical "Planet Y" (25-450 Pluto masses per EarthSky). Separate line of evidence from ETNO apsidal clustering; no Bayes-factor or evidence-history element.
- **Cheng, Li & Yang 2025, 2017 OF201 (arXiv:2505.15806) [V].** a ~ 830 au, q ~ 44.9 au, ~700 km, found in DECaLS/CFHT archive data (19 observations 2011-2018). Outside the clustered region (varpi ~ 60 deg); in their simulations P9 ejects it in ~60% of realizations, but they say this does not robustly exclude P9. It is one of the 31 objects in our sample; its position is a direct data point for the evidence history (a late, unclustered addition).
- **Kurlander et al. 2025 (AJ 170, 99) [V] and Murtagh/Schwamb et al. 2026 (AJ 171, 90) [V], Sorcha-based LSST yield predictions.** Predict ~37,000 total TNOs after LSST (about 4-10x current census). The summaries I saw give no split for detached or ETNO objects; the Neptune Trojan paper forecasts ~130-300 Trojans. They provide the survey-simulator framework (Sorcha) on which an ETNO-specific forecast could be built. Overlap with (3): infrastructure only; no ETNO count seen.
- **Schwamb et al. 2023, observing strategy tuning (arXiv:2303.02355) [V].** Describes the Rubin "NES" mini-survey as key to testing ETNO clustering (ETNO defined q > 42, a > 150 au); notes that standard Rubin pipelines are sensitive only to ~100-150 au, so very distant objects need community-built methods. Partly relevant to (3): motivates the survey but gives no count.
- **Batygin et al. 2024, Neptune-crossing TNOs from P9 (arXiv:2404.11594) [V].** Predicts a ratio of Neptune-crossing objects with a = 100-1000 au to those with q > 30 au of ~3% with P9 versus ~0.5% without. A non-clustering test; no overlap with (1)-(3), but relevant context.
- **Pfalzner 2026 (arXiv:2609.03575), Bansal 2026 (arXiv:2607.15646), Bougakov 2026 (arXiv:2609.02693) [V, via a GitHub paper-watch issue only].** Alternatives or constraints (stellar flyby, primordial cluster plus P9 inclinations, galactic tides). Dynamics papers; no statistical-evidence overlap seen.
- **Brown, Holman & Batygin 2024, Pan-STARRS1 search [V title only].** Observational exclusion of part of the P9 parameter space; not a clustering-statistics paper.

No 2024-2026 paper found that reports a Bayes factor for planet vs isotropy on ETNOs. The searches were limited by the arXiv block; an ADS/arXiv listing check is still needed.

### Novelty assessment
Of our three contributions, the evidence-history idea is partly published and the other two appear new, with the caveat that I could not read any full text and my searches were summary-only. (1) A bias-aware Bayes factor for planet versus isotropy on the 31 known a > 250 AU ETNOs: I found no published Bayes factor; the literature uses frequentist significance (Brown & Batygin 2019), joint-probability tests against a uniform null (Napier et al. 2021, which found no evidence), and, most recently, a conditional-likelihood significance (Siraj, Chyba & Tremaine 2026, 2.7 sigma falling to 1.9 sigma), so this looks new in form but overlaps in aim with Siraj et al. (2) The evidence history: Siraj et al. 2026 already show, in sigma, that the clustering significance rose and then fell as the sample grew from 21 to 25, and Napier et al. 2021 already argued the signal weakens with better bias treatment, so the qualitative finding is published; what appears new is the Bayes-factor trajectory (~17 pre-2016 to ~2.5 now) and its framing as an out-of-sample, prequential test of Batygin & Brown 2016, which I found nowhere. (3) The Rubin/LSST forecast of when ~120 a > 250 AU ETNOs will settle the question: I found no published ETNO-specific yield or decision-time forecast (only a ~37,000 total-TNO projection and qualitative statements that LSST will find many more than the <20 known), so this appears new, but the Kurlander et al. 2025 full text should be read for a detached/ETNO breakdown before claiming priority.

## 3. Published quantitative forecasts of Rubin/LSST ETNO yields

| Source | Population | Number | Status |
|---|---|---|---|
| Kurlander et al. 2025, AJ 170, 99 (Sorcha) | all TNOs | ~37,000 total (4-10x current census) | [V] headline only; no ETNO/detached split seen |
| Murtagh/Schwamb et al. 2026, AJ 171, 90 | Neptune Trojans | median ~130-300 | [V]; not ETNO |
| Rubin solar-system white paper (arXiv:2009.07653) | q > 40, a > 150-250 au | "many more than the <20 currently known"; no number | [V] qualitative only |
| Popular articles (Inverse etc.) | P9 itself | "60%" / "70-80%" chance to detect P9 | UNVERIFIED, no primary paper found; not an ETNO count |
| Trilling et al. 2018 (via white paper) | P9 | LSST rules out P9 in 61% of sky if not detected | [V] second-hand; not an ETNO count |

Conclusion: no published numerical forecast of Rubin a > 250 AU (or a > 150 AU) ETNO counts was found. Absence is based on summary-level searches only.

## 4. Rubin/LSST status news (2025-2026)

- First Look images released 23 June 2025; 1,514 asteroids found in April-May 2025 First Look data (behindtheblack.com, secondary).
- Early engineering data yielded >11,000 new asteroids including 33 NEOs (GeekWire, iflscience, starlust, NOIRLab coverage dated ~April 2026; secondary). Same coverage reports ~380 TNOs from early data and two extremely elongated icy bodies, 2025 LS2 and 2025 MX348 (secondary summary; orbits not checked). No distant (a > 250 au) ETNO announcement from Rubin was found.
- LSST 10-year survey start: Rubin's survey-strategy "Survey Progress" page (survey-strategy.lsst.io/progress) is reported by the search summary as saying it began 29 June 2026; other outlets say 30 June, with first image 1 July 2026 (Universe Magazine), and B612's "Asteroid Day 2026: Rubin Officially Begins Operations" (30 June 2026 Asteroid Day). The date differs by a day across secondary sources; check the primary page. Earlier plans targeted an earlier 2026 start (delay noted by one outlet).
- Data Preview 1 (DP1): described in an AJ 2026 paper (per Lancaster eprints listing); release date not verified.
- DP2: draft RTN-115 documents say Early DP2 targeted 2026-07-27 and full DP2 Oct-Dec 2026; it covers commissioning-era LSSTCam data. Whether it shipped on schedule was not verified.
- DR1: older community post said start of survey year 2 (late 2026/early 2027); superseded by the delayed start, UNVERIFIED.

## 5. Training sets and announcement dates

### 5a. Batygin & Brown 2016 (AJ 151, 22) and Brown & Batygin 2021 (AJ 162, 219)
- BB16 object list: the sample is the objects with a > 250 au and q > 30 au (six objects). From memory, UNVERIFIED: Sedna, 2012 VP113, 2004 VN112, 2007 TG422, 2013 RF98, 2010 GB174. The search only confirmed that Sedna and 2012 VP113 were central and that the paper uses q > Neptune-orbit and a > 150 AU for the "perihelion clustering" statement.
- BB16 planet parameters [V via A&A review citing B16 and a Brown & Batygin follow-up summary]: m ~ 10 M_earth, a = 700 au, e = 0.6, i ~ 30 deg (notional; one source quotes e = 0.7); perihelion roughly anti-aligned (~180 deg) to the KBO clustering. KBO clustering values quoted in the A&A review: omega = 318 +/- 8 deg, Omega = 113 +/- 13 deg. Planet varpi9 and Omega9 numerical values UNVERIFIED.
- BB21 [V from abstract and secondary]: sample is 11 objects with 150 < a < 1000 au, q > 42 au (names not seen; UNVERIFIED). Posterior: m9 = 6.2 (+2.2/-1.3) M_earth, a9 = 380 (+140/-80) au, q9 = 300 (+85/-60) au, i9 = 16 +/- 5 deg. A 2025 secondary paper quotes best-fit m = 6.2, a = 382.4 au, e = 0.20, i = 15.6 deg, varpi = 246.7 deg [V second-hand]. omega9 and Omega9 posteriors, and e9 posterior interval: NOT verified.
- Note: BB21's selection (a > 150, q > 42) differs from our a > 250 / q > 30 sample.

### 5b. Announcement dates (31 objects)
I could not verify announcement/MPEC dates for any object. The only partial confirmations are 2017 OF201 (arXiv preprint May 2025, found in DECaLS/CFHT archival data) and a Buie astrometry page for 2024 FX26 (observations in 2024; possible 2021 precovery; not confirmed as a distant TNO). Discovering-survey column filled only from memory and marked as such; blank = unknown.

| object | announced | source | discovering survey |
|---|---|---|---|
| Sedna | | | (memory, unverified) Palomar, Brown/Trujillo/Rabinowitz |
| 2012 VP113 | | (Trujillo & Sheppard 2014, Nature; memory) | (memory, unverified) Sheppard-Trujillo, DECam |
| 2004 VN112 | | | |
| 2010 GB174 | | | |
| 2007 TG422 | | | |
| 2013 RF98 | | | |
| 2013 FT28 | | (Sheppard & Trujillo 2016; memory) | |
| 2013 SL102 | | | |
| 2013 FL28 | | | |
| 2013 RA109 | | | |
| 2013 SY99 | | | (memory, unverified) OSSOS |
| 2014 WB556 | | | |
| 2014 SR349 | | (Sheppard & Trujillo 2016; memory) | |
| 2014 TU115 | | | |
| 2014 SX403 | | | |
| 2014 FE72 | | | |
| 2015 DM319 | | | |
| 2015 GT50 | | | |
| 2015 DY248 | | | |
| 2015 RX245 | | | (memory, unverified) OSSOS |
| 2015 BP519 | | | (memory, unverified) DES |
| 2015 KG163 | | | (memory, unverified) OSSOS |
| 2015 TG387 | | (Sheppard et al. 2019; memory) | (memory, unverified) Sheppard-Trujillo, Subaru |
| 2016 SA59 | | | |
| 2016 SD106 | | | |
| 2017 OF201 | arXiv May 2025 (2505.15806) [V] | Cheng, Li & Yang 2025 | archival DECaLS + CFHT [V] |
| 2018 VM35 | | | |
| 2019 EU5 | | | |
| 2021 DK18 | | | |
| 2021 RR205 | | | |
| 2024 FX26 | | | |

To fill this table reliably: query the MPC database / JPL SBDB (first observation date and MPEC id) and Bernardinelli et al. 2022 / Bannister et al. 2018 / Sheppard et al. 2019 tables. MPC and JPL hosts were also blocked in this session.
