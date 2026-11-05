# Planet IX, rebuilt (October 2026)

A clean, tested rewrite of the original numbered scripts as one Python package, `planetix`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[plot,dev]"        # add ,nbody for REBOUND
python scripts/build_etnos.py       # writes data/processed/
pytest
```

```python
from planetix.data import load_etnos
etnos = load_etnos("bb16")          # canonical sample: a > 250 AU, q > 30 AU, arc >= 1 yr
```

| path | contents |
|---|---|
| `src/planetix/angles.py` | shared angle helpers |
| `src/planetix/data/` | MPC parser, JPL SBDB client, canonical ETNO sample |
| `src/planetix/stats/` | circular statistics, null tests, PCA/clustering |
| `src/planetix/dynamics/` | secular theory and N-body checks |
| `src/planetix/bias/` | survey-selection modelling and bias-aware clustering tests |
| `src/planetix/inference/` | Bayes factor for a distant planet vs no planet, with per-object selection functions |
| `src/planetix/ml/` | calibrated autoencoder and classifier two-sample tests |
| `results/` | outputs of `scripts/run_*.py` |
| `data/` | raw snapshot and processed samples, see `data/README.md` |
| `scripts/` | thin command-line entry points |
| `notes/` | findings for the paper |
| `paper/` | mini-paper `planet9_bayes_factor.md`, its figures and result files (`scripts/infer_planet9.py`, `scripts/make_paper_figures.py`) |

## Second paper: evidence history, out-of-sample test and LSST forecast

```bash
python scripts/forecast_and_history.py            # history, predictive, selmarg, lsst, lsstsim, yield
python scripts/nbody_population.py                # 1 Gyr N-body forward models (~2 h per planet)
python scripts/make_paper2_figures.py             # figures 6-11
python scripts/make_numbers_tex.py                # paper/aas/numbers.tex from paper/results/
```

| path | contents |
|---|---|
| `paper/aas/ms.tex`, `refs.bib`, `numbers.tex` | AAS (AASTeX 6.31) manuscript skeleton for PSJ/AJ |
| `paper/PUBLISHABILITY_REVIEW.md` | referee-style evaluation and the iterations it drove |
| `paper/SECOND_PAPER_TRIGGERS.md` | checkable criteria the weekly data tracker reads |
| `paper/notes/literature_review.md` | literature check, with verification status per reference |
| `src/planetix/inference/lsst.py` | LSST selection model and information-rate forecast |
| `src/planetix/dynamics/population.py` | gigayear N-body model of the confined population |
