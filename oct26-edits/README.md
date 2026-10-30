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
| `src/planetix/ml/` | calibrated autoencoder and classifier two-sample tests |
| `results/` | outputs of `scripts/run_*.py` |
| `data/` | raw snapshot and processed samples, see `data/README.md` |
| `scripts/` | thin command-line entry points |
| `notes/` | findings for the paper |
