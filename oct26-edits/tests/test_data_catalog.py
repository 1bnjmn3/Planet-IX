import pytest

from planetix.data import CUTS, load_catalog, load_etnos, select_etnos
from planetix.data.catalog import COLUMNS


@pytest.fixture(scope="module")
def catalog():
    return load_catalog()


def test_catalog_is_complete_and_well_formed(catalog):
    assert len(catalog) == 6536
    assert list(catalog.columns) == COLUMNS
    for col in ["a", "e", "i", "Omega", "omega", "varpi"]:
        assert catalog[col].notna().all()
    assert catalog["Omega"].between(0, 360).all()
    assert catalog["i"].between(0, 180).all()


@pytest.mark.parametrize("cut", list(CUTS))
def test_cuts_are_respected(catalog, cut):
    df = select_etnos(catalog, cut)
    assert (df["a"] > CUTS[cut]["a_min"]).all()
    assert (df["q"] > CUTS[cut]["q_min"]).all()
    assert (df["arc_days"] >= 365).all()
    assert df["designation"].is_unique


def test_canonical_sample_contains_known_etnos():
    etnos = load_etnos("bb16")
    # Sedna, 2012 VP113 and Leleakuhonua (541132)
    for d in ["90377", "2012 VP113", "541132"]:
        assert d in set(etnos["designation"])


def test_quality_filter_only_removes(catalog):
    loose = select_etnos(catalog, "ts14", min_arc_days=0, reliable_only=False)
    strict = select_etnos(catalog, "ts14")
    assert set(strict["designation"]) <= set(loose["designation"])
    assert len(strict) < len(loose)


def test_year_span_arc_needs_proof_of_one_year(tmp_path):
    # 2025 LS2: MPCORB says '2025-2026' (read as 365.25 d) but the real arc is 350 d.
    import pandas as pd
    from planetix.data.catalog import apply_arc_overrides

    df = pd.DataFrame({
        "designation": ["2025 LS2", "X"], "a": [530.0, 530.0], "q": [43.0, 43.0], "e": [0.92, 0.92],
        "arc_days": [365.25, 365.25], "arc_days_min": [141.0, 141.0], "last_obs": ["20260521", "20260521"], "U": ["9", "9"],
    })
    assert select_etnos(df, "bb16").empty  # ambiguous arcs fail without an exact value
    ov = tmp_path / "arcs.csv"
    ov.write_text("designation,first_obs,last_obs\n2025 LS2,2025-06-05,2026-05-21\nX,2025-05-01,2026-05-21\n")
    out = apply_arc_overrides(df, ov)
    assert out["arc_days_min"].tolist() == [350.0, 385.0]
    assert select_etnos(out, "bb16")["designation"].tolist() == ["X"]
    # a stale override (different last_obs) is ignored
    assert apply_arc_overrides(df.assign(last_obs="20261001"), ov)["arc_days_min"].tolist() == [141.0, 141.0]


def test_paper_samples_unchanged_by_arc_rule():
    import pandas as pd
    from planetix.data.catalog import DATA_DIR

    for cut, n in {"bb16": 31, "ts14": 62, "detached": 19}.items():
        ref = pd.read_csv(DATA_DIR / "processed" / f"etnos_{cut}_mpc.csv")["designation"].astype(str)
        assert sorted(load_etnos(cut)["designation"]) == sorted(ref) and len(ref) == n
