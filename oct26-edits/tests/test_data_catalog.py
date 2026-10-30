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
