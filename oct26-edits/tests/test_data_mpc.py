import pytest

from planetix.data.mpc import _arc_days, read_mpcorb, unpack_epoch
from pathlib import Path

SAMPLE = Path(__file__).parent / "data" / "sample_mpcorb.dat"


def test_unpack_epoch_known_dates():
    assert unpack_epoch("K25BL") == 2461000.5  # 2025-11-21
    assert unpack_epoch("J9611") == 2450083.5  # 1996-01-01
    with pytest.raises(ValueError):
        unpack_epoch("X25BL")


def test_arc_days():
    assert _arc_days("1990-2026") == 36 * 365.25
    assert _arc_days("  36 days") == 36.0


def test_packed_provisional_designation_parses_by_column():
    # 2012 VP113 has packed designation K12VB3P, which broke the old regex parser.
    df = read_mpcorb(SAMPLE).set_index("designation")
    vp = df.loc["2012 VP113"]
    assert vp["a"] == pytest.approx(269.6820373)
    assert vp["e"] == pytest.approx(0.7009952)
    assert vp["i"] == pytest.approx(23.99666)
    assert vp["Omega"] == pytest.approx(90.90200)
    assert vp["omega"] == pytest.approx(294.29130)
    assert vp["q"] == pytest.approx(269.6820373 * (1 - 0.7009952))
    assert vp["varpi"] == pytest.approx((90.90200 + 294.29130) % 360)


def test_numbered_object_split_into_number_and_name():
    df = read_mpcorb(SAMPLE).set_index("designation")
    assert df.loc["90377", "name"] == "Sedna"
    assert df.loc["90377", "n_opp"] == 26
