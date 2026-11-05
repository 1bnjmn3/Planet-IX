import io
import json

import pytest

from planetix.data import select_etnos
from planetix.data.catalog import COLUMNS
from planetix.data.sbdb import _FIELD_MAP, fetch, parse_response

FIELDS = list(_FIELD_MAP)
ROW = ["90377", "Sedna", "  90377 Sedna (2003 VB12)", "549.5", "0.8613", "11.93",
       "144.48", "311.01", "76.22", "1022.8", "358.6", "1.5", "2461000.5", "620", "13149", "2"]
PAYLOAD = {"signature": {"version": "1.0"}, "fields": FIELDS, "data": [ROW], "count": 1}


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_parse_response_matches_catalog_schema():
    df = parse_response(PAYLOAD)
    assert list(df.columns) == COLUMNS
    row = df.iloc[0]
    assert row["a"] == pytest.approx(549.5)
    assert row["varpi"] == pytest.approx((144.48 + 311.01) % 360)
    assert len(select_etnos(df, "bb16")) == 1


def test_fetch_caches_and_reuses(tmp_path):
    calls = []

    def opener(url, timeout):
        calls.append(url)
        return FakeResponse(json.dumps(PAYLOAD).encode())

    first = fetch(cache_dir=tmp_path, opener=opener)
    second = fetch(cache_dir=tmp_path, opener=opener)
    assert len(calls) == 1
    assert "sbdb_query.api" in calls[0]
    assert first.equals(second)
    cached = json.loads(next(tmp_path.glob("sbdb_*.json")).read_text())
    assert "_retrieved_utc" in cached
