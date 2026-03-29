"""Tests for markutils.utils — save/load dataframe helpers."""
import os
import tempfile

import pandas as pd
import pytest

from markutils.utils import save_dataframe, load_dataframe


def _df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def _roundtrip(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        save_dataframe(df, path)
        return load_dataframe(path)


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------

def test_roundtrip_csv():
    result = _roundtrip(_df(), "data.csv")
    pd.testing.assert_frame_equal(result, _df())


def test_roundtrip_parquet():
    result = _roundtrip(_df(), "data.parquet")
    pd.testing.assert_frame_equal(result, _df())


def test_roundtrip_json():
    result = _roundtrip(_df(), "data.json")
    pd.testing.assert_frame_equal(result, _df())


def test_roundtrip_jsonl():
    result = _roundtrip(_df(), "data.jsonl")
    pd.testing.assert_frame_equal(result, _df())


def test_roundtrip_tsv():
    result = _roundtrip(_df(), "data.tsv")
    pd.testing.assert_frame_equal(result, _df())


# ---------------------------------------------------------------------------
# Case-insensitive extension detection
# ---------------------------------------------------------------------------

def test_save_uppercase_csv_extension():
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.CSV")
        save_dataframe(_df(), path)
        assert os.path.exists(path)


def test_load_uppercase_csv_extension():
    result = _roundtrip(_df(), "data.CSV")
    pd.testing.assert_frame_equal(result, _df())


# ---------------------------------------------------------------------------
# Unsupported format raises ValueError
# ---------------------------------------------------------------------------

def test_save_unsupported_format_raises():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Unsupported"):
            save_dataframe(_df(), os.path.join(tmpdir, "data.xlsx"))


def test_load_unsupported_format_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataframe("/tmp/data.xlsx")


# ---------------------------------------------------------------------------
# Preserves data shape
# ---------------------------------------------------------------------------

def test_save_load_csv_preserves_shape():
    df = pd.DataFrame({"x": range(100), "y": range(100, 200)})
    result = _roundtrip(df, "big.csv")
    assert result.shape == df.shape


def test_save_load_parquet_preserves_columns():
    df = pd.DataFrame({"col_a": [1], "col_b": ["test"], "col_c": [3.14]})
    result = _roundtrip(df, "data.parquet")
    assert list(result.columns) == list(df.columns)
