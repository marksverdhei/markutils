"""Tests for markutils.utils — save/load dataframe helpers."""
import os
import tempfile

import pandas as pd
import pytest

from markutils.utils import save_dataframe, load_dataframe


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def _roundtrip(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, filename)
        save_dataframe(df, path)
        return load_dataframe(path)


# ---------------------------------------------------------------------------
# save_dataframe
# ---------------------------------------------------------------------------

class TestSaveDataframe:
    def test_saves_csv(self, tmp_path):
        path = str(tmp_path / "out.csv")
        save_dataframe(_df(), path)
        assert os.path.exists(path)

    def test_saves_parquet(self, tmp_path):
        path = str(tmp_path / "out.parquet")
        save_dataframe(_df(), path)
        assert os.path.exists(path)

    def test_saves_json(self, tmp_path):
        path = str(tmp_path / "out.json")
        save_dataframe(_df(), path)
        assert os.path.exists(path)

    def test_saves_jsonl(self, tmp_path):
        path = str(tmp_path / "out.jsonl")
        save_dataframe(_df(), path)
        assert os.path.exists(path)

    def test_saves_tsv(self, tmp_path):
        path = str(tmp_path / "out.tsv")
        save_dataframe(_df(), path)
        assert os.path.exists(path)

    def test_unsupported_format_raises(self, tmp_path):
        path = str(tmp_path / "out.xlsx")
        with pytest.raises(ValueError, match="Unsupported output format"):
            save_dataframe(_df(), path)


# ---------------------------------------------------------------------------
# load_dataframe
# ---------------------------------------------------------------------------

class TestLoadDataframe:
    def test_load_csv(self, tmp_path):
        path = str(tmp_path / "out.csv")
        save_dataframe(_df(), path)
        result = load_dataframe(path)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_load_parquet(self, tmp_path):
        path = str(tmp_path / "out.parquet")
        save_dataframe(_df(), path)
        result = load_dataframe(path)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_load_json(self, tmp_path):
        path = str(tmp_path / "out.json")
        save_dataframe(_df(), path)
        result = load_dataframe(path)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_load_jsonl(self, tmp_path):
        path = str(tmp_path / "out.jsonl")
        save_dataframe(_df(), path)
        result = load_dataframe(path)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_load_tsv(self, tmp_path):
        path = str(tmp_path / "out.tsv")
        save_dataframe(_df(), path)
        result = load_dataframe(path)
        assert list(result.columns) == ["a", "b"]
        assert len(result) == 3

    def test_unsupported_format_raises(self, tmp_path):
        path = str(tmp_path / "out.xlsx")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataframe(path)


# ---------------------------------------------------------------------------
# Round-trip correctness
# ---------------------------------------------------------------------------

class TestRoundTrip:
    def test_csv_values_preserved(self):
        df = _df()
        result = _roundtrip(df, "data.csv")
        assert list(result["a"]) == [1, 2, 3]
        assert list(result["b"]) == ["x", "y", "z"]

    def test_parquet_values_preserved(self):
        df = _df()
        result = _roundtrip(df, "data.parquet")
        assert list(result["a"]) == [1, 2, 3]
        assert list(result["b"]) == ["x", "y", "z"]

    def test_jsonl_values_preserved(self):
        df = _df()
        result = _roundtrip(df, "data.jsonl")
        assert list(result["a"]) == [1, 2, 3]
        assert list(result["b"]) == ["x", "y", "z"]

    def test_tsv_values_preserved(self):
        df = _df()
        result = _roundtrip(df, "data.tsv")
        assert list(result["a"]) == [1, 2, 3]
        assert list(result["b"]) == ["x", "y", "z"]

    def test_empty_dataframe_csv(self):
        df = pd.DataFrame({"a": [], "b": []})
        result = _roundtrip(df, "empty.csv")
        assert len(result) == 0
        assert list(result.columns) == ["a", "b"]

    def test_csv_case_insensitive_extension(self, tmp_path):
        """Extensions should be matched case-insensitively."""
        path = str(tmp_path / "out.CSV")
        save_dataframe(_df(), path)
        assert os.path.exists(path)
        result = load_dataframe(path)
        assert len(result) == 3
