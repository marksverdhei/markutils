"""Tests for markutils.utils (save_dataframe and load_dataframe)."""

from __future__ import annotations

import pandas as pd
import pytest

from markutils.utils import load_dataframe, save_dataframe


class TestSaveDataframe:
    def _df(self):
        return pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    def test_saves_csv(self, tmp_path):
        path = str(tmp_path / "data.csv")
        save_dataframe(self._df(), path)
        assert (tmp_path / "data.csv").exists()

    def test_saves_jsonl(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        save_dataframe(self._df(), path)
        assert (tmp_path / "data.jsonl").exists()

    def test_unsupported_format_raises(self, tmp_path):
        path = str(tmp_path / "data.xlsx")
        with pytest.raises(ValueError, match="Unsupported output format"):
            save_dataframe(self._df(), path)


class TestLoadDataframe:
    def _df(self):
        return pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})

    def test_roundtrip_csv(self, tmp_path):
        path = str(tmp_path / "data.csv")
        save_dataframe(self._df(), path)
        loaded = load_dataframe(path)
        pd.testing.assert_frame_equal(self._df(), loaded)

    def test_roundtrip_jsonl(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        save_dataframe(self._df(), path)
        loaded = load_dataframe(path)
        pd.testing.assert_frame_equal(self._df(), loaded)

    def test_unsupported_format_raises(self, tmp_path):
        path = str(tmp_path / "data.xlsx")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataframe(path)
