"""Tests for markutils.utils.data — load_dataframe and save_dataframe."""

import json

import pandas as pd
import pytest

from markutils.utils.data import load_dataframe, save_dataframe


SAMPLE_DATA = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}, {"a": 3, "b": "z"}]


def _sample_df():
    return pd.DataFrame(SAMPLE_DATA)


# ---------------------------------------------------------------------------
# save_dataframe
# ---------------------------------------------------------------------------

class TestSaveDataframe:
    def test_save_csv(self, tmp_path):
        path = str(tmp_path / "out.csv")
        save_dataframe(_sample_df(), path)
        df = pd.read_csv(path)
        assert list(df["a"]) == [1, 2, 3]

    def test_save_tsv(self, tmp_path):
        path = str(tmp_path / "out.tsv")
        save_dataframe(_sample_df(), path)
        df = pd.read_csv(path, sep="\t")
        assert list(df["b"]) == ["x", "y", "z"]

    def test_save_json(self, tmp_path):
        path = str(tmp_path / "out.json")
        save_dataframe(_sample_df(), path)
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert data[0]["a"] == 1

    def test_save_jsonl(self, tmp_path):
        path = str(tmp_path / "out.jsonl")
        save_dataframe(_sample_df(), path)
        with open(path) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == 3
        assert rows[1]["b"] == "y"

    def test_save_parquet(self, tmp_path):
        path = str(tmp_path / "out.parquet")
        save_dataframe(_sample_df(), path)
        df = pd.read_parquet(path)
        assert list(df["a"]) == [1, 2, 3]

    def test_save_unsupported_format_raises(self, tmp_path):
        path = str(tmp_path / "out.xlsx")
        with pytest.raises(ValueError, match="Unsupported output format"):
            save_dataframe(_sample_df(), path)

    def test_save_no_index_in_csv(self, tmp_path):
        """Saved CSV should not have an index column."""
        path = str(tmp_path / "out.csv")
        save_dataframe(_sample_df(), path)
        df = pd.read_csv(path)
        assert "Unnamed: 0" not in df.columns

    def test_save_case_insensitive_extension(self, tmp_path):
        """Extension matching should be case-insensitive."""
        path = str(tmp_path / "out.CSV")
        save_dataframe(_sample_df(), path)
        assert (tmp_path / "out.CSV").exists()


# ---------------------------------------------------------------------------
# load_dataframe
# ---------------------------------------------------------------------------

class TestLoadDataframe:
    def _write_csv(self, path):
        _sample_df().to_csv(path, index=False)

    def test_load_csv(self, tmp_path):
        path = str(tmp_path / "data.csv")
        self._write_csv(path)
        df = load_dataframe(path)
        assert list(df["a"]) == [1, 2, 3]

    def test_load_tsv(self, tmp_path):
        path = str(tmp_path / "data.tsv")
        _sample_df().to_csv(path, sep="\t", index=False)
        df = load_dataframe(path)
        assert list(df["b"]) == ["x", "y", "z"]

    def test_load_json(self, tmp_path):
        path = str(tmp_path / "data.json")
        with open(path, "w") as f:
            json.dump(SAMPLE_DATA, f)
        df = load_dataframe(path)
        assert len(df) == 3

    def test_load_jsonl(self, tmp_path):
        path = str(tmp_path / "data.jsonl")
        with open(path, "w") as f:
            for row in SAMPLE_DATA:
                f.write(json.dumps(row) + "\n")
        df = load_dataframe(path)
        assert len(df) == 3
        assert list(df["a"]) == [1, 2, 3]

    def test_load_parquet(self, tmp_path):
        path = str(tmp_path / "data.parquet")
        _sample_df().to_parquet(path, index=False)
        df = load_dataframe(path)
        assert list(df["a"]) == [1, 2, 3]

    def test_load_unsupported_format_raises(self, tmp_path):
        path = str(tmp_path / "data.xlsx")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_dataframe(path)

    def test_load_case_insensitive_extension(self, tmp_path):
        path = str(tmp_path / "data.CSV")
        _sample_df().to_csv(path, index=False)
        df = load_dataframe(path)
        assert len(df) == 3

    def test_roundtrip_csv(self, tmp_path):
        path = str(tmp_path / "rt.csv")
        original = _sample_df()
        save_dataframe(original, path)
        loaded = load_dataframe(path)
        pd.testing.assert_frame_equal(original, loaded)

    def test_roundtrip_jsonl(self, tmp_path):
        path = str(tmp_path / "rt.jsonl")
        original = _sample_df()
        save_dataframe(original, path)
        loaded = load_dataframe(path)
        pd.testing.assert_frame_equal(original, loaded)

    def test_roundtrip_parquet(self, tmp_path):
        path = str(tmp_path / "rt.parquet")
        original = _sample_df()
        save_dataframe(original, path)
        loaded = load_dataframe(path)
        pd.testing.assert_frame_equal(original, loaded)
