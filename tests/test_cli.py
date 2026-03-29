"""Tests for markutils CLI (convert and inspect subcommands)."""
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from markutils.cli import build_parser, main, _cmd_convert, _cmd_inspect


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(tmp_path: Path, fmt: str) -> Path:
    """Write a tiny DataFrame in the requested format and return its path."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    p = tmp_path / f"data.{fmt}"
    if fmt == "parquet":
        df.to_parquet(p, index=False)
    elif fmt == "csv":
        df.to_csv(p, index=False)
    elif fmt == "tsv":
        df.to_csv(p, sep="\t", index=False)
    elif fmt == "json":
        df.to_json(p, orient="records", lines=False)
    elif fmt == "jsonl":
        df.to_json(p, orient="records", lines=True)
    return p


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------

class TestBuildParser:
    def test_no_command_parses(self):
        parser = build_parser()
        args = parser.parse_args([])
        assert args.command is None

    def test_convert_parses(self):
        parser = build_parser()
        args = parser.parse_args(["convert", "in.csv", "out.parquet"])
        assert args.command == "convert"
        assert args.input == "in.csv"
        assert args.output == "out.parquet"

    def test_inspect_parses(self):
        parser = build_parser()
        args = parser.parse_args(["inspect", "model.safetensors"])
        assert args.command == "inspect"
        assert args.path == "model.safetensors"


# ---------------------------------------------------------------------------
# convert subcommand
# ---------------------------------------------------------------------------

class TestCmdConvert:
    @pytest.mark.parametrize("src_fmt,dst_fmt", [
        ("csv", "parquet"),
        ("parquet", "csv"),
        ("csv", "jsonl"),
        ("jsonl", "tsv"),
        ("json", "csv"),
    ])
    def test_roundtrip(self, tmp_path, src_fmt, dst_fmt):
        src = _make_df(tmp_path, src_fmt)
        dst = tmp_path / f"out.{dst_fmt}"
        parser = build_parser()
        args = parser.parse_args(["convert", str(src), str(dst)])
        rc = _cmd_convert(args)
        assert rc == 0
        assert dst.exists()

    def test_output_has_correct_row_count(self, tmp_path, capsys):
        src = _make_df(tmp_path, "csv")
        dst = tmp_path / "out.parquet"
        parser = build_parser()
        args = parser.parse_args(["convert", str(src), str(dst)])
        rc = _cmd_convert(args)
        assert rc == 0
        captured = capsys.readouterr()
        assert "3 rows" in captured.out

    def test_unsupported_input_format(self, tmp_path):
        bad = tmp_path / "data.xyz"
        bad.write_text("garbage")
        dst = tmp_path / "out.csv"
        parser = build_parser()
        args = parser.parse_args(["convert", str(bad), str(dst)])
        rc = _cmd_convert(args)
        assert rc == 1

    def test_unsupported_output_format(self, tmp_path):
        src = _make_df(tmp_path, "csv")
        dst = tmp_path / "out.xyz"
        parser = build_parser()
        args = parser.parse_args(["convert", str(src), str(dst)])
        rc = _cmd_convert(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# inspect subcommand
# ---------------------------------------------------------------------------

class TestCmdInspect:
    def test_inspect_calls_print_shapes(self, tmp_path, capsys):
        fake_tensors = {
            "layer.weight": MagicMock(),
            "layer.bias": MagicMock(),
        }

        with (
            patch("markutils.checkpoints.get_state_dict", return_value=fake_tensors),
            patch("markutils.checkpoints.print_state_dict_shapes") as mock_print,
        ):
            parser = build_parser()
            args = parser.parse_args(["inspect", "fake.safetensors"])
            rc = _cmd_inspect(args)

        assert rc == 0
        mock_print.assert_called_once_with(fake_tensors)

    def test_inspect_prints_key_count(self, tmp_path, capsys):
        fake_tensors = {f"w{i}": MagicMock() for i in range(5)}

        with (
            patch("markutils.checkpoints.get_state_dict", return_value=fake_tensors),
            patch("markutils.checkpoints.print_state_dict_shapes"),
        ):
            parser = build_parser()
            args = parser.parse_args(["inspect", "fake.safetensors"])
            rc = _cmd_inspect(args)

        assert rc == 0
        captured = capsys.readouterr()
        assert "5" in captured.out

    def test_inspect_bad_path_returns_1(self, tmp_path):
        parser = build_parser()
        args = parser.parse_args(["inspect", str(tmp_path / "nonexistent.safetensors")])
        rc = _cmd_inspect(args)
        assert rc == 1


# ---------------------------------------------------------------------------
# main() exit codes
# ---------------------------------------------------------------------------

class TestMain:
    def test_no_command_exits_0(self):
        with patch("sys.argv", ["markutils"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code == 0

    def test_unknown_command_exits_nonzero(self):
        with patch("sys.argv", ["markutils", "bogus"]):
            with pytest.raises(SystemExit) as exc:
                main()
        assert exc.value.code != 0
