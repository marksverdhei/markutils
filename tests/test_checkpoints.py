"""Tests for markutils.checkpoints — print_state_dict_shapes and get_state_dict."""

from unittest.mock import MagicMock, patch

from markutils.checkpoints import print_state_dict_shapes, get_state_dict


# ---------------------------------------------------------------------------
# print_state_dict_shapes
# ---------------------------------------------------------------------------

class TestPrintStateDictShapes:
    def test_prints_each_key(self, capsys):
        t1 = MagicMock()
        t1.shape = (4, 8)
        t2 = MagicMock()
        t2.shape = (16,)
        print_state_dict_shapes({"a": t1, "b": t2})
        out = capsys.readouterr().out
        assert "a" in out
        assert "b" in out

    def test_keys_sorted(self, capsys):
        """Keys are printed in sorted order."""
        t = MagicMock()
        t.shape = (1,)
        print_state_dict_shapes({"z_key": t, "a_key": t, "m_key": t})
        out = capsys.readouterr().out
        lines = [line for line in out.splitlines() if line.strip()]
        assert lines[0].startswith("a_key")
        assert lines[1].startswith("m_key")
        assert lines[2].startswith("z_key")

    def test_empty_dict_prints_nothing(self, capsys):
        print_state_dict_shapes({})
        out = capsys.readouterr().out
        assert out == ""

    def test_shape_appears_in_output(self, capsys):
        t = MagicMock()
        t.shape = (32, 64)
        print_state_dict_shapes({"layer": t})
        out = capsys.readouterr().out
        assert "(32, 64)" in out

    def test_single_key(self, capsys):
        t = MagicMock()
        t.shape = (100,)
        print_state_dict_shapes({"weight": t})
        out = capsys.readouterr().out
        assert "weight" in out
        assert "(100,)" in out

    def test_shape_rendered_without_debug_expression_leak(self, capsys):
        # Regression: the f-string '=' debug specifier (`{tensors[k].shape=}`)
        # echoed the literal expression, so `markutils inspect` printed lines
        # like `w: tensors[k].shape=(4, 8)`. Output should be just the shape.
        t = MagicMock()
        t.shape = (4, 8)
        print_state_dict_shapes({"w": t})
        out = capsys.readouterr().out
        assert "tensors[k].shape" not in out
        assert out.strip() == "w: (4, 8)"


# ---------------------------------------------------------------------------
# get_state_dict — mocked safe_open
# ---------------------------------------------------------------------------

class TestGetStateDict:
    def _make_mock_context(self, keys, tensors):
        """Build a mock safe_open context manager."""
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=ctx)
        ctx.__exit__ = MagicMock(return_value=False)
        ctx.keys.return_value = keys
        ctx.get_tensor.side_effect = lambda k: tensors[k]
        return ctx

    def test_returns_dict(self, tmp_path):
        fake_path = str(tmp_path / "model.safetensors")
        t = MagicMock()
        ctx = self._make_mock_context(["w"], {"w": t})
        with patch("markutils.checkpoints.safe_open", return_value=ctx):
            result = get_state_dict(fake_path)
        assert isinstance(result, dict)

    def test_all_keys_present(self, tmp_path):
        fake_path = str(tmp_path / "model.safetensors")
        tensors = {"w1": MagicMock(), "w2": MagicMock()}
        ctx = self._make_mock_context(list(tensors.keys()), tensors)
        with patch("markutils.checkpoints.safe_open", return_value=ctx):
            result = get_state_dict(fake_path)
        assert set(result.keys()) == {"w1", "w2"}

    def test_tensor_values_returned(self, tmp_path):
        fake_path = str(tmp_path / "model.safetensors")
        t = MagicMock(name="tensor")
        ctx = self._make_mock_context(["weight"], {"weight": t})
        with patch("markutils.checkpoints.safe_open", return_value=ctx):
            result = get_state_dict(fake_path)
        assert result["weight"] is t

    def test_empty_checkpoint(self, tmp_path):
        fake_path = str(tmp_path / "empty.safetensors")
        ctx = self._make_mock_context([], {})
        with patch("markutils.checkpoints.safe_open", return_value=ctx):
            result = get_state_dict(fake_path)
        assert result == {}

    def test_opens_with_pt_framework(self, tmp_path):
        """safe_open must be called with framework='pt'."""
        fake_path = str(tmp_path / "model.safetensors")
        ctx = self._make_mock_context([], {})
        with patch("markutils.checkpoints.safe_open", return_value=ctx) as mock_open:
            get_state_dict(fake_path)
        mock_open.assert_called_once_with(fake_path, framework="pt", device="cpu")
