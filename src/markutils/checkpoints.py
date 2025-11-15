import sys

from typing import Any
from safetensors import safe_open


def get_state_dict(path: str) -> dict[str, Any]:
    """Load a state dict from a safetensors file."""
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)

    return tensors


def print_state_dict_shapes(tensors: dict[str, Any]) -> None:
    """Print the shapes of tensors in a state dict."""
    for k in sorted(tensors.keys()):
        print(f"{k}: {tensors[k].shape=}")
