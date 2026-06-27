# markutils

Small grab-bag of data and model utilities. Two CLI subcommands so far:

```bash
# Convert between csv / tsv / json / jsonl / parquet (format inferred from extension)
markutils convert input.csv output.parquet

# Print tensor shapes from a safetensors checkpoint
markutils inspect model.safetensors
```

Library-side: `markutils.utils.data.load_dataframe` / `save_dataframe` for the same
format-roundtrip helpers, and `markutils.checkpoints.get_state_dict` /
`print_state_dict_shapes` for safetensors inspection.

## Install

```bash
pip install git+https://github.com/marksverdhei/markutils.git
```

## Develop

This project uses [`uv`](https://github.com/astral-sh/uv):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/marksverdhei/markutils.git
cd markutils
uv sync --dev
source .venv/bin/activate
uv run pytest
```
