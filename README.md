# markutils

This project uses [`uv`](https://github.com/astral-sh/uv), which can be installed in one line: 
```bash
# On macOS and Linux.
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To get started with the project:

```
git clone https://github.com/marksverdhei/markutils.git
cd markutils
```

Then install project with
```
uv sync --dev
source .venv/bin/activate
```

Then you can run the default cli, which can be customized in `cli.py` and `pyproject.toml`  
```
$ markutils
Hello from markutils
```

How to install:
```bash
pip install git+https://github.com/marksverdhei/markutils.git
```
