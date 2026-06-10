# Installation

pFLEX is developed and tested with Python 3.10. We recommend installing it in a dedicated Python 3.10 environment to keep the package and its scientific Python dependencies separate from other projects.

## Create an Environment

```bash
conda create -n p310 python=3.10
conda activate p310
pip install uv
```

## Install from PyPI

```bash
uv pip install pflex
```

or:

```bash
pip install pflex
```

## Install for Local Development

```bash
git clone https://github.com/tyasird/pFLEX.git
cd pFLEX
uv pip install -e .
```
