---
name: pkg-mgmt
description: Python package and environment management using uv and mamba. Use when installing packages, creating virtual environments, setting up new projects, or managing dependencies. NOT for general Python coding questions.
---

# Python Environment Skill (Comp Chem & AI Edition)

**Default to `uv`** for speed/ML; use **`mamba`** for heavy C++/Fortran binaries (e.g., OpenMM).

## uv — Fast Project Management (Primary)

*Best for: New projects, PyTorch/JAX, RDKit, CI/CD.*

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver written in Rust.

### Installation

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or with Homebrew
brew install uv
```

### Modern Workflow (Replaces pip/venv)

```bash
# Initialize project
uv init my-project && cd my-project

# (Optional) Pin Python version — uv downloads it automatically if missing
uv python pin 3.11

# Add dependencies (updates pyproject.toml & uv.lock)
uv add rdkit pandas torch
uv add --dev pytest ruff

# Sync environment (guarantees reproducibility)
uv sync

# Run commands in the environment
uv run python train_model.py
uv run pytest
```

### The "Quick Experiment"

Run a script with dependencies ephemerally (no permanent env created):

```bash
uv run --with rdkit --with matplotlib molecular_vis.py
```

### Legacy Workflow (pip-style)

```bash
# Create a virtual environment
uv venv
uv venv --python 3.11  # specific version

# Activate
source .venv/bin/activate

# Install packages
uv pip install rdkit scikit-learn pandas
uv pip install -r requirements.txt
uv pip install -e .
```

## mamba — Complex Binaries (Secondary)

*Best for: OpenMM, AmberTools, legacy projects, or strict system library requirements.*

[mamba](https://mamba.readthedocs.io/) is a fast, drop-in replacement for conda.

### Installation

```bash
# Install miniforge (includes mamba)
# macOS ARM
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh"
bash Miniforge3-MacOSX-arm64.sh

# macOS Intel
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh"
bash Miniforge3-MacOSX-x86_64.sh

# Linux
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh
```

### Reproducible Workflow

Always use `environment.yml` with `conda-forge`:

```yaml
# environment.yml
name: md-sim
channels:
  - conda-forge
dependencies:
  - python=3.11
  - openmm
  - ambertools
  - rdkit
  - numpy
  - pandas
  - pip  # Allow pip for pure python packages if needed
```

```bash
# Create from file
mamba env create -f environment.yml

# Update (use --prune to remove deleted deps)
mamba env update -f environment.yml --prune

# Export environment
mamba env export --no-builds > environment.yml
```

### Quick Commands

```bash
# Create environment
mamba create -n myenv python=3.11

# Activate/deactivate
mamba activate myenv
mamba deactivate

# Install packages
mamba install rdkit numpy pandas
```

## Decision Matrix: Chemist Edition

| Scenario | Tool | Reasoning |
|----------|------|-----------|
| **General ML / PyTorch** | `uv` | 100x faster, handles wheels perfectly |
| **Cheminformatics (RDKit)** | `uv` | RDKit PyPI wheels are now stable |
| **MD Sims (OpenMM/Amber)** | `mamba` | Complex CUDA/C++ bindings are fragile on PyPI |
| **Publishing/Sharing** | `uv` | `pyproject.toml` is the modern standard (PEP 621) |
| **Quick Scripts** | `uv` | `uv run --with` enables single-file reproducibility |
| **Legacy Projects** | `mamba` | If it already uses Conda, stick with it |

## The Hybrid Approach

Need mamba binaries (e.g., OpenMM) but want `uv` speed for everything else? Create the env with mamba, then use `uv pip` inside it:

```bash
mamba create -n hybrid-env openmm python=3.11 -c conda-forge
mamba activate hybrid-env
uv pip install torch rdkit scikit-learn  # Installs into the active Conda env
```

## Best Practices

### 1. Lockfiles are Mandatory

```bash
# uv does this automatically (uv.lock)
uv lock

# For mamba, export without build strings for portability
mamba env export --no-builds > environment.yml
```

### 2. pyproject.toml is Truth

Stop using `requirements.txt`. Define deps in `pyproject.toml`:

```toml
[project]
name = "my-project"
version = "0.1.0"
dependencies = [
    "rdkit",
    "numpy>=1.24",
    "pandas>=2.0",
]

[project.optional-dependencies]
dev = ["pytest", "ruff"]
```

### 3. Strict Channels for Mamba

Avoid ABI conflicts by strictly prioritizing conda-forge:

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

### 4. Never Install to System Python

```bash
# Bad
pip install rdkit

# Good
uv venv && source .venv/bin/activate && uv pip install rdkit
```

## CI/CD Pipeline

```yaml
# GitHub Actions example
- uses: astral-sh/setup-uv@v4
- run: uv sync
- run: uv run pytest
```
