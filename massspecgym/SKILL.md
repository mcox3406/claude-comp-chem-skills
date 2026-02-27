---
name: massspecgym
description: MassSpecGym benchmark reference — dataset schema, Python API, transforms, evaluation metrics, retrieval pipeline, and common patterns. Use when working with MassSpecGym data, models, or evaluation.
---

# MassSpecGym Reference

NeurIPS 2024 Spotlight benchmark for molecular discovery from MS/MS spectra. Three challenges: de novo generation, molecule retrieval, spectrum simulation.

- **GitHub:** https://github.com/pluskal-lab/MassSpecGym
- **Paper:** arXiv:2410.23326
- **HuggingFace:** `roman-bushuiev/MassSpecGym`
- **PyPI:** `massspecgym` (v1.3.1, requires Python ≥3.11)

## Installation

```bash
pip install massspecgym
# or with extras:
pip install massspecgym[notebooks,dev]
```

## Dataset Schema

231,104 spectra total (train: 194,119 / val: 19,429 / test: 17,556). HuggingFace dataset with a single `main` subset.

When loaded via `load_massspecgym()`, `identifier` becomes the DataFrame **index** (not a column). The 13 columns are:

| Column | Type (loaded) | Type (raw HF) | Description |
|--------|--------------|----------------|-------------|
| `mzs` | np.ndarray (float64) | str | m/z values per peak |
| `intensities` | np.ndarray (float64) | str | Intensity values per peak |
| `smiles` | str | str | SMILES string (5–255 chars) |
| `inchikey` | str | str | InChI Key (14 chars) |
| `formula` | str | str | Molecular formula |
| `precursor_formula` | str | str | Precursor formula |
| `parent_mass` | float64 | float64 | Parent mass (59–998) |
| `precursor_mz` | float64 | float64 | Precursor m/z (60–999) |
| `adduct` | str | str | `[M+H]+` or `[M+Na]+` |
| `instrument_type` | str | str | `Orbitrap` or `QTOF` |
| `collision_energy` | float64 | float64 | 0–358, nullable |
| `fold` | str | str | `train`, `val`, or `test` |
| `simulation_challenge` | bool | bool | Part of simulation challenge |

**Important:** In the raw HuggingFace Parquet files, `mzs` and `intensities` are comma-separated strings. `load_massspecgym()` parses them to numpy arrays automatically.

## Data Loading

### Quick load (pandas DataFrame)

```python
from massspecgym.utils import load_massspecgym

df = load_massspecgym()              # full dataset, identifier as index
df = load_massspecgym(fold="train")  # single fold
```

Returns DataFrame with `mzs`/`intensities` already parsed to numpy arrays.

### Direct HuggingFace load (no massspecgym dependency)

```python
from datasets import load_dataset
ds = load_dataset("roman-bushuiev/MassSpecGym")
# ds["val"] is a Dataset; use .to_pandas() to get DataFrame
# NOTE: mzs/intensities remain as strings — parse manually
```

### Direct Parquet load

```python
import pandas as pd
df = pd.read_parquet("hf://datasets/roman-bushuiev/MassSpecGym/main/val/*.parquet")
```

### Utility functions

```python
from massspecgym.utils import (
    load_massspecgym,         # full dataset → DataFrame
    load_massspecgym_mols,    # molecules only → Series of SMILES
    load_train_mols,          # training fold molecules
    load_val_mols,            # validation fold molecules
    load_unlabeled_mols,      # ~3.9M unlabeled molecules (disjoint from test)
    parse_spec_array,         # comma-separated string → np.ndarray
    pad_spectrum,             # pad to fixed peak count
    compute_mass,             # SMILES → exact mass via RDKit
)
```

## Splits

Fold assignments are in the `fold` column: `train`, `val`, `test`. Unique molecules: 25,046 in train, 31,602 total. The `MassSpecDataModule` reads these from the dataset metadata or a custom TSV with `identifier` and `fold` columns.

## Transforms

### Spectrum transforms (SpecTransform subclasses)

```python
from massspecgym.data.transforms import SpecTokenizer, SpecBinner, SpecToMzsInts

# Top-k peaks as (m/z, intensity) matrix, zero-padded
SpecTokenizer(n_peaks=60, prec_mz_intensity=1.1)
# Output: tensor of shape (n_peaks+1, 2) if prec_mz_intensity set, else (n_peaks, 2)

# Binned histogram representation
SpecBinner(max_mz=1005, bin_width=1, to_rel_intensities=True)
# Output: tensor of shape (max_mz / bin_width,)

# Sparse representation (no padding)
SpecToMzsInts(n_peaks=None, mz_from=10.0, mz_to=1000.0, mz_bin_res=0.01)
# Output: dict with separate mz and intensity tensors
```

### Molecule transforms (MolTransform subclasses)

```python
from massspecgym.data.transforms import (
    MolFingerprinter,    # SMILES → Morgan fingerprint vector
    MolToInChIKey,       # SMILES → InChIKey string
    MolToFormulaVector,  # SMILES → 118-dim element count vector
    MolToPyG,            # SMILES → PyTorch Geometric Data graph
    MolToFingerprints,   # SMILES → concatenated multi-type fingerprints
)

MolFingerprinter(type="morgan", fp_size=2048, radius=2)  # output: (fp_size,) int32
MolToInChIKey(twod=True)  # output: str (14-char InChIKey)
MolToFormulaVector()  # output: (118,) element count vector
MolToPyG(pyg_node_feats=[...], pyg_edge_feats=[...])
MolToFingerprints(fp_types=["morgan", "maccs", "rdkit"])
```

## Dataset Classes

### Base: MassSpecDataset

```python
from massspecgym.data import MassSpecDataset

dataset = MassSpecDataset(
    spec_transform=SpecTokenizer(n_peaks=60),
    mol_transform=MolFingerprinter(),
    pth=None,                  # downloads from HF if None
    return_mol_freq=False,     # include molecular frequency in output
    return_identifier=False,   # include identifier string in output
    identifiers_subset=None,   # filter to specific IDs
    dtype=torch.float32,
)
```

`__getitem__` returns dict. Base keys always present: `spec`, `mol`, `precursor_mz`, `adduct`. With defaults, that's all 4 keys. Set `return_mol_freq=True` to add `mol_freq` (tensor), `return_identifier=True` to add `identifier` (str).

### RetrievalDataset

```python
from massspecgym.data import RetrievalDataset

dataset = RetrievalDataset(
    spec_transform=SpecTokenizer(n_peaks=60),
    mol_transform=MolFingerprinter(fp_size=4096),
    mol_label_transform=MolToInChIKey(),  # default
    candidates_pth=None,                   # downloads candidate JSON if None
)
```

**Per-item keys** (from `__getitem__`): base keys plus `smiles` (str), `candidates_smiles` (list[str]), `candidates_mol` (tensor `[N_cands, fp_size]`), `labels` (list[bool]), `identifier`, `mol_freq`.

**Per-batch keys** (after `collate_fn`): candidates are **flattened** across the batch:
- `candidates_mol`: `(total_cands, fp_size)` — all candidates concatenated
- `candidates_smiles`: flat list of all candidate SMILES
- `labels`: `(total_cands,)` bool tensor
- `batch_ptr`: `(batch_size,)` int64 tensor — number of candidates per sample
- `smiles`: list of query SMILES
- Plus standard: `spec`, `mol`, `precursor_mz`, `adduct`, `mol_freq`, `identifier`

Candidate lists are **variable-length** per sample. Use `batch_ptr` to reconstruct per-sample groups (e.g., `torch.split(scores, batch_ptr.tolist())`).

### DataModule

```python
from massspecgym.data import MassSpecDataModule

dm = MassSpecDataModule(
    dataset=dataset,
    batch_size=32,
    num_workers=0,
    persistent_workers=True,
    split_pth=None,  # optional custom split TSV
)
```

Splits into train/val/test subsets based on the `fold` column. Training loader shuffles; val/test do not.

## Model Base Classes

Built on PyTorch Lightning.

### MassSpecGymModel (abstract base)

```python
from massspecgym.models.base import MassSpecGymModel, Stage

# Stage enum: TRAIN, VAL, TEST, NONE
# stage.to_pref() → "train_", "val_", "test_", or ""

class MassSpecGymModel(pl.LightningModule, ABC):
    def __init__(self, lr=1e-4, weight_decay=0.0, ...):
        ...

    @abstractmethod
    def step(self, batch: dict, stage: Stage) -> dict:
        # Must return dict with at least "loss" key
        ...

    def configure_optimizers(self):
        # Default: Adam(lr, weight_decay)
        ...
```

### RetrievalMassSpecGymModel

```python
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel

class MyRetrieval(RetrievalMassSpecGymModel):
    def __init__(self, at_ks=(1, 5, 20), myopic_mces_kwargs=None, **kwargs):
        super().__init__(at_ks=at_ks, **kwargs)

    def step(self, batch: dict, stage: Stage) -> dict:
        # batch keys: spec, mol, candidates_mol, labels, batch_ptr, smiles, ...
        # Must return: dict(loss=..., scores=...)
        # scores: flat tensor aligned with concatenated candidates_mol
        ...
```

Evaluation methods (called automatically):
- `evaluate_retrieval_step()` — computes HitRate@K (Recall@K) using `torchmetrics.functional.retrieval`
- `evaluate_mces_at_1()` — MCES distance between top-ranked and ground-truth molecule
- `test_step()` — sorts candidates by score, optionally saves to DataFrame

### DeNovoMassSpecGymModel

For structure generation from spectra.

### SimulationMassSpecGymModel

For spectrum prediction from molecular structures.

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| Recall@K (HitRate@K) | Ground truth in top-K ranked | higher = better |
| MRR | Mean Reciprocal Rank | higher = better |
| MCES@1 | Max Common Edge Subgraph distance at rank 1 | lower = better |

MCES is computed via `MyopicMCES` — graph edit distance based on maximum common edge subgraph.

## Retrieval Candidate Libraries

MassSpecGym provides **two** candidate libraries per spectrum for retrieval:

1. **Weight-based:** candidates by molecular weight from precursor m/z
2. **Formula-based:** candidates by matching chemical formula

Candidate mappings are JSON files: `{query_smiles: [candidate_smiles_1, ...]}`. Downloaded automatically by `RetrievalDataset`.

## Special Tokens

```python
from massspecgym.definitions import PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN
# "<pad>", "<s>", "</s>", "<unk>"
```

## Common Patterns

### Bypassing massspecgym for custom pipelines (like GLMR)

The `massspecgym` package is PyTorch-Lightning-heavy. For custom training loops:

```python
# Option A: use load_massspecgym() for data, skip their DataModule/Model classes
from massspecgym.utils import load_massspecgym
df = load_massspecgym(fold="train")
# df has parsed mzs (np.ndarray), intensities (np.ndarray), smiles (str), etc.
# Build your own Dataset/DataLoader around this DataFrame

# Option B: skip massspecgym entirely, load from HF
from datasets import load_dataset
ds = load_dataset("roman-bushuiev/MassSpecGym")
df = ds["val"].to_pandas()
# Parse mzs/intensities yourself:
import numpy as np
df["mzs"] = df["mzs"].apply(lambda s: np.fromstring(s, sep=","))
df["intensities"] = df["intensities"].apply(lambda s: np.fromstring(s, sep=","))
```

### Retrieval evaluation without their framework

```python
# Compute Recall@K manually
def recall_at_k(ranks, k):
    return (ranks <= k).float().mean().item() * 100

# Compute MRR manually
def mrr(ranks):
    return (1.0 / ranks).mean().item() * 100

# MCES requires rdkit + myopic-mces package
from myopic_mces import MCES
mces_dist = MCES(mol1, mol2)  # lower = more similar
```

### Loading candidate libraries directly

Candidate JSONs map each query SMILES to its candidate list. They're auto-downloaded by `RetrievalDataset`, but you can also download them manually from the HuggingFace repo's data files.

## Key Gotchas

1. **Python ≥3.11 required** for `massspecgym` package. If using Python 3.10, load data via HuggingFace `datasets` directly instead.
2. **mzs/intensities are strings** in the raw HuggingFace dataset — must parse to arrays.
3. **Variable-length candidates** — each query has a different number of candidates. Handle with padding or batch_ptr.
4. **Fold column** — use `fold` column for splits, not random splitting. The test fold is held out.
5. **InChIKey for matching** — ground-truth matching uses InChIKey comparison, not SMILES string equality.
6. **SpecTokenizer prepends precursor** — if `prec_mz_intensity` is set, an extra row `(precursor_mz, prec_mz_intensity)` is prepended, making output shape `(n_peaks+1, 2)`.
7. **MolFingerprinter returns int32** but the dataset casts tensors to float32 by default (`dtype` param). So `mol` in batch is float32 even though raw fingerprints are int32.
8. **RetrievalDataset always returns mol_freq/identifier** regardless of the `return_mol_freq`/`return_identifier` flags (those only apply to base MassSpecDataset).
