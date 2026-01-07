---
name: mol-ml
description: Machine learning workflows for molecular data. Use when building ML models with molecules, splitting datasets, selecting fingerprints for ML, avoiding data leakage, or doing scaffold-based train/test splits. Covers common pitfalls in molecular ML.
---

# Molecular ML Skill

Best practices for machine learning with molecular data using RDKit.

## Scaffold Splitting

**Critical for avoiding data leakage** — molecules with the same scaffold in train and test sets lead to overly optimistic performance estimates.

See `references/scaffolds.md` for complete documentation.

### Quick Reference

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from collections import defaultdict

def get_scaffold(smiles):
    """Extract Murcko scaffold from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffoldSmiles(mol=mol, includeChirality=False)

def scaffold_split(smiles_list, test_size=0.2, random_state=42):
    """Split molecules by scaffold to avoid data leakage."""
    import random
    random.seed(random_state)

    # Group by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        scaffold = get_scaffold(smi)
        scaffold_to_indices[scaffold].append(idx)

    # Shuffle scaffolds and split
    scaffolds = list(scaffold_to_indices.keys())
    random.shuffle(scaffolds)

    train_idx, test_idx = [], []
    test_count = int(len(smiles_list) * test_size)

    for scaffold in scaffolds:
        indices = scaffold_to_indices[scaffold]
        if len(test_idx) < test_count:
            test_idx.extend(indices)
        else:
            train_idx.extend(indices)

    return train_idx, test_idx
```

## Fingerprint Selection

**Different fingerprints suit different tasks.** See `references/fingerprint-selection.md` for guidance.

### Quick Recommendations

| Task | Recommended FP | Why |
|------|---------------|-----|
| **Similarity search** | Morgan2 (ECFP4) | Captures local environment well |
| **Virtual screening** | Morgan2 or RDKit | Both work well |
| **QSAR modeling** | Morgan2 with counts | Count info helps regression |
| **Scaffold hopping** | FCFP (Feature Morgan) | Abstracts to pharmacophore features |
| **Substructure matching** | RDKit FP | Path-based, captures connectivity |

### Fingerprint Similarity Varies Dramatically

The **same molecule pair** gives very different similarity values with different fingerprints:

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

mol1 = Chem.MolFromSmiles('COc1ccc2nc([nH]c2c1)[S@](=O)Cc1ncc(C)c(OC)c1C')  # esomeprazole
mol2 = Chem.MolFromSmiles('FC(F)(F)COc1ccnc(c1C)CS(=O)c2[nH]c3ccccc3n2')   # lansoprazole

rdk_gen = rdFingerprintGenerator.GetRDKitFPGenerator()
mfp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

rdk_sim = DataStructs.TanimotoSimilarity(
    rdk_gen.GetFingerprint(mol1), rdk_gen.GetFingerprint(mol2))
mfp_sim = DataStructs.TanimotoSimilarity(
    mfp_gen.GetFingerprint(mol1), mfp_gen.GetFingerprint(mol2))

# RDKit FP: ~0.79, Morgan2: ~0.43
```

**Always specify the fingerprint type** when reporting similarity values.

## Common Pitfalls

### 1. Scaffold Leakage

**Problem**: Random splits put similar molecules in train and test.

**Solution**: Use scaffold-based splits.

### 2. Ignoring Fingerprint Type for Similarity

**Problem**: Reporting "Tanimoto similarity = 0.6" without specifying the fingerprint.

**Solution**: "Tanimoto similarity using Morgan2 = 0.6"

### 3. Using Bit Fingerprints for Regression

**Problem**: Bit fingerprints lose count information important for QSAR.

**Solution**: Use count fingerprints or count simulation:

```python
# Count fingerprints
fp = gen.GetCountFingerprintAsNumPy(mol)

# Or bit with count simulation
gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=4096, countSimulation=True)
```

### 4. Not Handling Parse Failures

**Problem**: Some SMILES fail to parse, causing index misalignment.

**Solution**: Track valid indices:

```python
valid_fps = []
valid_indices = []
for idx, smi in enumerate(smiles_list):
    mol = Chem.MolFromSmiles(smi)
    if mol is not None:
        valid_fps.append(gen.GetFingerprintAsNumPy(mol))
        valid_indices.append(idx)
```

### 5. Feature Matrix Shape Issues

**Problem**: Ragged arrays when some molecules fail.

**Solution**: Filter first, then stack:

```python
import numpy as np

fps = [gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s))
       for s in smiles_list if Chem.MolFromSmiles(s) is not None]
X = np.vstack(fps)  # shape: (n_valid, fp_size)
```
