# Fingerprint Selection Guide

Choosing the right fingerprint for your molecular ML task.

## Fingerprint Types Overview

| Fingerprint | Type | Captures | Best For |
|-------------|------|----------|----------|
| **Morgan (ECFP)** | Circular | Local atom environments | General purpose, similarity |
| **Morgan (FCFP)** | Circular | Pharmacophore features | Scaffold hopping |
| **RDKit** | Path-based | Bond paths | Substructure detection |
| **Atom Pairs** | Pair-based | Atom pair distances | Pharmacophore matching |
| **Topological Torsion** | Path-based | 4-atom paths | Conformational info |

## Similarity Score Variability

**The same molecule pair gives dramatically different similarity values** depending on fingerprint:

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Esomeprazole vs Lansoprazole (both PPIs)
mol1 = Chem.MolFromSmiles('COc1ccc2nc([nH]c2c1)[S@](=O)Cc1ncc(C)c(OC)c1C')  # esomeprazole
mol2 = Chem.MolFromSmiles('FC(F)(F)COc1ccnc(c1C)CS(=O)c2[nH]c3ccccc3n2')   # lansoprazole

fingerprints = {
    'RDKit': rdFingerprintGenerator.GetRDKitFPGenerator(),
    'Morgan2': rdFingerprintGenerator.GetMorganGenerator(radius=2),
    'Morgan3': rdFingerprintGenerator.GetMorganGenerator(radius=3),
    'AtomPair': rdFingerprintGenerator.GetAtomPairGenerator(),
    'TopTorsion': rdFingerprintGenerator.GetTopologicalTorsionGenerator(),
}

for name, gen in fingerprints.items():
    fp1 = gen.GetFingerprint(mol1)
    fp2 = gen.GetFingerprint(mol2)
    sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    print(f"{name}: {sim:.3f}")

# Example output:
# RDKit: 0.787
# Morgan2: 0.431
# Morgan3: 0.333
# AtomPair: 0.520
# TopTorsion: 0.421
```

## Noise Thresholds

Different fingerprints have different "noise thresholds" — the similarity value below which two molecules are essentially unrelated:

| Fingerprint | Approximate Noise Threshold |
|-------------|---------------------------|
| RDKit | ~0.5 |
| Morgan2 | ~0.3 |
| Morgan3 | ~0.25 |
| AtomPair | ~0.4 |

These are rough guidelines; actual thresholds depend on your dataset.

## Task-Specific Recommendations

### Similarity Search / Virtual Screening

**Recommended**: Morgan2 (ECFP4)

```python
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
```

- Well-characterized performance in benchmarks
- Good balance of sensitivity and specificity
- Fast to compute

### QSAR / Property Prediction

**Recommended**: Morgan2 with counts

```python
# Count fingerprints for regression
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fp = gen.GetCountFingerprintAsNumPy(mol)

# Or count simulation for bit vectors
gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=4096, countSimulation=True)
```

Count information helps models learn quantity relationships.

### Scaffold Hopping

**Recommended**: FCFP (Feature Morgan)

```python
gen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)
```

Abstracts atoms to pharmacophore types (H-bond donor, acceptor, aromatic, etc.), finding functionally similar but structurally different molecules.

### Substructure-Like Matching

**Recommended**: RDKit fingerprint

```python
gen = rdFingerprintGenerator.GetRDKitFPGenerator(
    minPath=1, maxPath=7, fpSize=2048)
```

Path-based fingerprints better capture exact substructure connectivity.

### Reaction Fingerprints

For comparing reactions, use Morgan fingerprints on difference fingerprints:

```python
from rdkit.Chem import AllChem

# Difference fingerprint (product - reactants)
rxn_fp = AllChem.CreateDifferenceFingerprintForReaction(rxn)
```

## Fingerprint Size Considerations

| Size | Pros | Cons |
|------|------|------|
| 1024 | Faster, less memory | More collisions |
| 2048 | Good default | Balanced |
| 4096 | Fewer collisions | Larger models |

For ML with many molecules, 2048 bits is usually sufficient. Use 4096 with count simulation for QSAR.

## Count vs Bit Fingerprints

```python
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Bit: presence/absence only
bit_fp = gen.GetFingerprintAsNumPy(mol)  # [0, 1, 0, 1, ...]

# Count: frequency information
count_fp = gen.GetCountFingerprintAsNumPy(mol)  # [0, 2, 0, 1, ...]
```

**Use counts for**:
- Regression tasks (QSAR)
- When fragment frequency matters

**Use bits for**:
- Classification
- Similarity search
- When speed/memory matters

## Sparse vs Dense Fingerprints

```python
# Dense (fixed size) - for ML
fp = gen.GetFingerprint(mol)        # 2048 bits

# Sparse (variable size) - for exact matching
sfp = gen.GetSparseFingerprint(mol)  # only on bits stored
```

Use dense for ML pipelines. Use sparse for database storage or when you need the actual bit indices.

## Complete Example: Comparing Fingerprints for a Task

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def evaluate_fingerprint(smiles_list, labels, gen, n_folds=5):
    """Evaluate a fingerprint for classification."""
    # Generate fingerprints
    fps = []
    valid_labels = []
    for smi, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(gen.GetFingerprintAsNumPy(mol))
            valid_labels.append(label)

    X = np.vstack(fps)
    y = np.array(valid_labels)

    # Cross-validate
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=n_folds, scoring='roc_auc')
    return scores.mean(), scores.std()

# Compare fingerprints
generators = {
    'Morgan2': rdFingerprintGenerator.GetMorganGenerator(radius=2),
    'Morgan3': rdFingerprintGenerator.GetMorganGenerator(radius=3),
    'RDKit': rdFingerprintGenerator.GetRDKitFPGenerator(),
    'AtomPair': rdFingerprintGenerator.GetAtomPairGenerator(),
}

for name, gen in generators.items():
    mean, std = evaluate_fingerprint(smiles, labels, gen)
    print(f"{name}: AUC = {mean:.3f} +/- {std:.3f}")
```
