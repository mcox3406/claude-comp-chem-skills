# Fingerprint Generation Reference

**Always use `rdFingerprintGenerator`** — the old API (`AllChem.GetMorganFingerprint`, etc.) is deprecated.

## Generator Creation

```python
from rdkit.Chem import rdFingerprintGenerator

# Morgan fingerprints (ECFP)
# ECFP4 = radius 2, ECFP6 = radius 3
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Feature Morgan (FCFP) — uses pharmacophore-type atom invariants
fmgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=2048,
    atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
)

# RDKit fingerprint (path-based)
rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(
    minPath=1,
    maxPath=7,
    fpSize=2048
)

# Atom pairs
apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)

# Topological torsions
ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
```

## Generating Fingerprints

All generators have the same four methods:

```python
# Bit vector (fixed size)
fp = gen.GetFingerprint(mol)                    # ExplicitBitVect

# Count vector (fixed size)  
cfp = gen.GetCountFingerprint(mol)              # UIntSparseIntVect

# Sparse bit vector (variable size, very large)
sfp = gen.GetSparseFingerprint(mol)             # SparseBitVect

# Sparse count vector (variable size)
scfp = gen.GetSparseCountFingerprint(mol)       # ULongSparseIntVect
```

## NumPy Arrays

```python
# Direct to numpy (most efficient for ML)
np_bits = gen.GetFingerprintAsNumPy(mol)        # uint8 array
np_counts = gen.GetCountFingerprintAsNumPy(mol) # uint32 array
```

## Explaining Bits (Additional Output)

```python
from rdkit.Chem import rdFingerprintGenerator

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)

# Allocate output containers
ao = rdFingerprintGenerator.AdditionalOutput()
ao.AllocateAtomCounts()      # how many bits each atom sets
ao.AllocateAtomToBits()      # which bits each atom sets
ao.AllocateBitInfoMap()      # for Morgan: (center_atom, radius) per bit

fp = mfpgen.GetFingerprint(mol, additionalOutput=ao)

# Access the info
bit_info = ao.GetBitInfoMap()       # {bit_id: ((atom, radius), ...)}
atom_counts = ao.GetAtomCounts()    # tuple of counts per atom
atom_to_bits = ao.GetAtomToBits()   # tuple of bit tuples per atom
```

### Bit Info by Fingerprint Type

| Fingerprint | Use | Returns |
|-------------|-----|---------|
| Morgan | `AllocateBitInfoMap()` | (center_atom, radius) tuples |
| RDKit | `AllocateBitPaths()` | bond index tuples |
| TopologicalTorsion | `AllocateBitPaths()` | atom index tuples |
| AtomPair | `AllocateBitInfoMap()` | (atom1, atom2) tuples |

## Count Simulation

Simulates count fingerprints using bit vectors (useful when you need bit vectors but want count-like behavior):

```python
# Enable count simulation
mfpgen = rdFingerprintGenerator.GetMorganGenerator(
    radius=2,
    fpSize=4096,  # use 4x size for better simulation
    countSimulation=True
)
```

## Rooted Fingerprints

Generate fingerprints only from specific atoms:

```python
# Find atoms of interest
carboxyl = Chem.MolFromSmarts('[$(C(=O)[OH,O-])]')
matches = [x[0] for x in mol.GetSubstructMatches(carboxyl)]

# Generate fingerprint rooted at those atoms
fp = gen.GetFingerprint(mol, fromAtoms=matches)
```

## Saving Generator Info

```python
# Get string describing generator parameters
info_string = gen.GetInfoString()
# Example output:
# 'Common arguments : countSimulation=0 fpSize=2048 ...'
```

## Similarity Metrics

**CRITICAL**: Always specify which fingerprint when reporting similarity values. Different fingerprints give dramatically different results for the same molecule pair:

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

print(f"RDKit FP: {rdk_sim:.2f}")   # ~0.79
print(f"Morgan2: {mfp_sim:.2f}")    # ~0.43
```

**Best practice**: "The Tanimoto similarity using Morgan2 fingerprints was 0.43" — not just "the Tanimoto similarity."

## Complete Example: Similarity Search

```python
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Create generator
mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# Generate fingerprints
query = Chem.MolFromSmiles('c1ccccc1O')
query_fp = mfpgen.GetFingerprint(query)

database_mols = [Chem.MolFromSmiles(s) for s in smiles_list]
database_fps = [mfpgen.GetFingerprint(m) for m in database_mols if m]

# Calculate similarities
similarities = DataStructs.BulkTanimotoSimilarity(query_fp, database_fps)

# Get top hits
hits = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:10]
```

## Complete Example: ML Features

```python
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def mol_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mfpgen.GetFingerprintAsNumPy(mol)

# Build feature matrix
fps = [mol_to_fp(s) for s in smiles_list]
fps = [fp for fp in fps if fp is not None]
X = np.vstack(fps)
```

## Visualizing Fingerprint Bits

Understanding what each fingerprint bit represents is critical for interpretability.

### Morgan Fingerprint Bits

```python
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Draw

mol = Chem.MolFromSmiles('CNCC(O)c1ccc(O)c(O)c1')  # epinephrine

# Generate fingerprint with bit info
fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2)
ao = rdFingerprintGenerator.AdditionalOutput()
ao.AllocateBitInfoMap()
fp = fpg.GetFingerprint(mol, additionalOutput=ao)
bit_info = ao.GetBitInfoMap()

# Draw a single bit
on_bits = list(fp.GetOnBits())
Draw.DrawMorganBit(mol, on_bits[0], bit_info)

# Draw multiple bits
tuples = [(mol, bit, bit_info) for bit in on_bits[:6]]
img = Draw.DrawMorganBits(tuples, molsPerRow=3, legends=[str(b) for b in on_bits[:6]])
```

Visual key: Central atom in **blue**, aromatic atoms in **yellow**, aliphatic rings in **gray**.

### RDKit Fingerprint Bits

```python
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Draw

mol = Chem.MolFromSmiles('CNCC(O)c1ccc(O)c(O)c1')

# Generate fingerprint with bit paths
fpg = rdFingerprintGenerator.GetRDKitFPGenerator(maxPath=5)
ao = rdFingerprintGenerator.AdditionalOutput()
ao.AllocateBitPaths()
fp = fpg.GetFingerprint(mol, additionalOutput=ao)
bit_paths = ao.GetBitPaths()

# Draw a single bit
on_bits = list(fp.GetOnBits())
Draw.DrawRDKitBit(mol, on_bits[0], bit_paths)

# Draw multiple bits
tuples = [(mol, bit, bit_paths) for bit in on_bits[:6]]
img = Draw.DrawRDKitBits(tuples, molsPerRow=3, legends=[str(b) for b in on_bits[:6]])
```

RDKit bits show the bond paths that contribute to each bit.
