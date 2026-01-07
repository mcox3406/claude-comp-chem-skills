# Murcko Scaffolds Reference

Murcko scaffolds extract the core ring systems and linkers from molecules, useful for clustering and train/test splitting.

## Basic Scaffold Extraction

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import (
    MurckoScaffoldSmiles,
    GetScaffoldForMol,
    MakeScaffoldGeneric
)

mol = Chem.MolFromSmiles('Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1')

# Get scaffold as SMILES (most common)
scaffold_smi = MurckoScaffoldSmiles(mol=mol, includeChirality=False)

# Get scaffold as mol object
scaffold_mol = GetScaffoldForMol(mol)

# Generic scaffold (all carbons, single bonds)
generic = MakeScaffoldGeneric(scaffold_mol)
```

## Scaffold Clustering

Group molecules by their scaffold:

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from collections import defaultdict

def cluster_by_scaffold(smiles_list):
    """Group SMILES by their Murcko scaffold."""
    scaffold_to_mols = defaultdict(list)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffold_to_mols[scaffold].append(smi)

    return dict(scaffold_to_mols)

clusters = cluster_by_scaffold(smiles_list)
print(f"Found {len(clusters)} unique scaffolds")
```

## Scaffold-Based Train/Test Split

Prevent data leakage by ensuring no scaffold appears in both sets:

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from collections import defaultdict
import random

def scaffold_split(smiles_list, labels, test_size=0.2, random_state=42):
    """
    Split dataset by scaffold.

    Returns:
        train_smiles, test_smiles, train_labels, test_labels
    """
    random.seed(random_state)

    # Group indices by scaffold
    scaffold_to_indices = defaultdict(list)
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold = None  # group invalid SMILES together
        else:
            scaffold = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffold_to_indices[scaffold].append(idx)

    # Shuffle scaffolds
    scaffolds = list(scaffold_to_indices.keys())
    random.shuffle(scaffolds)

    # Allocate scaffolds to train/test
    n_test = int(len(smiles_list) * test_size)
    train_idx, test_idx = [], []

    for scaffold in scaffolds:
        indices = scaffold_to_indices[scaffold]
        if len(test_idx) < n_test:
            test_idx.extend(indices)
        else:
            train_idx.extend(indices)

    # Build output
    train_smiles = [smiles_list[i] for i in train_idx]
    test_smiles = [smiles_list[i] for i in test_idx]
    train_labels = [labels[i] for i in train_idx]
    test_labels = [labels[i] for i in test_idx]

    return train_smiles, test_smiles, train_labels, test_labels
```

## Visualizing Scaffold Clusters

Display molecules grouped by scaffold:

```python
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor, Descriptors
from rdkit.Chem.Draw import MolsMatrixToGridImage
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

def visualize_scaffold_cluster(smiles_list, max_mols_per_scaffold=5):
    """Create grid showing scaffolds with their derivative molecules.

    Scaffold atoms are highlighted in each derivative molecule.
    """
    from collections import defaultdict

    # Cluster by scaffold
    scaffold_to_mols = defaultdict(list)
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        scaffold_smi = MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffold_to_mols[scaffold_smi].append(mol)

    # Build visualization matrix
    matrix = []
    legends = []
    highlights = []  # Track scaffold atoms to highlight

    for scaffold_smi, mols in scaffold_to_mols.items():
        scaffold = Chem.MolFromSmiles(scaffold_smi)
        rdDepictor.Compute2DCoords(scaffold)

        # Sort molecules by molecular weight
        mols = sorted(mols, key=lambda m: Descriptors.MolWt(m))[:max_mols_per_scaffold]

        row = [scaffold]
        row_legends = [f"Scaffold\n{scaffold_smi[:20]}"]
        row_highlights = [list(range(scaffold.GetNumAtoms()))]  # Highlight all scaffold atoms

        for mol in mols:
            # Align to scaffold orientation
            rdDepictor.GenerateDepictionMatching2DStructure(mol, scaffold)
            row.append(mol)
            row_legends.append(f"MW: {Descriptors.MolWt(mol):.1f}")
            # Highlight scaffold atoms in the derivative molecule
            match = mol.GetSubstructMatch(scaffold)
            row_highlights.append(list(match))

        matrix.append(row)
        legends.append(row_legends)
        highlights.append(row_highlights)

    # Create grid (pad rows to same length)
    max_cols = max(len(row) for row in matrix)
    for i, row in enumerate(matrix):
        while len(row) < max_cols:
            row.append(None)
            legends[i].append("")
            highlights[i].append([])

    img = MolsMatrixToGridImage(
        matrix,
        subImgSize=(200, 200),
        legendsMatrix=legends,
        highlightAtomLists=highlights  # Highlight scaffold atoms in derivatives
    )
    return img
```

## Highlighting Scaffold in Molecule

Show which atoms belong to the scaffold:

```python
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol

def draw_with_scaffold_highlight(mol):
    """Draw molecule with scaffold atoms highlighted."""
    rdDepictor.Compute2DCoords(mol)

    scaffold = GetScaffoldForMol(mol)
    match = mol.GetSubstructMatch(scaffold)

    d2d = Draw.MolDraw2DCairo(400, 300)
    dopts = d2d.drawOptions()
    dopts.setHighlightColour((0.9, 0.9, 0.6))

    d2d.DrawMolecule(mol, highlightAtoms=match)
    d2d.FinishDrawing()
    return d2d.GetDrawingText()
```

## Generic vs Non-Generic Scaffolds

```python
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import (
    GetScaffoldForMol,
    MakeScaffoldGeneric
)

mol = Chem.MolFromSmiles('c1ccc(C2CCNCC2)cc1')  # phenylpiperidine

# Non-generic: preserves atom types and bond orders
scaffold = GetScaffoldForMol(mol)
# c1ccc(C2CCNCC2)cc1

# Generic: all carbons, all single bonds
generic = MakeScaffoldGeneric(scaffold)
# C1CCC(C2CCCCC2)CC1
```

Use **generic scaffolds** when you want broader structural classes (ignores heteroatoms and aromaticity).

## Notes on RDKit's Murcko Implementation

- Based on Bemis-Murcko framework but with RDKit-specific variations
- Sidechains are removed, linkers between rings are kept
- `includeChirality=False` is recommended for clustering (stereoisomers share scaffold)
