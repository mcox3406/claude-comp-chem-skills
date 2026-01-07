# Molecule Parsing & Sanitization Reference

Control how RDKit perceives chemistry during molecule parsing.

## Sanitization Overview

**Sanitization** performs chemistry perception and validation:
- Aromaticity detection
- Valence checking
- Implicit hydrogen assignment
- Ring detection
- Stereochemistry cleanup

By default, sanitization is enabled. Disable it for custom preprocessing workflows.

## SMILES Parsing Options

```python
from rdkit import Chem

# Default parsing (sanitize=True, removeHs=True)
mol = Chem.MolFromSmiles('CCO')

# Custom parsing with SmilesParserParams
params = Chem.SmilesParserParams()
params.sanitize = False    # disable chemistry perception
params.removeHs = False    # keep explicit hydrogens

mol = Chem.MolFromSmiles('[H]C([H])([H])C([H])([H])O[H]', params=params)
```

### Behavior Matrix

| sanitize | removeHs | Aromaticity | Valence Calc | Stereo Cleanup | Hs Removed |
|----------|----------|-------------|--------------|----------------|------------|
| True | True | Yes | Yes | Yes | Yes |
| True | False | Yes | Yes | Yes | No |
| False | True | No | Partial* | No | Yes |
| False | False | No | No | No | No |

*When removeHs=True with sanitize=False, valence is calculated for stereochemistry but full validation is skipped.

## Mol Block / SDF Parsing

```python
from rdkit import Chem

# Default parsing
mol = Chem.MolFromMolBlock(mol_block)
mol = Chem.MolFromMolFile('molecule.mol')

# With options
mol = Chem.MolFromMolBlock(mol_block, sanitize=False, removeHs=False)
mol = Chem.MolFromMolFile('molecule.mol', sanitize=False, removeHs=False)
```

**Important**: Unlike SMILES parsing, `removeHs=True` has **no effect** when `sanitize=False` for mol blocks. Explicit hydrogens remain regardless.

## SDF Supplier with Options

```python
from rdkit import Chem

# Default
suppl = Chem.SDMolSupplier('molecules.sdf')

# Without sanitization
suppl = Chem.SDMolSupplier('molecules.sdf', sanitize=False, removeHs=False)

for mol in suppl:
    if mol is None:
        continue
    # process mol...
```

## Custom Preprocessing Workflow

When you need to modify molecules before sanitization:

```python
from rdkit import Chem

# 1. Parse without sanitization
params = Chem.SmilesParserParams()
params.sanitize = False
params.removeHs = False
mol = Chem.MolFromSmiles(smiles, params=params)

if mol is None:
    raise ValueError("Failed to parse SMILES")

# 2. Do custom preprocessing
# ... your modifications here ...

# 3. Remove explicit Hs
mol = Chem.RemoveHs(mol)

# 4. Assign stereochemistry
Chem.AssignStereochemistry(mol,
    force=True,                      # reassign even if already assigned
    cleanIt=True,                    # remove invalid stereo labels
    flagPossibleStereoCenters=True   # mark potential chiral centers
)

# 5. Full sanitization (optional, if needed)
try:
    Chem.SanitizeMol(mol)
except Exception as e:
    print(f"Sanitization failed: {e}")
```

## AssignStereochemistry Parameters

```python
Chem.AssignStereochemistry(mol,
    cleanIt=True,                    # remove invalid stereo specifications
    force=True,                      # reassign even if already present
    flagPossibleStereoCenters=True   # mark atoms that could be chiral
)
```

| Parameter | Effect |
|-----------|--------|
| `cleanIt` | Removes invalid chirality/stereo labels |
| `force` | Recalculates even if stereo already assigned |
| `flagPossibleStereoCenters` | Sets `_ChiralityPossible` prop on potential centers |

## Validation Without Full Sanitization

Check for problems without modifying the molecule:

```python
from rdkit import Chem

mol = Chem.MolFromSmiles(smiles, params=params)  # sanitize=False

# Check for issues
problems = Chem.DetectChemistryProblems(mol)
for problem in problems:
    print(f"{problem.GetType()}: {problem.Message()}")
```

## Common Gotchas

### Invalid valences pass silently

```python
params = Chem.SmilesParserParams()
params.sanitize = False
mol = Chem.MolFromSmiles('F(C)C', params=params)  # fluorine with valence 2
# mol is valid! No error raised without sanitization
```

### Stereo info lost without removeHs

When `removeHs=False` and `sanitize=False`, stereochemistry information is not processed:

```python
params = Chem.SmilesParserParams()
params.sanitize = False
params.removeHs = False
mol = Chem.MolFromSmiles('C/C=C/C', params=params)
# E/Z info stored but not fully processed
```

### Mol block removeHs ignored

```python
# This does NOT remove hydrogens:
mol = Chem.MolFromMolBlock(block, sanitize=False, removeHs=True)
# Hs still present! Must use Chem.RemoveHs(mol) manually
```

## Complete Example: Handling Problematic Molecules

```python
from rdkit import Chem

def parse_permissive(smiles):
    """Parse SMILES permissively, then validate."""
    params = Chem.SmilesParserParams()
    params.sanitize = False
    params.removeHs = False

    mol = Chem.MolFromSmiles(smiles, params=params)
    if mol is None:
        return None, ["Failed to parse SMILES"]

    # Check for problems
    problems = Chem.DetectChemistryProblems(mol)
    if problems:
        return mol, [p.Message() for p in problems]

    # Sanitize if no problems
    try:
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        return mol, []
    except Exception as e:
        return None, [str(e)]

mol, errors = parse_permissive('CCO')
if errors:
    print("Problems:", errors)
```
