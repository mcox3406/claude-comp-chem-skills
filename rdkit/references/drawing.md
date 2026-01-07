# Molecule Drawing Reference

## Basic Drawing

```python
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

mol = Chem.MolFromSmiles('c1ccccc1O')

# Generate 2D coordinates (required for drawing)
rdDepictor.Compute2DCoords(mol)
rdDepictor.StraightenDepiction(mol)  # optional: cleaner layout

# Simple image
img = Draw.MolToImage(mol, size=(300, 300))
img.save('molecule.png')
```

## MolDraw2D API (Full Control)

```python
from rdkit.Chem import Draw

# PNG output
d2d = Draw.MolDraw2DCairo(350, 300)

# SVG output
d2d = Draw.MolDraw2DSVG(350, 300)

# Draw and finish
d2d.DrawMolecule(mol)
d2d.FinishDrawing()

# Get output
png_data = d2d.GetDrawingText()  # bytes for Cairo, string for SVG

# Save
with open('molecule.png', 'wb') as f:
    f.write(d2d.GetDrawingText())
```

## Drawing Options

Access via `d2d.drawOptions()`:

```python
d2d = Draw.MolDraw2DCairo(350, 300)
dopts = d2d.drawOptions()

# Then modify dopts before drawing
```

### Display Options

```python
dopts.addAtomIndices = True      # show atom numbers
dopts.addBondIndices = True      # show bond numbers
dopts.explicitMethyl = True      # show CH3 explicitly
dopts.noAtomLabels = True        # hide all atom labels
dopts.includeRadicals = False    # hide radical markers
dopts.isotopeLabels = False      # hide isotope labels
dopts.atomLabelDeuteriumTritium = True  # show D/T instead of 2H/3H
```

### Colors

Colors are (R, G, B) or (R, G, B, A) tuples with values 0-1.

```python
# Background
dopts.setBackgroundColour((1, 1, 1))           # white
dopts.setBackgroundColour((0, 0, 0, 0))        # transparent

# Atom palettes
dopts.useBWAtomPalette()                        # black & white
dopts.useAvalonAtomPalette()                    # Avalon colors
dopts.useCDKAtomPalette()                       # CDK colors
dopts.updateAtomPalette({6: (0.7, 0, 0.7)})    # custom: purple carbons
dopts.setAtomPalette({...})                     # replace entire palette

# Highlighting
dopts.setHighlightColour((1, 0.8, 0.8))        # light red highlights
```

### Highlighting Atoms/Bonds

```python
# Highlight specific atoms
d2d.DrawMolecule(mol, highlightAtoms=[0, 1, 2, 3])

# Highlight with custom colors per atom
atom_colors = {0: (1, 0, 0), 1: (0, 1, 0), 2: (0, 0, 1)}
d2d.DrawMolecule(mol, highlightAtoms=[0, 1, 2], highlightAtomColors=atom_colors)

# Highlight bonds too
d2d.DrawMolecule(mol, 
    highlightAtoms=[0, 1, 2],
    highlightBonds=[0, 1],
    highlightAtomColors=atom_colors,
    highlightBondColors={0: (1, 0, 0), 1: (0, 1, 0)}
)
```

### Highlight Styles

```python
dopts.continuousHighlight = False    # don't connect highlights
dopts.circleAtoms = False            # no circles around atoms
dopts.atomHighlightsAreCircles = True  # force circular highlights
dopts.fillHighlights = False         # outline only, no fill
dopts.highlightRadius = 0.4          # highlight size
dopts.highlightBondWidthMultiplier = 12  # thicker bond highlights
```

### Fonts

```python
dopts.baseFontSize = 0.6            # default font size
dopts.legendFontSize = 16           # legend text size
dopts.fixedFontSize = 14            # force exact font size
dopts.minFontSize = 10              # minimum font size
dopts.maxFontSize = 20              # maximum font size
dopts.annotationFontScale = 0.75    # scale for annotations

# Change font (truetype)
dopts.fontFile = "BuiltinRobotoRegular"  # built-in option
dopts.fontFile = "/path/to/font.ttf"     # custom font
```

### Bond Drawing

```python
dopts.bondLineWidth = 2.0           # line thickness
dopts.multipleBondOffset = 0.15     # spacing for double/triple bonds
dopts.singleColourWedgeBonds = True # wedge bonds single color
dopts.scaleBondWidth = True         # scale with image size
```

### Stereochemistry

```python
dopts.addStereoAnnotation = True           # show R/S, E/Z labels
dopts.simplifiedStereoGroupLabel = True    # simplified stereo groups
dopts.includeChiralFlagLabel = True        # show ABS flag
dopts.unspecifiedStereoIsUnknown = True    # draw ? for unknown stereo
dopts.useMolBlockWedging = True            # use wedging from mol file
```

### Layout

```python
dopts.rotate = 30                   # rotation in degrees
dopts.padding = 0.1                 # padding fraction
dopts.drawMolsSameScale = True      # same scale for multiple mols
```

## Flexicanvas Mode

Let RDKit determine canvas size based on bond length:

```python
d2d = Draw.MolDraw2DCairo(-1, -1)  # flexicanvas
dopts = d2d.drawOptions()
dopts.fixedBondLength = 25         # target bond length in pixels
```

## ACS Publication Mode

For publication-quality figures meeting ACS standards:

```python
d2d = Draw.MolDraw2DCairo(-1, -1)
Draw.SetACS1996Mode(d2d.drawOptions(), Draw.MeanBondLength(mol))
d2d.DrawMolecule(mol)
d2d.FinishDrawing()

# Or simpler:
d2d = Draw.MolDraw2DCairo(-1, -1)
Draw.DrawMoleculeACS1996(d2d, mol, legend="Compound 1")
```

## Grid Images

```python
from rdkit.Chem import Draw

mols = [Chem.MolFromSmiles(s) for s in smiles_list]
legends = [f"Mol {i}" for i in range(len(mols))]

img = Draw.MolsToGridImage(
    mols,
    molsPerRow=4,
    subImgSize=(200, 200),
    legends=legends
)
img.save('grid.png')
```

## Complete Example: Publication Figure

```python
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')  # aspirin
rdDepictor.Compute2DCoords(mol)
rdDepictor.StraightenDepiction(mol)

# Publication quality
d2d = Draw.MolDraw2DCairo(-1, -1)
Draw.DrawMoleculeACS1996(d2d, mol, legend="Aspirin")

with open('aspirin_acs.png', 'wb') as f:
    f.write(d2d.GetDrawingText())
```

## Complete Example: Highlighted Substructure

```python
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
rdDepictor.Compute2DCoords(mol)

# Find carboxylic acid
pattern = Chem.MolFromSmarts('C(=O)O')
match = mol.GetSubstructMatch(pattern)

# Draw with highlight
d2d = Draw.MolDraw2DCairo(400, 300)
dopts = d2d.drawOptions()
dopts.setHighlightColour((1, 0.9, 0.9))

d2d.DrawMolecule(mol, highlightAtoms=match)
d2d.FinishDrawing()

with open('highlighted.png', 'wb') as f:
    f.write(d2d.GetDrawingText())
```

## Advanced Highlighting Techniques

### Similarity Maps (Continuous Data)

Visualize continuous numerical properties (charges, contributions, etc.) as color gradients:

```python
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import SimilarityMaps

mol = Chem.MolFromSmiles('COc1ccc2[nH]c(nc2c1)S(=O)Cc1ncc(C)c(OC)c1C')
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol)
AllChem.ComputeGasteigerCharges(mol)

# Get charges as weights
charges = [mol.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge')
           for i in range(mol.GetNumAtoms())]

# Draw similarity map
d2d = Draw.MolDraw2DSVG(550, 350)
dopts = d2d.drawOptions()
dopts.useBWAtomPalette()  # better visibility
SimilarityMaps.GetSimilarityMapFromWeights(mol, charges, d2d, colorMap='coolwarm')
d2d.FinishDrawing()
svg = d2d.GetDrawingText()
```

### Highlight Radii (Show Magnitude)

Use circle size to represent magnitude:

```python
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

mol = Chem.MolFromSmiles('c1ccccc1O')
rdDepictor.Compute2DCoords(mol)

d2d = Draw.MolDraw2DCairo(350, 300)
dopts = d2d.drawOptions()
dopts.atomHighlightsAreCircles = True

# Radii proportional to some property
radii = {i: 0.2 + 0.1 * i for i in range(mol.GetNumAtoms())}
colors = {i: (0.8, 0.8, 1.0) for i in range(mol.GetNumAtoms())}

d2d.DrawMolecule(mol,
    highlightAtoms=list(range(mol.GetNumAtoms())),
    highlightAtomColors=colors,
    highlightAtomRadii=radii,
    highlightBonds=[])
d2d.FinishDrawing()
```

### Atom Notes (Display Values)

Show numerical values directly on atoms:

```python
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, rdDepictor
import copy

mol = Chem.MolFromSmiles('c1ccccc1O')
mol = Chem.AddHs(mol)
AllChem.ComputeGasteigerCharges(mol)
rdDepictor.Compute2DCoords(mol)

molcopy = copy.deepcopy(mol)
for atom in molcopy.GetAtoms():
    charge = atom.GetDoubleProp('_GasteigerCharge')
    atom.SetProp('atomNote', f'{charge:.2f}')

d2d = Draw.MolDraw2DCairo(400, 300)
d2d.DrawMolecule(molcopy)
d2d.FinishDrawing()
```

### Multi-Color Substructure Highlighting

Highlight multiple substructures with different colors using `DrawMoleculeWithHighlights`:

```python
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from collections import defaultdict

mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
rdDepictor.Compute2DCoords(mol)

# Define substructures and colors
patterns = [
    (Chem.MolFromSmarts('C(=O)O'), (1, 0.6, 0.6, 0.5)),  # carboxylic acid - red
    (Chem.MolFromSmarts('c1ccccc1'), (0.6, 0.6, 1, 0.5)), # benzene - blue
]

atomHighlights = defaultdict(list)
bondHighlights = defaultdict(list)

for pattern, color in patterns:
    for match in mol.GetSubstructMatches(pattern):
        for aid in match:
            atomHighlights[aid].append(color)

d2d = Draw.MolDraw2DCairo(400, 300)
d2d.DrawMoleculeWithHighlights(mol, "", dict(atomHighlights), dict(bondHighlights), {}, {})
d2d.FinishDrawing()
```

### Ring Fill Highlighting (Polygons)

Draw filled polygons to highlight ring systems:

```python
from rdkit import Chem, Geometry
from rdkit.Chem import Draw, rdDepictor

mol = Chem.MolFromSmiles('c1ccc2ccccc2c1')  # naphthalene
rdDepictor.Compute2DCoords(mol)
conf = mol.GetConformer()

# Find rings
ring_pattern = Chem.MolFromSmarts('[r6]')
ring_atoms = mol.GetSubstructMatches(ring_pattern)

d2d = Draw.MolDraw2DCairo(350, 300)
d2d.ClearDrawing()

# Draw polygon for each ring
ring_info = mol.GetRingInfo()
for ring in ring_info.AtomRings():
    ps = [Geometry.Point2D(conf.GetAtomPosition(idx).x,
                           conf.GetAtomPosition(idx).y) for idx in ring]
    d2d.SetColour((0.9, 0.9, 0.6, 0.5))
    d2d.SetFillPolys(True)
    d2d.DrawPolygon(ps)

dopts = d2d.drawOptions()
dopts.clearBackground = False  # preserve polygons
d2d.DrawMolecule(mol)
d2d.FinishDrawing()
```
