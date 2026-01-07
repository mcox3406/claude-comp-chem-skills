# Computational Chemistry Skills for Claude

Claude Code skills for cheminformatics, molecular simulation, and AI for science workflows.

## Skills

| Skill | Description |
|-------|-------------|
| `rdkit` | Molecular fingerprints, drawing, property calculations, parsing (up-to-date API patterns) |
| `mol-ml` | ML workflows for molecules: scaffold splitting, fingerprint selection, avoiding data leakage |
| `pkg-mgmt` | Python package/environment management with uv and mamba |

## Installation

### Claude Code

Copy the skill folder into your project:

```bash
# Clone this repo
git clone https://github.com/mcox3406/claude-comp-chem-skills.git

# Copy skill(s) to your project
cp -r claude-comp-chem-skills/rdkit your-project/.claude/skills/
```

Or symlink for shared use across projects:

```bash
mkdir -p ~/.claude/skills
ln -s /path/to/claude-comp-chem-skills/rdkit ~/.claude/skills/rdkit
```

### Claude.ai / Claude App

1. Zip the skill folder (e.g., `cd rdkit && zip -r ../rdkit.skill .`)
2. Go to **Settings → Capabilities → Skills**
3. Upload the `.skill` file

## Structure

Each skill follows Claude's skill format:

```
skill-name/
├── SKILL.md              # Main instructions (always loaded)
└── references/           # Detailed docs (loaded on demand)
```

## Contributing

PRs welcome. Keep skills focused and concise.

## License

MIT

## Acknowledgements

- Many of the RDKit skills are based directly off of posts from Greg Landrum's [RDKit blog](https://greglandrum.github.io/rdkit-blog/). 
- The skills `pymatgen` and `torch_geometric` are based off of the skills in K-Dense-AI's [Claude scientific skills repository](https://github.com/K-Dense-AI/claude-scientific-skills/tree/main).

