# BuildDataCollector (BDC)

Two things in one repo:

- **Collector**: a local Flask app (`server.py`, port 5000) that stores Minecraft builds with
  their screenshots, AI image and TRELLIS mesh. State lives on disk: `data/builds/` (tracked)
  and `data/collector.db` (untracked, rebuilt from disk). See `README.md`.
- **Synth tooling**: `synth/`, offline training-pair generation on top of `mcvoxel`, the voxel
  core in the sibling `../Real2Minecraft` repo (installed editable via `requirements-synth.txt`).

## The pivot (2026-10-11)

Human-built schematics are now the training targets, and each model input is generated from
its target (schem → coloured mesh → `synth/degrade.py` → voxels at 2x block resolution).
This reverses the old plan, where Objaverse meshes were the source and `mcvoxel` rules made
the targets. Source of truth: `docs/mesh-to-schem-plan.md`, Version 4 (D13–D18, phase plan).

## Parked, not deleted

| Part | Status |
| --- | --- |
| Objaverse path: `synth/select.py`, `fetch.py`, `objaverse_index.py`, `run.py` | PARKED |
| Grammar houses: `house_grammar.py`, `gen_houses.py`, `house_params.py` | PARKED |
| The 16 synthetic-house builds in `data/builds/` | EXCLUDED from any real-build dataset |
| `synth/degrade.py`, `trellis_compare.py`, `real_eval.py`, `mcvoxel` | ACTIVE |

Full table: `docs/dataset-pipeline.md`, Status. Licensing (tiers A/B/C/X):
`docs/dataset-licensing.md`, "Schematic dataset (2026-10)".

## Rules

- Never modify `data/builds/`, `data/collector.db` or other data folders without being asked.
- Treat the schematic dataset folder named in the plan as read-only until the Phase 2 freeze.
- Changing Python code needs a server restart (README, "Restart after changing Python code").

## Tests

```powershell
.venv\Scripts\python -m unittest discover -s tests
```
