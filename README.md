# Aneurysm rupture model history

This private repository tracks the source-only evolution of the aneurysm rupture workflow from Version 1 through Version 1. Filenames are normalized across tags so GitHub can show meaningful line-by-line changes. The current snapshot is **Version 1**.

The Version 1 source snapshot dates to January 29, 2026, with retained support files from December 2025 and was imported into Git on July 30, 2026. Git commit dates have not been backdated; the source snapshot field records the pre-existing file chronology.

## Version 1

Initial geometry, hemodynamic, graph, and physics-informed rupture prototypes.

### Initial snapshot

- Added VTP batch inference and DeepONet inference utilities.
- Added separate geometry PointNet, graph, multichannel, and PINN rupture classifiers.
- Added the first combined geometry-hemodynamic classifier and PINN correction workflow.

## Code in this snapshot

- Python: `batch_vtp_inference.py`, `combined_rupture_classification.py`, `geometry_pointnet.py`, `gnn_rupture_classification.py`, `inference_deeponet.py`, `multichannel_pointnet_rupture.py`, `neural_networks.py`, `optimized_pinn.py`, `pinn_correction_batch.py`, `pinn_rupture_classification.py`, `visualize_advanced_params.py`

Every Python and shell source carries a Version 1 source snapshot header. Version prefixes were removed from filenames and matching imports/launchers so the same logical file remains visible as an edit across tags.

## Setup

Python 3.10 or newer is recommended for the normalized history.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

PyTorch, PyTorch Geometric, and torch-cluster builds must match the local CPU/CUDA environment. Historical training runs may also require cluster-specific resources.

## Inputs

Data is intentionally not stored here. Most snapshots expect a metadata CSV with rupture status and case identifiers plus per-case geometry/hemodynamic files. Paths and CLI options vary by tag; inspect the current launchers and each Python entry point before running an older snapshot.

## Version history

| Tag | Source snapshot | Main change |
| --- | --- | --- |
| `v1` | January 29, 2026, with retained support files from December 2025 | Initial geometry, hemodynamic, graph, and physics-informed rupture prototypes |

See [CHANGELOG.md](CHANGELOG.md) for the detailed progression. Commit timestamps show the July 2026 import sequence; they do not claim that the original work happened on those commit dates.

## Scope and limitations

- This history contains project source only. Manuscripts, manuscript-editing scripts, raw and derived datasets, metadata, predictions, result tables, figures, checkpoints, logs, rendered documents, virtual environments, and local tool state are excluded.
- Result folders are outputs rather than independent source snapshots and are not represented as code tags.
- The snapshots preserve the modeling decisions of their versions. Cleanup is limited to stable filenames, import/launcher alignment, syntax repair, formatting, import hygiene, standardized comments, and removal of clearly redundant scaffolding.
- Results should not be compared across tags without accounting for changes in cohorts, feature boundaries, split logic, status handling, and dependencies.
