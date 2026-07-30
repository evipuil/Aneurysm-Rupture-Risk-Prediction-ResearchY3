# Aneurysm rupture model history

This private repository tracks the source-only evolution of the aneurysm rupture workflow from Version 1 through Version 8. Filenames are normalized across tags so GitHub can show meaningful line-by-line changes. The current snapshot is **Version 8**.

The Version 8 source snapshot dates to April 28-29, 2026 and was imported into Git on July 30, 2026. Git commit dates have not been backdated; the source snapshot field records the pre-existing file chronology.

## Version 8

Self-contained PointNeXt-style geometry, flow, and stacking experiments.

### Changes from Version 7

- Added self-contained geometry, flow-geometry, and stacked ensemble trainers.
- Added post-hoc feature association and PINN-derived diagnostic tools.
- Added a single launcher for the Version 8 experiment set.

## Code in this snapshot

- Python: `diagnose_from_pinn.py`, `feature_extraction_posthoc.py`, `train_ensemble.py`, `train_flow_geometry.py`, `train_geometry.py`
- Shell: `run_all.sh`

Every Python and shell source carries a Version 8 source snapshot header. Version prefixes were removed from filenames and matching imports/launchers so the same logical file remains visible as an edit across tags.

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
| `v2` | February 1, 2026 | Fusion and ensemble experiments with repeatable training launchers |
| `v3` | February 8, 2026 | Comprehensive multimodal training plus flow-field repair and inspection tools |
| `v4` | February 12, 2026 | Smaller model-specific trainers and focused execution scripts |
| `v5` | March 14-29, 2026 | Hardened modular trainers, state-aware metrics, and an RRT ensemble variant |
| `v6` | April 21, 2026 | End-to-end flow simulation and PINN-only execution paths |
| `v7` | April 21-May 4, 2026 | Shared helpers, compact trainers, and a direct VTP-to-PINN pipeline |
| `v8` | April 28-29, 2026 | Self-contained PointNeXt-style geometry, flow, and stacking experiments |

See [CHANGELOG.md](CHANGELOG.md) for the detailed progression. Commit timestamps show the July 2026 import sequence; they do not claim that the original work happened on those commit dates.

## Scope and limitations

- This history contains project source only. Manuscripts, manuscript-editing scripts, raw and derived datasets, metadata, predictions, result tables, figures, checkpoints, logs, rendered documents, virtual environments, and local tool state are excluded.
- Result folders are outputs rather than independent source snapshots and are not represented as code tags.
- The snapshots preserve the modeling decisions of their versions. Cleanup is limited to stable filenames, import/launcher alignment, syntax repair, formatting, import hygiene, standardized comments, and removal of clearly redundant scaffolding.
- Results should not be compared across tags without accounting for changes in cohorts, feature boundaries, split logic, status handling, and dependencies.
