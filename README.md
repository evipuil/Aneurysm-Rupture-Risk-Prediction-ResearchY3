# Aneurysm rupture model history

This private repository tracks the source-only evolution of the aneurysm rupture workflow from Version 1 through Version 4. Filenames are normalized across tags so GitHub can show meaningful line-by-line changes. The current snapshot is **Version 4**.

The Version 4 source snapshot dates to February 12, 2026 and was imported into Git on July 30, 2026. Git commit dates have not been backdated; the source snapshot field records the pre-existing file chronology.

## Version 4

Smaller model-specific trainers and focused execution scripts.

### Changes from Version 3

- Replaced the large prototype suite with focused clinical, geometry, flow-geometry, GNN, and ensemble trainers.
- Shortened the VTP inference and PINN correction paths.
- Added one launcher per maintained workflow.

## Code in this snapshot

- Python: `batch_vtp_inference.py`, `neural_networks.py`, `pinn_correction.py`, `train_clinical_age_sex.py`, `train_clinical_only.py`, `train_ensemble.py`, `train_flow_geometry.py`, `train_geometry.py`, `train_gnn.py`
- Shell: `run_clinical.sh`, `run_ensemble.sh`, `run_flow_geometry.sh`, `run_geometry.sh`, `run_gnn.sh`, `run_pinn_correction.sh`, `run_vtp_inference.sh`

Every Python and shell source carries a Version 4 source snapshot header. Version prefixes were removed from filenames and matching imports/launchers so the same logical file remains visible as an edit across tags.

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

See [CHANGELOG.md](CHANGELOG.md) for the detailed progression. Commit timestamps show the July 2026 import sequence; they do not claim that the original work happened on those commit dates.

## Scope and limitations

- This history contains project source only. Manuscripts, manuscript-editing scripts, raw and derived datasets, metadata, predictions, result tables, figures, checkpoints, logs, rendered documents, virtual environments, and local tool state are excluded.
- Result folders are outputs rather than independent source snapshots and are not represented as code tags.
- The snapshots preserve the modeling decisions of their versions. Cleanup is limited to stable filenames, import/launcher alignment, syntax repair, formatting, import hygiene, standardized comments, and removal of clearly redundant scaffolding.
- Results should not be compared across tags without accounting for changes in cohorts, feature boundaries, split logic, status handling, and dependencies.
