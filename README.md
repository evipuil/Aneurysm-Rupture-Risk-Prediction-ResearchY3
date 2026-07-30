# Aneurysm rupture model history

This private repository tracks the source-only evolution of the aneurysm rupture workflow from Version 1 through Version 14. Filenames are normalized across tags so GitHub can show meaningful line-by-line changes. The current snapshot is **Version 14**.

The Version 14 source snapshot dates to Mid-June-July 10, 2026 and was imported into Git on July 30, 2026. Git commit dates have not been backdated; the source snapshot field records the pre-existing file chronology.

## Version 14

Known-status filtering, expanded model families, and hardened execution.

### Changes from Version 13

- Excluded blank or unknown rupture status before targets, folds, feature extraction, and metrics are constructed.
- Added explicit status normalization, status auditing, a voxel CNN baseline, and fusion experiments.
- Reworked flow augmentation, GNN construction and caching, pooled predictions, and PINN unit handling.
- Hardened local and Slurm launchers and made result reporting current-run-only.
- Added grouped, legacy-composite, and modality Shapley importance analyses.

## Code in this snapshot

- Python: `analyze_grouped_feature_importance.py`, `analyze_legacy_composite_importance.py`, `analyze_modality_shapley_importance.py`, `base_trainer.py`, `feature_extraction.py`, `fusion_percentage_sweep.py`, `fusion_strategy_benchmark.py`, `late_fusion_ensemble.py`, `make_result_figures.py`, `model_architectures.py`, `run_multimodal_local.py`, `run_non_pinn.py`, `rupture_status.py`, `tests/test_rupture_status.py`, `train_clinical.py`, `train_cnn.py`, `train_flow_geometry.py`, `train_geometry.py`, `train_geometry_clinical.py`, `train_geometry_flow_clinical.py`, `train_gnn.py`, `train_pinn.py`, `train_pinn_carreau.py`, `unknown_status_audit.py`, `visualize_pinn_flow.py`
- Shell: `run_cnn.sh`, `run_flow_optimized.sh`, `run_non_pinn.sh`, `run_pinn.sh`

Every Python and shell source carries a Version 14 source snapshot header. Version prefixes were removed from filenames and matching imports/launchers so the same logical file remains visible as an edit across tags.

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
| `v9` | April 29-30, 2026 | Configurable early, late, and attention fusion experiments |
| `v10` | May 1, 2026 | Gated geometry, flow, clinical, and global-feature ensemble |
| `v11` | May 2-9, 2026, with a later trainer maintenance pass | One training entry point for the full model suite plus resumable aggregation |
| `v12` | May 9 prototype, finalized June 14, 2026 | Shared trainer and architecture modules with one entry point per model family |
| `v13` | June 15, 2026 | Leakage controls, group-aware validation, and expanded evaluation metrics |
| `v14` | Mid-June-July 10, 2026 | Known-status filtering, expanded model families, and hardened execution |

See [CHANGELOG.md](CHANGELOG.md) for the detailed progression. Commit timestamps show the July 2026 import sequence; they do not claim that the original work happened on those commit dates.

## Scope and limitations

- This history contains project source only. Manuscripts, manuscript-editing scripts, raw and derived datasets, metadata, predictions, result tables, figures, checkpoints, logs, rendered documents, virtual environments, and local tool state are excluded.
- Result folders labeled V15-V24 are outputs, not independent source snapshots, and are therefore not represented as code tags.
- The snapshots preserve the modeling decisions of their versions. Cleanup is limited to stable filenames, import/launcher alignment, syntax repair, formatting, import hygiene, standardized comments, and removal of clearly redundant scaffolding.
- Results should not be compared across tags without accounting for changes in cohorts, feature boundaries, split logic, status handling, and dependencies.
