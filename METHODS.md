# Multimodal Aneurysm Rupture Risk Modeling

This repository contains computational methods for aneurysm rupture classification using
vascular geometry, hemodynamic measurements, clinical variables, graph representations,
and physics-informed neural networks. The current release is **Version 14**. Earlier
releases are available through the `v1`–`v14` tags.

## Research scope

The project evaluates complementary representations of cerebral aneurysms:

- geometry-only point-cloud models;
- hemodynamic and geometry–flow models;
- clinical baselines and multimodal fusion models;
- graph neural networks and voxel-based convolutional networks;
- Newtonian and Carreau physics-informed neural networks; and
- post-hoc feature, modality, and fusion analyses.

Version 14 restricts supervised analyses to cases with a known rupture status and
maintains patient-aware validation boundaries. It also adds a voxel CNN baseline,
expanded fusion experiments, scalable graph construction, consistent pooled prediction
exports, and explicit hemodynamic unit metadata.

## Repository structure

| Path | Purpose |
| --- | --- |
| `base_trainer.py` | Shared case discovery, cohort filtering, cross-validation, optimization, metrics, and prediction export |
| `model_architectures.py` | Point-cloud, clinical, fusion, voxel-CNN, and graph architectures |
| `train_*.py` | Model-specific training programs |
| `feature_extraction.py` | Per-case feature extraction and statistical importance analysis |
| `rupture_status.py` | Rupture-status normalization and known-status cohort selection |
| `fusion_*.py`, `late_fusion_ensemble.py` | Fusion strategy and weighting experiments |
| `analyze_*.py` | Feature-group, composite, and modality-level sensitivity analyses |
| `make_result_figures.py` | Summary figures, tables, captions, and report generation |
| `run_non_pinn.py`, `run_non_pinn.sh` | Complete non-PINN model suite |
| `run_cnn.sh`, `run_flow_optimized.sh` | Focused Slurm workflows |
| `tests/` | Regression tests for rupture-status handling |

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

PyTorch, PyTorch Geometric, and `torch-cluster` must be installed with builds compatible
with the local operating system and CUDA environment. PINN and high-density point-cloud
training are best suited to GPU-equipped systems.

## Data requirements

The modeling workflows require:

1. a metadata CSV containing case identifiers, rupture status, and available clinical
   variables; and
2. one directory per aneurysm containing `hemodynamics_aggregate.csv` with spatial
   coordinates and available hemodynamic fields.

Patient-level data and derived research outputs are not distributed with this repository.
Input and output locations can be provided explicitly to the command-line workflows.

## Running the Version 14 model suite

Run the complete non-PINN workflow:

```bash
python run_non_pinn.py \
  --seed 42 \
  --metadata-path /path/to/metadata.csv \
  --data-dir /path/to/hemodynamics \
  --output-root /path/to/results_V14_suite \
  --feature-output-dir /path/to/results_v14_feature_extraction
```

Individual model programs expose their available options through `--help`:

```bash
python train_geometry.py --help
python train_flow_geometry.py --help
python train_gnn.py --help
python train_pinn.py --help
python train_pinn_carreau.py --help
```

Generate the Version 14 result package after the requested model outputs are available:

```bash
python make_result_figures.py \
  --suite-dir /path/to/results_V14_suite \
  --feature-dir /path/to/results_v14_feature_extraction \
  --out-dir /path/to/results_v14_figure_pack \
  --version-label V14
```

## Validation

```bash
python -m unittest discover -s tests
ruff check .
ruff format --check .
```

Full training is not part of the automated test suite because it requires the research
cohort and compatible accelerated-computing libraries.

## Version history

| Release | Primary advance |
| --- | --- |
| `v1` | Initial geometry, hemodynamic, graph, and physics-informed classifiers |
| `v2` | Fusion, ensemble, and repeatable training workflows |
| `v3` | Comprehensive multimodal modeling and flow-field quality tools |
| `v4` | Model-specific clinical, geometry, flow, graph, and ensemble programs |
| `v5` | Expanded evaluation metrics and the RRT ensemble |
| `v6` | End-to-end flow-simulation and PINN pipelines |
| `v7` | Shared training utilities and a direct VTP-to-PINN workflow |
| `v8` | Self-contained PointNeXt-style geometry, flow, and stacking experiments |
| `v9` | Configurable early, late, and attention fusion |
| `v10` | Gated geometry, flow, clinical, and global-feature ensemble |
| `v11` | Unified training interface for the complete model family |
| `v12` | Shared trainer and architecture modules with dedicated model entry points |
| `v13` | Leakage controls, group-aware validation, and expanded metrics |
| `v14` | Known-status filtering, additional model families, and robust multimodal evaluation |

Detailed release notes are provided in [CHANGELOG.md](CHANGELOG.md).

## Interpretation

- Performance comparisons across releases should account for differences in cohort
  definition, feature availability, validation strategy, and rupture-status handling.
- The `pointnext` option is a project-specific PointNeXt-style point-cloud abstraction.
- Reported model performance should be interpreted in the context of the study cohort and
  external validation requirements.
- This software is intended for research use and is not a clinical decision-support
  system.
