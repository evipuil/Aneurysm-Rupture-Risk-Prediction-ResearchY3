# Change log

This log describes pre-existing source snapshots imported into Git on July 30, 2026. Commit timestamps record the import sequence and have not been backdated.

## Version 10

Source snapshot: May 1, 2026.

- Replaced the Version 9 experiment pair with a unified multibranch classifier.
- Added auxiliary branch losses, fold-level normalization, and balanced sampling.
- Added pooled prediction aggregation across seeds.

## Version 9

Source snapshot: April 29-30, 2026.

- Added switchable early, late, and attention fusion modes.
- Added a lightweight dry run for tensor-shape validation.
- Refined stacked ensembling and multi-seed execution.

## Version 8

Source snapshot: April 28-29, 2026.

- Added self-contained geometry, flow-geometry, and stacked ensemble trainers.
- Added post-hoc feature association and PINN-derived diagnostic tools.
- Added a single launcher for the Version 8 experiment set.

## Version 7

Source snapshot: April 21-May 4, 2026.

- Centralized metadata matching, point-cloud operations, metrics, folds, and logging.
- Rewrote model-specific trainers around the shared core and extracted ensemble logic.
- Added a Fourier/residual PINN pipeline, one-case wrapper, and dynamic metric plotting.

## Version 6

Source snapshot: April 21, 2026.

- Added a combined flow-simulation-to-PINN correction pipeline.
- Added a PINN-only pipeline and a dedicated flow-pipeline launcher.
- Carried the Version 5 trainers forward unchanged while the new pipeline was evaluated.

## Version 5

Source snapshot: March 14-29, 2026.

- Added expanded classification metrics and a shared plotting utility.
- Added the RRT ensemble experiment and its launcher.
- Added a suite submission script and more consistent early-stopping controls.

## Version 4

Source snapshot: February 12, 2026.

- Replaced the large prototype suite with focused clinical, geometry, flow-geometry, GNN, and ensemble trainers.
- Shortened the VTP inference and PINN correction paths.
- Added one launcher per maintained workflow.

## Version 3

Source snapshot: February 8, 2026.

- Added a comprehensive rupture model spanning geometry, flow, and clinical inputs.
- Added von Mises repair and velocity-field visualization utilities.
- Expanded the combined classifier while retaining the Version 2 experiment suite.

## Version 2

Source snapshot: February 1, 2026.

- Added standalone fusion and ensemble training paths.
- Expanded geometry and multimodal classifiers and added a single-case PINN diagnostic.
- Added a repeatable shell training matrix for the main model variants.

## Version 1

Source snapshot: January 29, 2026, with retained support files from December 2025.

- Added VTP batch inference and DeepONet inference utilities.
- Added separate geometry PointNet, graph, multichannel, and PINN rupture classifiers.
- Added the first combined geometry-hemodynamic classifier and PINN correction workflow.
