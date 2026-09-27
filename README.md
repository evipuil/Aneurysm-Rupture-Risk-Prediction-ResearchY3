# Multimodal Aneurysm Rupture Risk Modeling

Can vascular geometry, hemodynamic measurements, and clinical variables help distinguish ruptured from unruptured cerebral aneurysms?

I developed this research code to compare those inputs through point-cloud models, multimodal fusion, graph neural networks, voxel CNNs, and physics-informed neural networks. The study connects aneurysm detection, flow estimation, and classification of rupture status.

**Research Year 3 · Python · PyTorch · Computational hemodynamics**  
[Research paper, 2025–26 (PDF)](papers/science-research-2025-26.pdf)

## Findings

| Rupture classification model | Validation AUROC |
| --- | ---: |
| Clinical variables | 0.561 |
| Geometry | 0.699 |
| Geometry and flow ensemble | 0.791 |

These values come from the table on page 21 and the final Results section on page 24 of the attached 2025–26 paper. The paper describes five-fold validation with an 80/20 training/validation division. Its abstract contains earlier values, so the table above follows the final Results section. Patient grouping is not established by that paper's description; these values should not be presented as patient-grouped or external-validation results, or as results from the subsequent Year 4 study.

![Training curve and model comparison tables from the Year 3 paper](docs/images/research-figure.png)

*Page 21 of the paper, reproduced unchanged. A physics training loss is not a measurement of agreement with an independent CFD or FEM reference.*

## My contributions

- Developed models for learning from vascular geometry and hemodynamic point data.
- Compared clinical, geometry, flow, graph, and voxel representations and combinations of modalities.
- Implemented physics-informed flow-modeling experiments and analysis of derived hemodynamic features.
- Established consistent case selection and evaluation across the modeling experiments.

The paper places this work within a broader detection, flow modeling, and visualization pipeline, with mentorship from Dr. Xianqi Li at Florida Institute of Technology. This repository contains the rupture-classification study. The [web](https://github.com/evipuil/HemoViz3D-Web-App-ResearchY1) and [VR](https://github.com/evipuil/HemoViz3D-VR-App-ResearchY2) projects document the earlier visualization work.

## Research papers

- [2023–24: Web visualization (PDF)](papers/science-research-2023-24.pdf)
- [2024–25: Virtual reality visualization (PDF)](papers/science-research-2024-25.pdf)
- [2025–26: Aneurysm detection and rupture modeling (PDF)](papers/science-research-2025-26.pdf)


## Methods

[Methods and analysis](METHODS.md) describes the model families, data requirements, and available programs. The [Year 4 study](https://github.com/evipuil/Cerebral-Aneurysm-Modeling-ResearchY4) examines patient-grouped validation, generalization, and FEM-informed flow models.

This study classifies retrospective rupture status. It does not establish prospective rupture risk or clinical benefit.
