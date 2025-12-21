# Comparing fMRI Design Paradigms: Effects on Reliability in the Visual Cortex

This repository contains the analysis pipeline, statistical scripts, and visualization code for my Master's Thesis at Freie Universität Berlin (MCNB). The project systematically evaluates how different stimulus presentation designs—Event-Related, Sustained, and Miniblock—impact the reliability and consistency of BOLD responses in the visual cortex.

## Project Overview
Condition-rich fMRI studies often sample large stimulus spaces but can suffer from low trial-to-trial reliability. This study identifies the optimal design for maximizing signal-to-noise ratio (SNR) and measurement consistency by comparing three paradigms:

1.  **Event-Related (ER):** 0.75s presentation, 3.25s ISI.
2.  **Sustained (SUS):** 3.25s continuous presentation, 0.75s ISI.
3.  **Miniblock (MB):** 4.0s trial consisting of 4 rapid flashes (0.75s on, 0.25s off).

### Key Findings
* **Superior Reliability:** Both Miniblock (MB) and Sustained (SUS) designs significantly outperform traditional Event-Related (ER) designs across voxel-wise reliability, decoding accuracy, and representational similarity.
* **Miniblock Advantage:** In high-level visual areas (PPA and FFA), the MB design provided additional gains in consistency over the SUS design.
* **Impact:** These results suggest that maximizing stimulus exposure time via miniblocks is a superior strategy for condition-rich fMRI experiments, such as those used for encoding models or RSA.

## Analysis Pipeline
The project utilizes a state-of-the-art neuroimaging workflow:
* **Preprocessing:** Data organized in BIDS format and processed via `fMRIPrep` (v24.1.1).
* **Beta Estimation:** Single-trial responses estimated using `GLMSingle`, which fits unique HRFs to each voxel using ridge regression.
* **ROI Definition:** Functional ROIs (EVC, VRV, FFA, EBA, and PPA) defined via independent localizer runs.
* **Multivariate Analysis:** * **MVPA:** Linear SVM classification using *The Decoding Toolbox* (TDT).
    * **RSA:** Representational Similarity Analysis to compare within- and between-participant consistency.
    * **PCA:** Dimensionality assessment via Scree plots to evaluate representational structure.

## Repository Structure
```text
├── analysis/          # Python scripts for GLMSingle and reliability calculations
├── decoding/          # MATLAB scripts for MVPA (The Decoding Toolbox)
├── statistics/        # R scripts for repeated-measures ANOVA and post-hoc tests
├── plotting/          # Jupyter notebooks for generating t-maps and Scree plots
└── environment.yml    # Python environment dependencies
```
## Getting Started 
* Python 3.10+: nilearn, glmsingle, pybids, pandas
* MATLAB: Required for The Decoding Toolbox (TDT)
* R: Required for group-level frequentist statistics

## Basic usage
* Feel free to contact fabius.berner.fb@gmail.com to request access to raw images/preprocessed files
* If you choose to skip preprocessing, you can run the present scripts as they are

## Citation
```
@mastersthesis{berner2025comparing,
  author  = {Berner, Fabius},
  title   = {Comparing fMRI Design Paradigms: Effects on Reliability in the Visual Cortex},
  school  = {Freie Universität Berlin},
  year    = {2025},
  type    = {Master Thesis}
}
```

## Acknowledgements
Special thanks to Dr. Daniel Janini and Prof. Radoslaw Martin Cichy for their supervision and support throughout this project.
