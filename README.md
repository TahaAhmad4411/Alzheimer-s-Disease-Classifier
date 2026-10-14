# Multimodal Predictive Modeling of Alzheimer’s Disease Progression Using Video Vision Transformers

# Project Overview
Alzheimer's Disease (AD) is a progressive neurodegenerative disorder characterized by irreversible cognitive decline and memory loss. As the global population ages, the prevalence of AD is rising, making early and accurate diagnosis critical for effective intervention. However, disease progression is highly heterogeneous, influenced by a complex interplay of genetic, demographic, and clinical factors.

# Objective
Our goal is to develop a multimodal deep learning framework for Alzheimer's Disease that (1) classifies the current disease stage (CN / MCI / Dementia) from structural MRI plus demographic and genetic data, and (2) models **progression**: who converts to a worse stage, and how soon.

# Data Source
Longitudinal data from the Alzheimer's Disease Neuroimaging Initiative (ADNI):
- **Neuroimaging**: T1-weighted MPRAGE MRI (`MRI Scans-3D/`), several sessions per subject where available.
- **Clinical**: `ADNIMERGE_03May2025.csv`, one row per visit (diagnosis `DX`, age, sex, education, MMSE, ADAS).
- **Genetic**: APOE ε4 allele count and a TOMM40/APOE genetic risk score (`Final_TOMM40_APOE_GRS.csv`).

# Repository
| File | Content |
|---|---|
| `alzheimers_pipeline_v2.ipynb` | **Current pipeline**: leakage-free, with baselines, ViViT, a longitudinal model and survival analysis |
| `dl-project.ipynb` | First exploratory version, kept for reference. Its reported 81 % accuracy is **not valid** (see below) |
| `alzheimers_progression.gif` | Illustrative animation made of random noise, **not** model output or real MRI |

# Method (v2)
1. **Data integrity**: one scan per subject per session (the `MPRAGE_Repeat` scan and the duplicated `(1)` download folders are removed). Each scan is matched to the clinical visit **nearest in time** (≤ 180 days). Labels are the diagnosis *at that visit*. The split is **by subject** into train / validation / test, so no subject appears in two splits.
2. **MRI preprocessing**: reorientation to RAS, head mask and crop, foreground z-score, resampling to 32 × 128 × 128.
3. **Baselines**: majority class; logistic regression on demographics + genetics. A cognitive-score ablation shows that MMSE/ADAS largely encode the label.
4. **ViViT factorised encoder** (Arnab et al., 2021): a spatial transformer within each slice, then a transformer across slices with learned position embeddings. It has about 0.6 M parameters and is fused with the demographic/genetic branch. MRI-only and fused variants are compared. A **hybrid CNN + ViViT** (3-D conv front end) and a **3-D CNN** are also tested. Early stopping uses validation balanced accuracy, and the fusion head has a linear tabular skip path. Both a 3-class and a binary (CN vs impaired) task are evaluated.
5. **Longitudinal model**: frozen ViViT embeddings of a subject's successive scans (first 12 months) are fed to a GRU, which predicts later conversion.
6. **Time to conversion**: Cox proportional-hazards models (CN → MCI/Dementia, MCI → Dementia) on the full ADNIMERGE cohort, with censoring handled properly. Reported with the C-index and Kaplan–Meier curves by APOE ε4.

Metrics: balanced accuracy (with a 95 % CI from a subject-level bootstrap), macro-F1, ROC-AUC, and 5-fold grouped cross-validation for the baselines.

# Results (v2, subject-level held-out test set)
Balanced accuracy with a 95 % subject-bootstrap CI, and ROC-AUC.

**CN vs cognitively impaired (MCI + Dementia)**

| Model | Balanced acc. | AUC |
|---|---|---|
| Majority class | 0.50 | 0.50 |
| **Logistic regression: demographics + genetics + field strength** | **0.72** [0.56–0.85] | **0.74** |
| 3-D CNN: MRI + tabular | 0.63 [0.45–0.80] | 0.70 |
| Hybrid CNN+ViViT: MRI + tabular | 0.59 [0.44–0.75] | 0.58 |
| 3-D CNN: MRI only | 0.56 [0.47–0.69] | 0.62 |
| Hybrid CNN+ViViT: MRI only | 0.43 | 0.44 |
| *Ablation: + MMSE/ADAS (circular)* | *0.90* | *0.96* |

**CN / MCI / Dementia**: the best fair model is the fused hybrid CNN+ViViT (balanced acc. 0.51, AUC 0.65), close to logistic regression (0.55, AUC 0.65). MRI-only models stay at chance. Adding MMSE/ADAS gives 0.80, which reflects label circularity rather than predictive skill.

**Time to conversion (Cox model, full ADNIMERGE)**

| | CN → MCI/Dementia (736 subjects) | MCI → Dementia (987 subjects) |
|---|---|---|
| APOE ε4, per allele | HR 1.77 (p < 0.001) | HR 1.81 (p < 0.001) |
| Age, per year | HR 1.07 (p < 0.001) | HR 1.04 (p < 0.001) |
| MMSE, per point | n.s. | HR 0.78 (p < 0.001) |
| Test C-index | 0.57 | 0.61 |

**Longitudinal conversion (221 subjects)**: a GRU over a subject's sequence of scans (AUC 0.66) does not beat a first-scan + demographics model (AUC 0.69).

**Main conclusion:** with about 230 subjects and no skull stripping or registration, none of the image models (ViViT, hybrid CNN+ViViT, 3-D CNN) extracts reliable diagnostic signal from MRI beyond demographics and genetics. Diagnosis is also confounded with scanner field strength (χ² p < 0.001). The clinically meaningful findings come from the survival analysis.

# Known issues in v1 (`dl-project.ipynb`)
- Scans were merged with *every* visit of a subject, which duplicated each scan many times. Combined with a random scan-level split, the same scans appeared in both train and test.
- The test set was used for early stopping. The reported accuracy (81 %) was close to the majority-class rate (76 %).
- MMSE/ADAS (which define the diagnosis) were used as inputs. Genetic data was loaded but not used.
- The transformer had no positional embeddings or residual connections (43 M parameters). Augmentation was applied along the wrong axis.
- The "years to AD" model regressed visit time, and its event flag never fired (`DX` uses `Dementia`, not `AD`).

# Limitations
No skull stripping or template registration. A small cohort for a transformer. The deep models are evaluated on a single subject-level split. The longitudinal model is a pilot because few subjects have repeated scans *and* follow-up. ADNI is not representative of the general population.

# Running
On Kaggle, attach the `mri-scans-dataset`. Locally, place the CSVs anywhere inside this folder: paths are auto-detected. Then run `alzheimers_pipeline_v2.ipynb` top to bottom (a GPU is recommended for Sections 6–8). Preprocessed volumes are cached in `outputs/`.
