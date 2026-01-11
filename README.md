# Multimodal Predictive Modeling of Alzheimer’s Disease Progression Using Video Vision Transformers

# Project Overview
Alzheimer's Disease (AD) is a progressive neurodegenerative disorder characterized by irreversible cognitive decline and memory loss. As the global population ages, the prevalence of AD is rising, making early and accurate diagnosis critical for effective intervention. However, disease progression is highly heterogeneous, influenced by a complex interplay of genetic, demographic, and clinical factors.
# ObjectiveOur 
Our goal is to develop a robust, multimodal deep learning framework to predict the longitudinal progression of Alzheimer's Disease. Specifically, we aim to forecast the transition between disease stages (e.g., MCI to Dementia) to facilitate personalized therapeutic strategies and rigorous patient monitoring.
# Data Source & Methodology
We utilize a longitudinal dataset from the Alzheimer's Disease Neuroimaging Initiative (ADNI), integrating two distinct data modalities:
- High-Dimensional Neuroimaging: Serial MRI scans acquired at baseline and follow-up intervals (3 months, 6 months, 12 months, etc.) to capture spatiotemporal structural changes in the brain.
- Clinical & Genetic Data: Comprehensive tabular records including demographic factors (age, race, marital status) and key genetic biomarkers, specifically APOE E4 allele counts, which are significant risk factors for AD.
# Technical Approach
To effectively model the temporal evolution of neurodegeneration, we employ Video Vision Transformers (ViViT). This novel approach treats longitudinal MRI sequences as "video" frames, allowing the model to learn spatiotemporal features of brain atrophy over time. By fusing this imaging data with clinical tabular data via captioning or concatenation, our model predicts stage shifts with higher precision than unimodal approaches.

The work is in progress, so we'll be updating it time to time...
