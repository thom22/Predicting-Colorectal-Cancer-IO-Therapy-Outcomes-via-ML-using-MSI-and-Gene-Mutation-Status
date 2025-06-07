# Predicting-Colorectal-Cancer-IO-Therapy-Outcomes-via-ML-using-MSI-and-Gene-Mutation-Status

This project focuses on predicting progression-free survival (PFS) outcomes in colorectal adenocarcinoma (COADREAD) patients treated with immunotherapy. The data is collected from the TCGA-COADREAD cohort, accessed via cBioPortal, integrated clinical variables, gene mutations (KRAS, NRAS, BRAF), microsatellite instability (MSI) scores, tumor mutation burden (TMB), and immune checkpoint gene expression (PD-1, PD-L1, CTLA-4).
Two modeling approaches were developed: a Survival model to estimate time-to-event outcomes while handling censoring, and a set of classification models to predict binary PFS status (progression vs. censored). The classification models achieved promising performance with AUC scores ranging from 0.81 to 0.87, and key predictive features aligned well with known biological markers, such as MSI and PD-L1 expression.

This work was completed as part of a research project at Johns Hopkins University.

---



### Interactive web app for real-time prediction of progression-free survival(PFS)/PFS Status

![colorectal_cancer_prediction](https://github.com/user-attachments/assets/514e8e51-dc70-40f3-9680-831147471dc2)
