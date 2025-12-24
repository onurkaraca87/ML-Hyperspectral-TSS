# ML-Hyperspectral-TSS

Machine learning–based estimation of **Total Suspended Solids (TSS)** using **in situ hyperspectral reflectance** and **PRISMA, EMIT, and PACE** satellite imagery, with applications to optically complex estuarine environments.

---

## 📌 Overview

Accurate monitoring of Total Suspended Solids (TSS) is essential for understanding sediment transport, water clarity, and ecosystem health in coastal and estuarine systems.  
This repository presents an integrated **hyperspectral remote sensing and machine learning framework** for TSS estimation, developed and tested in **Matagorda Bay** and **Trinity Bay (Texas, USA)**.

The workflow combines:
- Field-measured hyperspectral reflectance (400–900 nm),
- Laboratory-derived TSS concentrations,
- Advanced machine learning models,
- Next-generation hyperspectral satellite data.

---

## 📊 Data

### In situ Measurements
- **117 water samples** collected during monthly field campaigns (Aug 2024 – Jul 2025)
- Subsurface hyperspectral reflectance measured with a spectroradiometer (400–900 nm)
- Laboratory-based gravimetric TSS analysis

### Satellite Data
- **PRISMA** (ASI)
- **EMIT** (NASA)
- **PACE** (NASA)

---

## 🧠 Machine Learning Models

The following models are implemented and evaluated:

- **CatBoost**
- **Random Forest (RF)**
- **XGBoost**
- **LightGBM**
- **Partial Least Squares Regression (PLSR)**

Model performance is evaluated using:
- R²
- RMSE
- MAE
- MAPE
- Pearson’s correlation coefficient  
- **Taylor diagram analysis**

---

## 🏆 Key Findings

- **CatBoost and Random Forest** consistently outperform other models, achieving:
  - Test R² up to **0.965**
  - RMSE as low as **8.1 mg L⁻¹**
- Feature importance analysis shows that **red and red–near-infrared wavelengths** dominate TSS retrieval, consistent with sediment scattering physics.
- Trained models successfully capture:
  - Nearshore–offshore TSS gradients
  - River-influenced sediment pathways
  - Seasonal variability across multiple sensors

---

## 🗺️ Outputs

- Spatially explicit TSS maps derived from PRISMA, EMIT, and PACE imagery
- Multi-temporal assessment of estuarine sediment dynamics
- Visualization-ready products for coastal management and research

---

## 🛠️ Software & Tools

- **Python** (NumPy, SciPy, Pandas, Matplotlib, Scikit-learn)
- **GDAL / Rasterio**
- **ArcGIS Pro**
- **ENVI**
- **SeaDAS**

---

## 📁 Repository Structure (Planned)

```text
ML-Hyperspectral-TSS/
│
├── data/              # In situ and satellite data (not publicly shared)
├── scripts/           # Python scripts for preprocessing and modeling
├── notebooks/         # Jupyter notebooks
├── figures/           # Figures and maps
├── results/           # Model outputs
└── README.md
