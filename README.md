# ML-Hyperspectral-TSS — One-Page Overview
This repository provides an end-to-end workflow to estimate **Total Suspended Solids (TSS)** from **hyperspectral reflectance** using pretrained machine-learning models and apply them to **EMIT / PRISMA** hyperspectral imagery to generate **GeoTIFF TSS maps**.

---

## What’s inside
- **Pretrained ML models** (`.pkl`): CatBoost, LightGBM, XGBoost, Random Forest, PLSR  
- **Apply scripts** for hyperspectral sensors:
  - `Scripts/EMIT_Scripts/` (EMIT workflows)
  - `Scripts/Prisma_Scripts/` (PRISMA workflows)
- **Data download links** (large files are not stored in the repo):
  - `Dataset/Hyperspectral Dataset/EMIT_Download_link.txt`
  - `Dataset/Hyperspectral Dataset/Prisma_Download_Link.txt`

---

## Repository structure

```text
ML-Hyperspectral-TSS/
├─ Dataset/
│  └─ Hyperspectral Dataset/
│     ├─ EMIT_Download_link.txt
│     └─ Prisma_Download_Link.txt
├─ Machine Learning Models/
│  └─ Models/
│     ├─ Catboost_Model.pkl
│     ├─ LightGBM_Model.pkl
│     ├─ XgBoost_Model.pkl
│     ├─ Random_Forest_Model.pkl
│     └─ PLSR_VisNIR_400_900_Model.pkl
└─ Scripts/
   ├─ EMIT_Scripts/
   └─ Prisma_Scripts/




---

## Quick start
### 1) Install
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt




2) Apply a pretrained model (example: CatBoost → EMIT GeoTIFF)
Goal: Open an EMIT hyperspectral GeoTIFF, match model wavelengths to the closest raster bands, predict TSS, and export a single-band GeoTIFF.
Requirements / assumptions
The EMIT GeoTIFF must include band descriptions with wavelength values readable via rasterio (src.descriptions), e.g., Band i (xxxx.xxxx).
The model input features (example below) must match the exact training order.
Nearest-band wavelength matching uses a tolerance (commonly 8 nm, adjustable).


Example feature list (10-band input)
model_bands = ["X_430","X_611","X_582","X_735","X_797","X_620","X_586","X_407","X_664","X_870"]


Run (example)
python Scripts/EMIT_Scripts/apply_catboost_emit.py


Output
Single-band float32 GeoTIFF (e.g., CatBoost_TSS_YYYYMMDD.tif)
Georeferencing (CRS/transform/extent) inherited from the input raster
Invalid pixels exported as NaN (nodata = NaN)

Models
Pretrained models are stored in:
Machine Learning Models/Models/

import joblib
model = joblib.load("Machine Learning Models/Models/Catboost_Model.pkl")

License
MIT License (see LICENSE).

Contact
Onur Karaca — GitHub: @onurkaraca87
