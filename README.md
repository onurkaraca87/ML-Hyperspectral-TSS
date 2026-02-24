ML-Hyperspectral-TSS
This repository provides an end-to-end workflow to estimate Total Suspended Solids (TSS) from hyperspectral reflectance using machine-learning models, and apply them to PRISMA / EMIT / PACE hyperspectral imagery to generate GeoTIFF TSS maps.

Main idea (2-step):
1) Train (tabular) -> produces a pretrained model file (.pkl)
2) Apply (raster) -> loads the .pkl, matches wavelengths to sensor bands, predicts pixel-wise TSS, exports a single-band GeoTIFF


----------------------------------------------------------------------
WHAT’S INSIDE
----------------------------------------------------------------------

Supported models (pretrained .pkl):
- CatBoost
- LightGBM
- XGBoost
- Random Forest
- PLSR

Supported sensors (apply scripts):
- PRISMA (L2D hyperspectral raster workflows)
- EMIT (hyperspectral raster workflows)
- PACE (hyperspectral products / exported raster workflows)

Outputs:
- Single-band TSS GeoTIFF (float32)
- NoData pixels exported as NaN (nodata = NaN)


----------------------------------------------------------------------
REPOSITORY STRUCTURE
----------------------------------------------------------------------

ML-Hyperspectral-TSS/
|-- Dataset/
|   |-- Hyperspectral Dataset/
|       |-- EMIT_Download_link.txt
|       |-- Prisma_Download_Link.txt
|       |-- PACE_Download_Link.txt
|
|-- Machine Learning Models/
|   |-- Models/
|       |-- Catboost_Model.pkl
|       |-- LightGBM_Model.pkl
|       |-- XgBoost_Model.pkl
|       |-- Random_Forest_Model.pkl
|       |-- PLSR_VisNIR_400_900_Model.pkl
|
|-- Scripts/
    |-- Train_Scripts/          (optional) training scripts (tabular -> .pkl)
    |-- Prisma_Scripts/         apply scripts for PRISMA rasters
    |-- EMIT_Scripts/           apply scripts for EMIT rasters
    |-- PACE_Scripts/           apply scripts for PACE products/rasters


NOTE:
Large datasets are not stored in this repository. See “DATA AVAILABILITY” below.


----------------------------------------------------------------------
INSTALLATION
----------------------------------------------------------------------

Create virtual environment and install requirements:

python -m venv .venv

Windows:
.venv\Scripts\activate

macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

Typical dependencies include:
numpy, pandas, rasterio, joblib
and the model libraries you plan to use:
catboost, lightgbm, xgboost, scikit-learn


----------------------------------------------------------------------
QUICK START (RECOMMENDED FLOW)
----------------------------------------------------------------------

STEP 1) TRAIN A MODEL (TABULAR -> .PKL)
Training scripts read your training table (samples x bands), fit a model, and save a .pkl file.

Example output:
Machine Learning Models/Models/Catboost_Model.pkl

Your training script may include:
- feature selection (e.g., top-10 wavelengths)
- hyperparameter tuning
- saving the final feature order used by the model


STEP 2) APPLY A PRETRAINED MODEL (RASTER -> GEOTIFF)
All apply scripts follow the same pattern (CatBoost / LightGBM / RF / XGBoost / PLSR) with sensor-specific handling.

What the apply script does (pixel-based):
1) Load pretrained model (joblib.load)
2) Map model wavelengths -> nearest sensor band indices
3) Read those raster bands
4) Apply reflectance scaling (example: PRISMA 0–10000 -> 0–1)
5) Flatten raster to pixels x features
6) Mask invalid pixels (NaN/Inf, and optionally non-water-like pixels)
7) Predict TSS for valid pixels
8) Reshape back to 2D and export single-band float32 GeoTIFF


----------------------------------------------------------------------
EXAMPLE: PRISMA APPLY (CATBOOST)
----------------------------------------------------------------------

Run:
python Scripts/Prisma_Scripts/apply_catboost_prisma.py

Inside the script you typically set:
- MODEL_PATH   -> path to .pkl
- PRISMA_TIF   -> input hyperspectral raster
- OUTPUT_DIR   -> output folder
- MODEL_FEATURES (ordered list of bands used during training)

IMPORTANT:
Model features MUST be in the exact order used in training.

Example feature list (order matters):
MODEL_FEATURES = [
  "X_634","X_647","X_422","X_584","X_482",
  "X_897","X_719","X_600","X_889","X_779"
]


----------------------------------------------------------------------
BAND MATCHING (WAVELENGTH ALIGNMENT)
----------------------------------------------------------------------

Model features are named like “X_634” meaning wavelength = 634 nm.
The script:
- extracts wavelength from the feature name
- finds the closest sensor wavelength
- reads that raster band index (Rasterio uses 1-based indexing)

Example approach used in PRISMA scripts:
- PRISMA VNIR wavelength array:
  PRISMA_VNIR_WL = linspace(400, 1010, 63)
- nearest-band match:
  closest_idx = argmin(abs(PRISMA_VNIR_WL - target_wl))
- band number for rasterio:
  band_num = closest_idx + 1


----------------------------------------------------------------------
REFLECTANCE SCALING
----------------------------------------------------------------------

Many hyperspectral products store reflectance as scaled integers.

PRISMA L2D common scaling:
- typical values: 0–10000
- convert to 0–1 using:
  SCALE_FACTOR = 0.0001

Some scripts apply scaling conditionally:
- if nanmax(band_arr) > 10 then band_arr *= 0.0001

This ensures the raster data matches the model training distribution.


----------------------------------------------------------------------
VALID PIXEL MASK (BASIC WATER/VALIDITY FILTER)
----------------------------------------------------------------------

Apply scripts usually predict only for pixels that are:
- finite values (no NaN/Inf)
- have some positive reflectance (not all zeros)

Example logic:
valid_mask = all(isfinite(pixel)) AND any(pixel > 0)

This reduces unnecessary prediction over empty/no-data areas and speeds up inference.

If you have a dedicated water mask, you can replace or extend this logic.


----------------------------------------------------------------------
OUTPUT
----------------------------------------------------------------------

- Single-band GeoTIFF (float32)
- CRS/transform/extent inherited from the input raster
- NoData exported as NaN (nodata = NaN)

Typical filename:
- PRISMA_TSS_Prediction_Map.tif
or
- MODEL_SENSOR_TSS_YYYYMMDD.tif


----------------------------------------------------------------------
APPLYING OTHER MODELS (LIGHTGBM / RF / XGBOOST / PLSR)
----------------------------------------------------------------------

You have equivalent apply scripts for:
- apply_lightgbm_*.py
- apply_rf_*.py
- apply_xgboost_*.py
- apply_plsr_*.py

They share the same structure:
- load .pkl
- map wavelengths to bands
- scale reflectance
- mask + predict + export GeoTIFF

You run them the same way, just with the correct script and model file for the sensor.


----------------------------------------------------------------------
DATA AVAILABILITY
----------------------------------------------------------------------

Large hyperspectral datasets are not stored in this repo.

- PRISMA (PRecursore IperSpettrale della Missione Applicativa):
  available via the Italian Space Agency (ASI) portal (registration required)

- EMIT (Earth Surface Mineral Dust Source Investigation):
  distributed by NASA via Earthdata / ORNL DAAC channels

- PACE (Plankton, Aerosol, Cloud, ocean Ecosystem):
  accessible via NASA OceanColor Web (and related NASA distribution portals)

See the text files under:
Dataset/Hyperspectral Dataset/


----------------------------------------------------------------------
TROUBLESHOOTING
----------------------------------------------------------------------

1) Output is all NaN / empty
- check scaling (0–10000 -> 0–1)
- check band mapping logs (wrong bands = wrong signal)
- relax the valid pixel mask if your raster has many zeros

2) Wrong-looking TSS range
- confirm model feature order matches training order exactly
- confirm correct sensor scaling
- confirm correct spectral subset (VNIR vs full range)

3) Band mapping mismatch
- ensure sensor wavelength definition matches the product you use
- if your raster contains band descriptions with wavelengths, consider reading wavelengths from metadata instead of assuming linspace


----------------------------------------------------------------------
LICENSE
----------------------------------------------------------------------

MIT License (see LICENSE).


----------------------------------------------------------------------
CONTACT
----------------------------------------------------------------------

Onur Karaca
GitHub: @onurkaraca87
Email: onurkaraca87@hotmail.com
Website: www.onurkaraca87.com
