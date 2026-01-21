# -*- coding: utf-8 -*-
"""
Apply a trained XGBoost model (10 spectral bands) to a PRISMA raster (GeoTIFF)
and save the prediction raster to the model results folder.

Author: sokaraca
Created: 2025-11-03
"""

import os
import numpy as np
import rasterio
import joblib

# ========================================
# 1) Set model path and load the XGBoost model
# ========================================
# 🔴 UPDATE THIS to your XGBoost output folder:
model_dir = r".......\XGBoost"

# 🔴 UPDATE THIS to your XGBoost model filename:
# The training script saved models like: xgb_{N_SELECTED}bands_model_{tag}.pkl
model_path = os.path.join(model_dir, "XGBoost_Model.pkl")

model = joblib.load(model_path)
print(f"✅ XGBoost model loaded: {model_path}")

# ========================================
# 2) PRISMA raster file (GeoTIFF)
# ========================================
# 🔴 UPDATE THIS to your PRISMA raster path:
prisma_tif = r".......\Prisma_20250310.tif"

# ========================================
# 3) Bands used by the model (10 bands)
#    -> Fill this based on the 'Selected bands: [...]' line from XGBoost training
# ========================================
# Example note (from importance figure / top 10):
# X_403, X_457, X_416, X_589, X_476, X_400, X_467, X_460, X_618, X_451
model_bands = [
    "X_440",
    "X_791",
    "X_472",
    "X_457",
    "X_495",
    "X_463",
    "X_400",
    "X_449",
    "X_467",
    "X_406",
]

# 🔴 UPDATE THIS according to your PRISMA band mapping:
# Example indices: 'X_400' → band 1, etc.
# IMPORTANT: Rasterio reads bands using 1-based indexing (band 1..N).
band_indices = {
    "X_440": 5,   # example
    "X_791": 46,  # example
    "X_472": 10,  # example
    "X_457": 7,   # example
    "X_495": 13,  # example
    "X_463": 8,   # example (replace with the real band number)
    "X_400": 1,   # example
    "X_449": 7,   # example
    "X_467": 9,   # example
    "X_406": 1,   # example
}

# ========================================
# 4) Read the raster and stack selected bands
# ========================================
with rasterio.open(prisma_tif) as src:
    profile = src.profile.copy()
    height = src.height
    width = src.width

    print(f"🚀 Raster dimensions: {width} x {height}")
    print(f"📏 Total raster bands: {src.count}")

    bands_data = []
    for band_name in model_bands:
        if band_name not in band_indices:
            raise ValueError(f"⚠️ No entry for {band_name} in band_indices!")
        band_num = band_indices[band_name]
        if band_num is None:
            raise ValueError(f"⚠️ Band index for {band_name} is not set in band_indices!")
        band_array = src.read(band_num).astype(np.float32)
        bands_data.append(band_array)

    # shape: (H, W, C) -> (height, width, 10 bands)
    stacked = np.stack(bands_data, axis=0).transpose(1, 2, 0)
    print(f"✅ Stacked bands shape (H, W, C): {stacked.shape}")

# ========================================
# 5) Flatten and mask for prediction
# ========================================
pixels_flat = stacked.reshape(-1, len(model_bands))  # (N_pixels, 10)
valid_mask = np.all(np.isfinite(pixels_flat), axis=1)
valid_pixels = pixels_flat[valid_mask]

print(f"Valid pixels: {valid_pixels.shape[0]} / {pixels_flat.shape[0]}")

# XGBoost prediction (model.predict is scikit-learn compatible)
preds = np.full((pixels_flat.shape[0],), np.nan, dtype=np.float32)
preds[valid_mask] = model.predict(valid_pixels)

ssc_raster = preds.reshape(height, width)
print(f"✅ Prediction raster shape: {ssc_raster.shape}")

# ========================================
# 6) Save the prediction raster
# ========================================
output_dir = os.path.join(model_dir, "Model_results")
os.makedirs(output_dir, exist_ok=True)

output_tif = os.path.join(
    output_dir,
    "PRISMA_20250310_XGB_SSC_Prediction_10bands_max232.tif",
)

profile.update(
    dtype=rasterio.float32,
    count=1,
    compress="lzw",
)

with rasterio.open(output_tif, "w", **profile) as dst:
    dst.write(ssc_raster, 1)

print(f"🎉 SSC prediction saved to:\n{output_tif}")
