# -*- coding: utf-8 -*-
"""
Apply trained CatBoost (10 bands) to PRISMA raster
and save prediction raster to output folder.

Created on Mon Nov  3 13:44:15 2025
@author: sokaraca
"""

import os
import numpy as np
import rasterio
import joblib

# ============================================
# 1) Load the trained CatBoost model
# ============================================

model_dir = r"D:\TWDB_5\Machine_Learning_Process\CatBoost\Prisma\Prisma_Output\Catboost_10band"
model_path = os.path.join(model_dir, "catboost_10bands_model_20251218.pkl")

model = joblib.load(model_path)
print(f"✅ CatBoost model loaded: {model_path}")

# ============================================
# 2) PRISMA raster path
# ============================================

prisma_tif = r"D:\TWDB_5\Machine_Learning_Process\Raw_Datasets\Prisma\Prisma_20250310\Prisma_geo_wm_correct_tif_renamed.tif"

# ============================================
# 3) Model input bands (10 bands)
#    ⚠️ IMPORTANT: Order must match training!
# ============================================

model_bands = [
    'X_430', 'X_611', 'X_582', 'X_735', 'X_797', 'X_620', 'X_586', 'X_407', 'X_664', 'X_870'
]

# ============================================
# 4) Band index mapping for PRISMA raster
#    ⚠️ UPDATE THESE NUMBERS based on your band map
#    (Example numbers shown below)
# ============================================

band_indices = {
    'X_430': 4,
    'X_611': 27,
    'X_582': 24,
    'X_735': 40,
    'X_797': 46,
    'X_620': 28,
    'X_586': 24,
    'X_407': 1,
    'X_664': 33,
    'X_870': 53
}

# ============================================
# 5) Read PRISMA raster and extract 10 bands
# ============================================

with rasterio.open(prisma_tif) as src:
    profile = src.profile.copy()
    height = src.height
    width = src.width

    print(f"📏 Raster size: {width} x {height}")
    print(f"📦 Total bands in PRISMA file: {src.count}")

    # Read selected bands and stack
    bands_data = []

    for band_name in model_bands:
        band_num = band_indices.get(band_name, None)
        if band_num is None:
            raise ValueError(f"❌ Missing band index for {band_name}!")

        print(f"→ Reading {band_name} (band #{band_num})")
        arr = src.read(band_num).astype(np.float32)
        bands_data.append(arr)

    # Convert to (H, W, C)
    stacked = np.stack(bands_data, axis=0).transpose(1, 2, 0)
    print(f"✅ Stacked bands shape: {stacked.shape}  (H, W, 10)")

# ============================================
# 6) Prepare data for CatBoost prediction
# ============================================

pixels_flat = stacked.reshape(-1, len(model_bands))    # (N, 10)
valid_mask = np.all(np.isfinite(pixels_flat), axis=1)
valid_pixels = pixels_flat[valid_mask]

print(f"✔️ Valid pixels: {valid_pixels.shape[0]} / {pixels_flat.shape[0]}")

# Predict TSS
preds = np.full((pixels_flat.shape[0],), np.nan, dtype=np.float32)
preds[valid_mask] = model.predict(valid_pixels)

# Back to raster shape
tss_raster = preds.reshape(height, width)
print(f"✅ Prediction raster generated: {tss_raster.shape}")

# ============================================
# 7) Save output raster
# ============================================

output_dir = os.path.join(model_dir, "Model_sonuclari")
os.makedirs(output_dir, exist_ok=True)

output_tif = os.path.join(
    output_dir,
    "PRISMA_20250310_CatBoost_TSS_10bands_NEW.tif"
)

profile.update(
    dtype=rasterio.float32,
    count=1,
    compress="lzw"
)

with rasterio.open(output_tif, 'w', **profile) as dst:
    dst.write(tss_raster, 1)

print(f"🎉 TSS prediction saved to:\n{output_tif}")
