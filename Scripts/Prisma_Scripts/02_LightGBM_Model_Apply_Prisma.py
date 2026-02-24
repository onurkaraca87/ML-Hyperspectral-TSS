# -*- coding: utf-8 -*-
"""
Apply trained LightGBM (10-band model) to PRISMA raster
and save prediction raster to the model results folder.

Author: sokaraca
Date: 2025-11-14
"""

import os
import numpy as np
import rasterio
import joblib

# ============================================================
# 1) Load trained LightGBM model
# ============================================================
model_dir = r".....\lightGBM"
model_path = os.path.join(model_dir, "LightGBM_Model.pkl")

model = joblib.load(model_path)
print(f"✅ LightGBM model loaded: {model_path}")

# ============================================================
# 2) PRISMA raster
# ============================================================
prisma_tif = r"...........\Prisma_20250310.tif"

# ============================================================
# 3) The 10 selected wavelengths (from your importance figure)
# ============================================================
model_bands = [
    "X_400",
    "X_696",
    "X_781",
    "X_508",
    "X_843",
    "X_545",
    "X_497",
    "X_717",
    "X_880",
    "X_401"
]

# ============================================================
# 4) Map PRISMA actual band indices (you MUST update these!)
#    PRISMA Band 1 = index 1
# ============================================================

band_indices = {
    "X_400": 1,   # example
    "X_696": 36,   # example
    "X_781": 45,    # example
    "X_508": 15,   # example
    "X_843": 51,   # example
    "X_545": 19,   # example
    "X_497": 13,   # example
    "X_717": 38,   # example
    "X_880": 54,    # example
    "X_401": 1    # example
}

# ============================================================
# 5) Read raster & extract selected bands
# ============================================================
with rasterio.open(prisma_tif) as src:
    profile = src.profile.copy()
    height, width = src.height, src.width

    print(f"🚀 Raster size: {width} x {height}")
    print(f"📏 Total raster bands available: {src.count}")

    bands_data = []
    for band_name in model_bands:
        if band_name not in band_indices:
            raise ValueError(f"❌ No band index for {band_name} in band_indices!")

        band_num = band_indices[band_name]
        arr = src.read(band_num).astype(np.float32)
        bands_data.append(arr)

    stacked = np.stack(bands_data, axis=0).transpose(1, 2, 0)
    print(f"✅ Stacked shape: {stacked.shape} (H, W, 10)")

# ============================================================
# 6) Predict per-pixel
# ============================================================
pixels_flat = stacked.reshape(-1, len(model_bands))
valid_mask = np.all(np.isfinite(pixels_flat), axis=1)

print(f"Valid pixels: {valid_mask.sum()} / {pixels_flat.shape[0]}")

preds = np.full((pixels_flat.shape[0],), np.nan, dtype=np.float32)
preds[valid_mask] = model.predict(pixels_flat[valid_mask])

tss_raster = preds.reshape(height, width)
print(f"✅ Prediction raster ready: {tss_raster.shape}")

# ============================================================
# 7) Save output raster
# ============================================================
output_dir = os.path.join(model_dir, "Model_sonuclari")
os.makedirs(output_dir, exist_ok=True)

output_tif = os.path.join(
    output_dir,
    "PRISMA_20250310_LightGBM_TSS_Prediction_10bands.tif"
)

profile.update(dtype=rasterio.float32, count=1, compress="lzw")

with rasterio.open(output_tif, "w", **profile) as dst:
    dst.write(tss_raster, 1)

print(f"🎉 TSS prediction saved to:\n{output_tif}")
