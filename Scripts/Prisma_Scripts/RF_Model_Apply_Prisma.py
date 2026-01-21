# -*- coding: utf-8 -*-
"""
Apply a trained Random Forest model (10 spectral bands) to a PRISMA GeoTIFF and
save the prediction raster to disk.

What this script does
---------------------
1) Loads a scikit-learn compatible Random Forest model from a .pkl file
2) Reads the required PRISMA bands (based on your feature list)
3) Stacks bands into a (H, W, C) array
4) Flattens pixels, masks invalid values, and runs model prediction
5) Writes the predicted raster as a single-band GeoTIFF (LZW compressed)

IMPORTANT
---------
- Update BAND_INDICES with the correct 1-based PRISMA band numbers for each feature
  (Rasterio reads bands using 1-based indexing: band 1..N).
- Ensure the band order in MODEL_BANDS matches the order used during training.

Author: Onur Karaca
Created: 2025-11-03
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import rasterio
import joblib


# =============================================================================
# 1) CONFIGURATION (UPDATED FOR: D:\TWDB_5\TEST)
# =============================================================================

# Base folder shown in your screenshot
BASE_DIR = Path(r".....\Random_Forest")

# Trained model file (in BASE_DIR)
MODEL_PATH = BASE_DIR / "Random_Forest_Model.pkl"

# PRISMA GeoTIFF file (in BASE_DIR)
PRISMA_TIF = BASE_DIR / "Prisma_20250310.tif"

# Output settings
OUTPUT_DIR = BASE_DIR / "Model_sonuclari"  # change to BASE_DIR if you want output directly in TEST
OUTPUT_TIF_NAME = "PRISMA_20250310_RF_TSS_Prediction.tif"


# =============================================================================
# 2) MODEL FEATURES (10 BANDS)
# =============================================================================

# The 10 feature/band names used during training (must match training feature names)
MODEL_BANDS = [
    "X_630",
    "X_625",
    "X_635",
    "X_615",
    "X_495",
    "X_684",
    "X_638",
    "X_585",
    "X_606",
    "X_485",
]

# Map each feature name to the PRISMA GeoTIFF band number (1-based).
# Replace the example values with your true PRISMA band indices.
BAND_INDICES = {
    "X_630": 30,  # example
    "X_625": 29,  # example
    "X_635": 31,  # example
    "X_615": 28,  # example
    "X_495": 13,  # example
    "X_684": 35,  # example
    "X_638": 32,  # example
    "X_585": 24,  # example
    "X_606": 27,  # example
    "X_485": 11,  # example
}


# =============================================================================
# 3) VALIDATION HELPERS
# =============================================================================

def validate_inputs() -> None:
    """Validate that input files and band mapping are consistent."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found:\n  {MODEL_PATH}")

    if not PRISMA_TIF.exists():
        raise FileNotFoundError(f"PRISMA GeoTIFF not found:\n  {PRISMA_TIF}")

    missing = [b for b in MODEL_BANDS if b not in BAND_INDICES]
    if missing:
        raise ValueError(f"Missing BAND_INDICES entries for: {missing}")

    invalid = {k: v for k, v in BAND_INDICES.items() if not isinstance(v, int) or v < 1}
    if invalid:
        raise ValueError(f"Invalid band indices (must be positive integers): {invalid}")


# =============================================================================
# 4) MAIN WORKFLOW
# =============================================================================

def main() -> None:
    validate_inputs()

    # --- Load the trained model
    model = joblib.load(MODEL_PATH)
    print(f"✅ Random Forest model loaded:\n  {MODEL_PATH}")

    # --- Read PRISMA bands and stack
    with rasterio.open(PRISMA_TIF) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width

        print(f"\n🛰️ Input raster:\n  {PRISMA_TIF}")
        print(f"📐 Raster dimensions (W x H): {width} x {height}")
        print(f"📦 Total bands in file: {src.count}")

        band_arrays = []
        for band_name in MODEL_BANDS:
            band_num = BAND_INDICES[band_name]

            if band_num > src.count:
                raise ValueError(
                    f"Band index out of range for {band_name}: {band_num} "
                    f"(GeoTIFF has {src.count} bands)"
                )

            print(f"→ Reading {band_name} (band #{band_num})")
            arr = src.read(band_num).astype(np.float32)
            band_arrays.append(arr)

        # Stack to shape (H, W, C)
        stacked = np.stack(band_arrays, axis=0).transpose(1, 2, 0)
        print(f"✅ Stacked shape (H, W, C): {stacked.shape}")

    # --- Flatten for prediction
    pixels_flat = stacked.reshape(-1, len(MODEL_BANDS))  # (N_pixels, 10)

    # Mask valid pixels (finite values across all bands)
    valid_mask = np.all(np.isfinite(pixels_flat), axis=1)
    valid_pixels = pixels_flat[valid_mask]

    print(f"\n🔎 Valid pixels: {valid_pixels.shape[0]} / {pixels_flat.shape[0]}")

    # --- Predict
    preds = np.full((pixels_flat.shape[0],), np.nan, dtype=np.float32)
    preds[valid_mask] = model.predict(valid_pixels).astype(np.float32)

    prediction_raster = preds.reshape(height, width)
    print(f"✅ Prediction raster shape: {prediction_raster.shape}")

    # --- Write output GeoTIFF
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_tif = OUTPUT_DIR / OUTPUT_TIF_NAME

    profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        # nodata=np.nan,  # optional; some GIS tools do not love NaN nodata
    )

    with rasterio.open(output_tif, "w", **profile) as dst:
        dst.write(prediction_raster, 1)

    print(f"\n🎉 Prediction saved to:\n  {output_tif}")


if __name__ == "__main__":
    main()
