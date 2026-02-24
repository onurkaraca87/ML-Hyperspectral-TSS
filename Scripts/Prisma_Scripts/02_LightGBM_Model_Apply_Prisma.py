# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Author:      Onur Karaca
Contact:     onurkaraca87@hotmail.com
Website:     www.onurkaraca87.com

-------------------------------------------------------------------------------
Project:     PRISMA Raster Prediction - LightGBM Pipeline
Description: 
    This script implements an inference pipeline for hyperspectral PRISMA L2D 
    imagery using a pre-trained LightGBM regression model. 
    
    Key Features:
    - Band Selection: Maps model-specific features to PRISMA VNIR bands.
    - Spectral Scaling: Calibrates PRISMA Digital Numbers to reflectance values.
    - Optimized Inference: Uses pandas DataFrame input to satisfy LightGBM 
      feature name requirements.
    - Spatial Export: Generates a georeferenced TSS (Total Suspended Solids) map.
-------------------------------------------------------------------------------
"""

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd
import rasterio

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# 1) CONFIGURATION & PATHS
# =============================================================================
# Path to the pre-trained LightGBM model (.pkl)
MODEL_PATH = r"path/to/your/lgbm_balanced_model_20260222.pkl"

# Input PRISMA Hyperspectral GeoTIFF
PRISMA_TIF = r"path/to/your/Prisma_geo_wm_correct_tif_renamed.tif"

# Output directory for predicted rasters
OUTPUT_DIR = r"path/to/your/output/Prisma_Results"

# LightGBM Top 10 Selected Spectral Bands
MODEL_BANDS = [
    'X_400', 'X_696', 'X_508', 'X_866', 'X_880', 
    'X_781', 'X_545', 'X_717', 'X_843', 'X_497'
]

# PRISMA VNIR Spectral range (63 bands: 400nm to 1010nm)
PRISMA_VNIR_WL = np.linspace(400, 1010, 63)



def main():
    # Load LightGBM Model
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at: {MODEL_PATH}")
        sys.exit()

    try:
        model = joblib.load(MODEL_PATH)
        logging.info("LightGBM model loaded successfully.")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit()

    # =========================================================================
    # 2) BAND ALIGNMENT & DATA INGESTION
    # =========================================================================
    

    with rasterio.open(PRISMA_TIF) as src:
        profile = src.profile.copy()
        
        band_indices = []
        logging.info("Mapping model features to PRISMA spectral bands...")
        
        for feat in MODEL_BANDS:
            # Extract target wavelength from feature name (e.g., 'X_400' -> 400.0)
            target_wl = float(feat.split('_')[1])
            
            # Identify the closest existing PRISMA band index
            idx = (np.abs(PRISMA_VNIR_WL - target_wl)).argmin()
            band_num = int(idx + 1) # Rasterio uses 1-based indexing
            
            band_indices.append(band_num)
            logging.info(f"{feat:>8} -> PRISMA Band #{band_num:2d} ({PRISMA_VNIR_WL[idx]:.2f} nm)")

        # Load and scale imagery
        bands_data = []
        for bnum in band_indices:
            arr = src.read(bnum).astype(np.float32)
            
            # PRISMA L2D data scaling: Convert 0-10000 range to 0.0-1.0 reflectance
            if np.nanmax(arr) > 10:
                arr *= 0.0001
            
            bands_data.append(arr)

        stacked = np.stack(bands_data, axis=-1)

    # =========================================================================
    # 3) SPATIAL PREDICTION (INFERENCE)
    # =========================================================================
    h, w, c = stacked.shape
    pixels_flat = stacked.reshape(-1, c)

    # Filter for valid water pixels (finite values and positive reflectance)
    valid_mask = np.all(np.isfinite(pixels_flat), axis=1) & (np.any(pixels_flat > 0, axis=1))

    logging.info(f"Processing {np.sum(valid_mask)} valid pixels via LightGBM...")

    # Initialize prediction array with NaNs for background
    predictions_flat = np.full((pixels_flat.shape[0],), np.nan, dtype=np.float32)

    if np.any(valid_mask):
        # LightGBM requires DataFrame with specific feature names to maintain consistency
        X_valid = pd.DataFrame(pixels_flat[valid_mask], columns=MODEL_BANDS)
        
        # Execute batch prediction
        preds = model.predict(X_valid)
        predictions_flat[valid_mask] = preds.astype(np.float32)

    # Reshape back to 2D raster dimensions
    tss_map = predictions_flat.reshape(h, w)

    # =========================================================================
    # 4) GEOTIFF EXPORT
    # =========================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = os.path.join(OUTPUT_DIR, "PRISMA_LightGBM_TSS_Prediction.tif")

    # Update metadata for single-band prediction output
    profile.update(
        dtype=rasterio.float32, 
        count=1, 
        compress="lzw", 
        nodata=np.nan
    )

    with rasterio.open(output_filename, "w", **profile) as dst:
        dst.write(tss_map, 1)

    logging.info(f"Process finalized. Predicted TSS map saved to:\n{output_filename}")

if __name__ == "__main__":
    main()
