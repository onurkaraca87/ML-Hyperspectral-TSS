# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Author:      Onur Karaca
Contact:     onurkaraca87@hotmail.com
Website:     www.onurkaraca87.com

-------------------------------------------------------------------------------
Project:     PRISMA Raster Prediction - XGBoost Pipeline
Description: 
    This script implements an automated spatial inference pipeline for PRISMA 
    L2D hyperspectral imagery using a pre-trained XGBoost regressor.
    
    Key Features:
    - Spectral Alignment: Maps model feature names (wavelengths) to the closest 
      available bands in the PRISMA VNIR sensor.
    - Data Scaling: Converts PRISMA Digital Numbers (0-10000) to the reflectance 
      scale (0-1) required for machine learning models.
    - Masked Processing: Optimized to process only valid water pixels, 
      reducing computational overhead.
    - Export Stability: Generates a georeferenced GeoTIFF with LZW compression.
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
# 1) CONFIGURATION & MODEL LOADING
# =============================================================================
# Update these paths to match your project directory
MODEL_PATH = r"path/to/your/xgboost_balanced_model_20260222.pkl"
PRISMA_TIF = r"path/to/your/Prisma_geo_wm_correct_tif_renamed.tif"
OUTPUT_DIR = r"path/to/your/output/Prisma_Results"

# Selected Top 10 Bands (Must match the exact feature names used in training)
MODEL_BANDS = [
    'X_625', 'X_495', 'X_503', 'X_489', 'X_598', 
    'X_486', 'X_499', 'X_508', 'X_422', 'X_419'
]

# Standard PRISMA VNIR spectral range (63 bands: 400nm to 1010nm)
PRISMA_VNIR_WL = np.linspace(400, 1010, 63)



def main():
    # Verify model existence
    if not os.path.exists(MODEL_PATH):
        logging.error(f"XGBoost model file not found at: {MODEL_PATH}")
        sys.exit()

    # Load pre-trained model
    try:
        model = joblib.load(MODEL_PATH)
        logging.info("XGBoost model successfully loaded.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        sys.exit()

    # =========================================================================
    # 2) BAND MAPPING & DATA INGESTION
    # =========================================================================
    

    with rasterio.open(PRISMA_TIF) as src:
        profile = src.profile.copy()
        
        band_indices = []
        logging.info("=== Executing Spectral Band Alignment ===")
        
        for feat in MODEL_BANDS:
            # Extract target wavelength from string (e.g., 'X_625' -> 625.0)
            target_wl = float(feat.split('_')[1])
            
            # Identify the closest PRISMA band index
            idx = (np.abs(PRISMA_VNIR_WL - target_wl)).argmin()
            band_num = int(idx + 1) # Rasterio uses 1-based indexing
            
            band_indices.append(band_num)
            logging.info(f"{feat:>8} -> PRISMA Band #{band_num:2d} ({PRISMA_VNIR_WL[idx]:.2f} nm)")

        # Read and scale spectral data
        bands_data = []
        for bnum in band_indices:
            arr = src.read(bnum).astype(np.float32)
            
            # Reflectance Calibration: PRISMA L2D data is typically scaled by 10,000
            if np.nanmax(arr) > 10:
                arr *= 0.0001
            bands_data.append(arr)

        stacked_cube = np.stack(bands_data, axis=-1)

    # =========================================================================
    # 3) PIXEL-BASED INFERENCE (PREDICTION)
    # =========================================================================
    h, w, c = stacked_cube.shape
    pixels_flat = stacked_cube.reshape(-1, c)

    # Filter for valid pixels (non-NaN and positive reflectance values)
    valid_mask = np.all(np.isfinite(pixels_flat), axis=1) & (np.any(pixels_flat > 0, axis=1))

    # Initialize results array with NaNs
    predictions_flat = np.full((pixels_flat.shape[0],), np.nan, dtype=np.float32)

    if np.any(valid_mask):
        logging.info(f"Processing {np.sum(valid_mask)} valid water pixels...")
        
        # XGBoost requires feature names to match training data exactly
        X_valid = pd.DataFrame(pixels_flat[valid_mask], columns=MODEL_BANDS)
        
        # Execute prediction
        preds_valid = model.predict(X_valid).astype(np.float32)
        predictions_flat[valid_mask] = preds_valid
    else:
        logging.warning("No valid water pixels detected in the input raster!")

    # Reshape prediction array back to 2D spatial dimensions
    tss_map = predictions_flat.reshape(h, w)

    # =========================================================================
    # 4) GEOTIFF EXPORT
    # =========================================================================
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, "PRISMA_XGBoost_TSS_Prediction.tif")

    # Update metadata for single-band output
    profile.update(
        dtype=rasterio.float32, 
        count=1, 
        compress="lzw", 
        nodata=np.nan
    )

    with rasterio.open(output_filename, "w", **profile) as dst:
        dst.write(tss_map, 1)

    logging.info(f"Success! Prediction map generated at:\n{output_filename}")

if __name__ == "__main__":
    main()
