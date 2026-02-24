# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Author:      Onur Karaca
Contact:     onurkaraca87@hotmail.com
Website:     www.onurkaraca87.com

-------------------------------------------------------------------------------
Project:     PRISMA Raster Prediction - Random Forest Pipeline (Top-10 Bands)
Description: 
    This script performs spatial inference on PRISMA hyperspectral L2D imagery 
    using a pre-trained Random Forest Regressor. 
    
    Key Features:
    - Specific Feature Mapping: Uses exactly the Top 10 bands identified 
      during feature selection.
    - Band Alignment: Matches model feature names (wavelengths) to the 
      nearest available PRISMA VNIR bands.
    - Reflectance Calibration: Automatically scales PRISMA DN (0-10000) 
      to reflectance (0-1) values.
    - Robust Validation: Includes error handling for feature name mismatches.
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
# Path to the pre-trained Random Forest model
MODEL_PATH = r"path/to/your/rf_model_20260222.pkl"

# Input PRISMA GeoTIFF (Hyperspectral cube)
PRISMA_TIF = r"path/to/your/Prisma_geo_wm_correct_tif_renamed.tif"

# Output directory for prediction maps
OUTPUT_DIR = r"path/to/your/output/Prisma_Results"

# Top 10 Selected Bands (Must match the exact feature names used in training)
TOP_10_FEATURES = [
    'X_630', 'X_677', 'X_488', 'X_522', 'X_684', 
    'X_625', 'X_819', 'X_489', 'X_500', 'X_504'
]

# PRISMA VNIR spectral range (Standard 63 bands: 400nm to 1010nm)
PRISMA_VNIR_WL = np.linspace(400, 1010, 63)



def main():
    # Load Random Forest Model
    if not os.path.exists(MODEL_PATH):
        logging.error("Model file not found! Please check the path.")
        sys.exit()

    model = joblib.load(MODEL_PATH)
    logging.info("Random Forest model successfully loaded.")

    # =========================================================================
    # 2) SPECTRAL ALIGNMENT & DATA READING
    # =========================================================================
    

    with rasterio.open(PRISMA_TIF) as src:
        profile = src.profile.copy()
        
        band_indices = []
        logging.info("=== Mapping Selected Features to PRISMA Bands ===")
        
        for feat in TOP_10_FEATURES:
            # Extract wavelength from string (e.g., 'X_630' -> 630.0)
            target_wl = float(feat.split('_')[1])
            
            # Find the nearest wavelength index in PRISMA VNIR
            idx = (np.abs(PRISMA_VNIR_WL - target_wl)).argmin()
            band_num = int(idx + 1) # Rasterio uses 1-based indexing
            
            band_indices.append(band_num)
            logging.info(f"{feat:>8} -> PRISMA Band #{band_num:2d} ({PRISMA_VNIR_WL[idx]:.2f} nm)")

        # Read spectral bands and scale to 0-1 range
        bands_list = []
        for bnum in band_indices:
            arr = src.read(int(bnum)).astype(np.float32)
            
            # Calibration: DN to Reflectance transformation
            if np.nanmax(arr) > 10:
                arr *= 0.0001
            bands_list.append(arr)

        stacked_cube = np.stack(bands_list, axis=-1)

    # =========================================================================
    # 3) PIXEL-WISE PREDICTION
    # =========================================================================
    h, w, c = stacked_cube.shape
    pixels_flat = stacked_cube.reshape(-1, c)
    
    # Generate mask for valid data (non-NaN and positive reflectance)
    valid_mask = np.all(np.isfinite(pixels_flat), axis=1) & (np.any(pixels_flat > 0, axis=1))

    logging.info(f"Processing {np.sum(valid_mask)} valid pixels...")

    # Initialize prediction array with NaNs for background
    predictions_flat = np.full((pixels_flat.shape[0],), np.nan, dtype=np.float32)

    if np.any(valid_mask):
        # Create DataFrame to maintain feature name consistency for Random Forest
        X_valid = pd.DataFrame(pixels_flat[valid_mask], columns=TOP_10_FEATURES)
        
        try:
            preds = model.predict(X_valid)
            predictions_flat[valid_mask] = preds.astype(np.float32)
        except ValueError as e:
            logging.error("FEATURE MISMATCH ERROR!")
            logging.info("The model may be expecting more features than the 10 provided.")
            logging.info("Verify if the model was re-fitted specifically with these 10 bands.")
            logging.error(f"Details: {e}")
            sys.exit()

    # Reshape prediction array back to 2D raster dimensions
    tss_map = predictions_flat.reshape(h, w)

    # =========================================================================
    # 4) GEOTIFF EXPORT
    # =========================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_filename = os.path.join(OUTPUT_DIR, "PRISMA_RF_TSS_Prediction.tif")

    # Update metadata for single-band output
    profile.update(
        dtype=rasterio.float32, 
        count=1, 
        compress="lzw", 
        nodata=np.nan
    )

    with rasterio.open(output_filename, "w", **profile) as dst:
        dst.write(tss_map, 1)

    logging.info(f"Success! Prediction map generated at: {output_filename}")

if __name__ == "__main__":
    main()
