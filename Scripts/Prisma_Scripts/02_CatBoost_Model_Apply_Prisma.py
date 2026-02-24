# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Author:      Onur Karaca
Contact:     onurkaraca87@hotmail.com
Website:     www.onurkaraca87.com

-------------------------------------------------------------------------------
Project:     PRISMA Raster Prediction - TSS Mapping
Description: 
    This production-ready script applies a trained CatBoost regression model 
    to PRISMA L2D hyperspectral satellite imagery. 
    
    Key Features:
    - Automated Band Alignment: Matches model features to PRISMA VNIR bands.
    - Reflectance Scaling: Converts 0-10000 DN values to 0-1 reflectance.
    - Masked Inference: Optimized for water pixel processing.
-------------------------------------------------------------------------------
"""

import os
import logging
import joblib
import numpy as np
import rasterio

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# 1) CONFIGURATION & PATHS
# =============================================================================
# Path to your trained .pkl model
MODEL_PATH = r"path/to/your/catboost_model_20260222.pkl"

# Input PRISMA GeoTIFF file
PRISMA_TIF = r"path/to/your/Prisma_geo_wm_correct_tif_renamed.tif"

# Output directory for the prediction map
OUTPUT_DIR = r"path/to/your/output/Prisma_Results"

# Critical: Ordered list of bands used during model training
MODEL_FEATURES = ['X_634', 'X_647', 'X_422', 'X_584', 'X_482', 
                  'X_897', 'X_719', 'X_600', 'X_889', 'X_779']

# PRISMA VNIR wavelength definition (Standard 63 bands: 400nm to 1010nm)
PRISMA_VNIR_WL = np.linspace(400, 1010, 63)



def main():
    # Load model
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found at: {MODEL_PATH}")
        return
    
    model = joblib.load(MODEL_PATH)
    logging.info("CatBoost model successfully loaded.")

    # =========================================================================
    # 2) SPECTRAL BAND MAPPING
    # =========================================================================
    with rasterio.open(PRISMA_TIF) as src:
        profile = src.profile.copy()
        
        band_indices = []
        logging.info("Starting manual band mapping...")
        
        for feat in MODEL_FEATURES:
            # Extract target wavelength from feature name (e.g., 'X_634' -> 634.0)
            target_wl = float(feat.split('_')[1])
            
            # Find index of the closest PRISMA band (0-indexed)
            closest_idx = (np.abs(PRISMA_VNIR_WL - target_wl)).argmin()
            band_num = int(closest_idx + 1) # Rasterio uses 1-based indexing
            
            band_indices.append(band_num)
            logging.info(f"Feature {feat} matched to PRISMA Band #{band_num} ({PRISMA_VNIR_WL[closest_idx]:.2f} nm)")

        # Read and scale spectral bands
        bands_data = []
        for bnum in band_indices:
            band_arr = src.read(bnum).astype(np.float32)
            
            # Scaling logic: PRISMA L2D data is typically scaled by 10,000. 
            # We convert it back to 0.0 - 1.0 range as expected by the model.
            if np.nanmax(band_arr) > 10:
                band_arr *= 0.0001
            
            bands_data.append(band_arr)

        # Reshape to (Height, Width, Features)
        stacked_data = np.stack(bands_data, axis=-1)

    # =========================================================================
    # 3) PIXEL-WISE PREDICTION
    # =========================================================================
    h, w, c = stacked_data.shape
    pixels_flat = stacked_data.reshape(-1, c)

    # Optimization: Process only valid pixels (finite values and positive reflectance)
    # This acts as a basic water/land mask based on spectral presence.
    valid_mask = np.all(np.isfinite(pixels_flat), axis=1) & (np.any(pixels_flat > 0, axis=1))

    logging.info(f"Executing prediction for {np.sum(valid_mask)} valid pixels...")

    # Initialize prediction array with NaNs
    final_preds = np.full((pixels_flat.shape[0],), np.nan, dtype=np.float32)

    if np.any(valid_mask):
        # Run CatBoost Inference
        predictions = model.predict(pixels_flat[valid_mask])
        final_preds[valid_mask] = predictions.astype(np.float32)

    # Reshape back to 2D spatial map
    tss_prediction_map = final_preds.reshape(h, w)

    # =========================================================================
    # 4) GEOTIFF EXPORT
    # =========================================================================
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "PRISMA_TSS_Prediction_Map.tif")

    # Update metadata for output
    profile.update(
        dtype=rasterio.float32, 
        count=1, 
        compress="lzw", 
        nodata=np.nan
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(tss_prediction_map, 1)

    logging.info(f"Process complete. Output saved to:\n{output_path}")

if __name__ == "__main__":
    main()
