# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Author:      Onur Karaca
Contact:     onurkaraca87@hotmail.com
Website:     www.onurkaraca87.com
-------------------------------------------------------------------------------
Project:     PRISMA Raster Prediction - TSS Mapping
Description: 
    This script performs pixel-based Total Suspended Solids (TSS) prediction 
    using a pre-trained CatBoost model and PRISMA hyperspectral L2D imagery.
    It handles band matching, scaling, and spatial export to GeoTIFF.
-------------------------------------------------------------------------------
"""

import os
import logging
import joblib
import numpy as np
import rasterio

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# CONFIGURATION & GENERIC PATHS
# =============================================================================
# Update these placeholders with your specific file paths
MODEL_PATH = r"path/to/your/models/catboost_model.pkl"
INPUT_TIF = r"path/to/your/data/prisma_input_image.tif"
OUTPUT_DIR = r"path/to/your/results"
OUTPUT_FILENAME = "PRISMA_TSS_Map_Output.tif"

# Model expects specific bands in this exact order (Wavelengths in nm)
MODEL_FEATURE_NAMES = [
    'X_634', 'X_647', 'X_422', 'X_584', 'X_482', 
    'X_897', 'X_719', 'X_600', 'X_889', 'X_779'
]

# PRISMA VNIR Wavelength Definition (Standard 63 bands from 400nm to 1010nm)
PRISMA_VNIR_WL = np.linspace(400, 1010, 63)

# Scaling Factor: Convert PRISMA L2D (0-10000) to Surface Reflectance (0-1)
SCALE_FACTOR = 0.0001

def map_prisma_bands(target_features, sensor_wavelengths):
    """
    Maps requested model feature names to the nearest PRISMA band indices.
    
    Args:
        target_features (list): Names of bands required by the model (e.g., 'X_634').
        sensor_wavelengths (ndarray): Center wavelengths of the PRISMA sensor.
        
    Returns:
        list: 1-based band indices for Rasterio.
    """
    indices = []
    logging.info("Initializing Band Mapping...")
    for feature in target_features:
        target_wl = float(feature.split('_')[1])
        # Find index of nearest wavelength in sensor array
        idx = (np.abs(sensor_wavelengths - target_wl)).argmin()
        band_no = int(idx + 1) # Rasterio uses 1-based indexing
        indices.append(band_no)
        logging.info(f"  Mapping {feature:>5} -> PRISMA Band #{band_no:2d} ({sensor_wavelengths[idx]:.2f} nm)")
    return indices

def run_prediction():
    """
    Executes the full prediction workflow:
    1. Load model 2. Extract bands 3. Scale data 4. Predict 5. Export GeoTIFF
    """
    
    # 1. Load Pre-trained Machine Learning Model
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model file not found. Please check path: {MODEL_PATH}")
        return

    logging.info("Loading CatBoost model...")
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return

    # 2. Read and Prepare Raster Data
    logging.info(f"Opening PRISMA hyperspectral image...")
    try:
        with rasterio.open(INPUT_TIF) as src:
            raster_profile = src.profile.copy()
            target_indices = map_prisma_bands(MODEL_FEATURE_NAMES, PRISMA_VNIR_WL)
            
            bands_list = []
            for b_idx in target_indices:
                # Apply constant scaling to match model training distribution (0-1)
                band_array = src.read(b_idx).astype(np.float32) * SCALE_FACTOR
                bands_list.append(band_array)
            
            data_stack = np.stack(bands_list, axis=-1)
            height, width, channels = data_stack.shape
    except Exception as e:
        logging.error(f"Error reading raster file: {e}")
        return

    # 3. Data Flattening and Water Masking
    pixels = data_stack.reshape(-1, channels)
    
    # Identify valid water pixels: Non-finite values and non-zero reflectance
    valid_mask = np.all(np.isfinite(pixels), axis=1) & (np.any(pixels > 0, axis=1))
    
    num_valid = np.sum(valid_mask)
    logging.info(f"Processing {num_valid} valid water pixels for TSS prediction...")

    # 4. Model Inference
    # Initialize output array with NaNs (NoData)
    predictions = np.full((pixels.shape[0],), np.nan, dtype=np.float32)

    if num_valid > 0:
        predictions[valid_mask] = model.predict(pixels[valid_mask]).astype(np.float32)
    else:
        logging.warning("No valid pixels identified for the current scene.")

    # Reshape prediction flat array back to 2D spatial dimensions
    tss_map = predictions.reshape(height, width)

    # 5. Export Result to GeoTIFF
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    
    # Update profile for single-band float output
    raster_profile.update(
        dtype=rasterio.float32, 
        count=1, 
        compress="lzw", 
        nodata=np.nan
    )

    logging.info(f"Exporting result to {output_path}...")
    with rasterio.open(output_path, "w", **raster_profile) as dst:
        dst.write(tss_map, 1)

    logging.info("Process completed successfully!")

if __name__ == "__main__":
    run_prediction()