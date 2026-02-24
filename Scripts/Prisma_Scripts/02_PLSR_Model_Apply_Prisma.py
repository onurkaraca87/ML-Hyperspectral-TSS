# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Author:      Onur Karaca
Contact:    onurkaraca87@hotmail.com
Website:     www.onurkaraca87.com

-------------------------------------------------------------------------------
Project:     PRISMA Raster Prediction - PLSR (Partial Least Squares Regression)
Description: 
    This production script implements a high-performance, windowed inference 
    pipeline for PRISMA L2D hyperspectral data using PLSR.
    
    Key Features:
    - Flexible Bundle Loader: Automatically extracts model objects, VIP tables, 
      and log-transform settings from joblib dictionaries.
    - Spectral Matching: Maps model-trained wavelengths to actual PRISMA 
      sensor center wavelengths.
    - Windowed Processing: Efficiently handles large rasters by processing 
      data in 512x512 blocks to optimize memory usage.
    - Inverse Log-Calibration: Handles exponential transformations for 
      TSS (mg/L) unit conversion.
-------------------------------------------------------------------------------
"""

import os
import logging
import joblib
import numpy as np
import rasterio
from pathlib import Path

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# 1) CONFIGURATION & PATHS
# =============================================================================
PKL_PATH = Path(r"path/to/your/plsr_vip_final_model.pkl")
PRISMA_TIF = Path(r"path/to/your/Prisma_geo_wm_correct_tif_renamed.tif")
OUT_DIR = Path(r"path/to/your/output/Prisma_Results")

# Prediction Settings
BLOCK_SIZE = 512
MAX_TSS_THRESHOLD = 500.0  # mg/L upper limit
EPSILON = 1e-6

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TIF = OUT_DIR / "PRISMA_PLSR_TSS_Prediction.tif"

# =============================================================================
# 2) MODEL LOADER (FLEXIBLE BUNDLE)
# =============================================================================


def load_plsr_bundle(path):
    """Extracts model and metadata from a joblib bundle or direct model file."""
    loaded = joblib.load(path)
    
    if isinstance(loaded, dict):
        model = loaded.get("pls_refit") or loaded.get("model") or loaded.get("pls")
        keep_cols = loaded.get("keep_cols", [])
        log_transform = loaded.get("log_transform", True)
        vip_table = loaded.get("vip_full_table")
    else:
        model = loaded
        keep_cols = []
        log_transform = True
        vip_table = None
        
    if model is None:
        raise ValueError("PLSR model object not found in the provided pickle file!")
        
    return model, keep_cols, log_transform, vip_table

def main():
    model, keep_cols, log_transform, vip_table = load_plsr_bundle(PKL_PATH)
    logging.info(f"Model Loaded. Log Transform: {log_transform}")

    # =========================================================================
    # 3) SPECTRAL BAND MAPPING
    # =========================================================================
    

    # Sensor-specific center wavelengths for PRISMA VNIR
    prisma_actual_wl = np.array([
        406.99, 415.83, 423.78, 431.33, 438.65, 446.01, 453.39, 460.73, 468.09, 475.31,
        482.54, 489.79, 497.05, 504.51, 512.04, 519.54, 527.30, 535.05, 542.88, 550.91,
        559.02, 567.20, 575.84, 583.84, 592.33, 601.01, 609.95, 618.72, 627.77, 636.67,
        645.96, 655.41, 664.89, 674.46, 684.13, 694.12, 703.73, 713.72, 723.87, 733.95,
        744.14, 754.65, 764.85, 775.27, 785.65, 796.12, 806.71, 817.31, 827.96, 838.52,
        849.27, 859.97, 870.74, 881.45, 892.05, 902.80, 913.63, 923.95, 934.67, 944.62,
        957.02, 967.00, 977.36
    ])

    # Identify wavelengths used during model training
    if vip_table is not None:
        train_wls = np.array(vip_table["Wavelength_nm"], dtype=float)
    else:
        train_wls = np.arange(400, 901, 1) # Fallback default

    # Map training wavelengths to PRISMA band indices (1-based for rasterio)
    matched_indices = [(np.abs(prisma_actual_wl - wl)).argmin() + 1 for wl in train_wls]

    # Filter for keep_cols if feature selection (VIP) was applied
    if keep_cols:
        full_feat_names = [f"X_{int(round(w))}" for w in train_wls]
        name_to_idx = {name: i for i, name in enumerate(full_feat_names)}
        selected_idx = [name_to_idx[c] for c in keep_cols if c in name_to_idx]
    else:
        selected_idx = list(range(len(train_wls)))

    # =========================================================================
    # 4) WINDOWED RASTER PROCESSING
    # =========================================================================
    with rasterio.open(PRISMA_TIF) as src:
        profile = src.profile.copy()
        H, W = src.height, src.width
        
        # Update metadata for single-band output
        profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)

        with rasterio.open(OUT_TIF, "w", **profile) as dst:
            logging.info(f"Starting windowed inference on {H}x{W} raster...")
            
            for row0 in range(0, H, BLOCK_SIZE):
                for col0 in range(0, W, BLOCK_SIZE):
                    h, w = min(BLOCK_SIZE, H - row0), min(BLOCK_SIZE, W - col0)
                    window = rasterio.windows.Window(col0, row0, w, h)
                    
                    # Read block spectral data
                    block_data = []
                    for b_idx in matched_indices:
                        arr = src.read(b_idx, window=window).astype(np.float32)
                        
                        # Apply PRISMA Scale Factor (DN to Reflectance)
                        if np.nanmax(arr) > 10:
                            arr /= 10000.0
                        block_data.append(arr.flatten())
                    
                    # Prepare Feature Matrix
                    X_block = np.stack(block_data, axis=-1)
                    X_block = np.clip(X_block, EPSILON, 1.0)

                    # Log Transformation (Feature Space)
                    if log_transform:
                        X_block = np.log(X_block)
                    
                    # Apply Feature Selection and Predict
                    X_final = X_block[:, selected_idx]
                    y_pred = model.predict(X_final).reshape(-1)

                    # Exponential Transformation (Target Space)
                    if log_transform:
                        y_pred = np.exp(y_pred)

                    # Apply Physical Thresholds
                    y_pred[(y_pred > MAX_TSS_THRESHOLD) | (y_pred < 0)] = np.nan
                    
                    # Write block to output raster
                    dst.write(y_pred.reshape(h, w).astype(np.float32), 1, window=window)

    logging.info(f"Process Finalized. Map generated at: {OUT_TIF}")

if __name__ == "__main__":
    main()
