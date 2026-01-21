# -*- coding: utf-8 -*-
"""
PRISMA → CatBoost TSS Prediction (10-band)  [HARDCODED VERSION]

- Input/Model/Band mapping are written INSIDE the script (no CLI arguments).
- Applies a trained CatBoost model (joblib/pickle) to a multiband PRISMA GeoTIFF.
- Reads specified bands in training order, predicts per-block (memory-safe),
  and writes a single-band GeoTIFF output (TSS).

Author: Onur Karaca
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import joblib


# =============================================================================
# 1) PATHS / SETTINGS  (EDIT THESE)
# =============================================================================

# Your folder
BASE_DIR = Path(r"......\CatBoost")  # <-- change if needed

# Input PRISMA multiband GeoTIFF
INPUT_TIF = BASE_DIR / "Prisma_20250310.tif"

# Trained CatBoost model (pickle/joblib)
MODEL_PKL = BASE_DIR / "Catboost_Model.pkl"

# Output GeoTIFF (single band: predicted TSS)
OUTPUT_TIF = BASE_DIR / "Prisma_20250310__TSS.tif"

# Output NoData and compression
OUT_NODATA = np.float32(-9999.0)
OUT_COMPRESS = "lzw"


# =============================================================================
# 2) MODEL BAND NAMES (MUST MATCH TRAINING ORDER EXACTLY!)
# =============================================================================
MODEL_BANDS: List[str] = [
    "X_430", "X_611", "X_582", "X_735", "X_797",
    "X_620", "X_586", "X_407", "X_664", "X_870",
]


# =============================================================================
# 3) BAND INDICES (MOST IMPORTANT PART!)
#    These integers are the band numbers inside your PRISMA GeoTIFF.
#    Rasterio uses 1-based indexing: band 1 = first band.
#
#    ⚠️ The values below are EXAMPLES. REPLACE with your true mapping.
# =============================================================================
BAND_INDICES: Dict[str, int] = {
    "X_430": 3,
    "X_611": 27,
    "X_582": 24,
    "X_735": 40,
    "X_797": 46,
    "X_620": 28,
    "X_586": 25,
    "X_407": 1,
    "X_664": 33,
    "X_870": 53,
}


# =============================================================================
# 4) HELPER FUNCTIONS
# =============================================================================

def warn_duplicate_indices(band_indices: Dict[str, int]) -> None:
    inv: Dict[int, List[str]] = {}
    for name, idx in band_indices.items():
        inv.setdefault(idx, []).append(name)
    dups = [(idx, names) for idx, names in inv.items() if len(names) > 1]
    if dups:
        print("⚠️ WARNING: Duplicate band indices detected (same raster band used multiple times):")
        for idx, names in dups:
            print(f"   band #{idx}: {names}")
        print("   Please verify your wavelength→band mapping.\n")


def validate_mapping(model_bands: List[str], band_indices: Dict[str, int], src_count: int) -> None:
    missing = [b for b in model_bands if b not in band_indices]
    if missing:
        raise ValueError(f"Missing band index for: {missing}")

    bad_range = [(b, band_indices[b]) for b in model_bands if not (1 <= band_indices[b] <= src_count)]
    if bad_range:
        msg = ", ".join([f"{b}={idx}" for b, idx in bad_range])
        raise ValueError(f"Band indices out of range (valid 1..{src_count}): {msg}")


def try_check_feature_names(model, model_bands: List[str]) -> None:
    feat: Optional[List[str]] = None
    if hasattr(model, "feature_names_"):
        try:
            feat = list(getattr(model, "feature_names_"))
        except Exception:
            feat = None

    if feat and feat != model_bands:
        print("⚠️ WARNING: Model feature_names_ != MODEL_BANDS.")
        print(f"   model feature_names_: {feat}")
        print(f"   provided MODEL_BANDS: {model_bands}")
        print("   If names/order do not match training, predictions will be invalid.\n")


def predict_block(
    model,
    src: rasterio.io.DatasetReader,
    window: rasterio.windows.Window,
    model_bands: List[str],
    band_indices: Dict[str, int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      pred_2d: float32, shape (h,w), np.nan where invalid
      valid_2d: bool, shape (h,w)
    """
    arrays: List[np.ndarray] = []
    masks: List[np.ndarray] = []

    for bname in model_bands:
        bnum = band_indices[bname]
        arr = src.read(bnum, window=window).astype(np.float32)
        msk = src.read_masks(bnum, window=window)  # uint8 (0 invalid, 255 valid)

        # Apply source nodata if defined
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

        arrays.append(arr)
        masks.append(msk)

    # (C,H,W)
    arr_stack = np.stack(arrays, axis=0)
    msk_stack = np.stack(masks, axis=0)

    # Valid if: all bands finite AND all masks > 0
    finite_ok = np.all(np.isfinite(arr_stack), axis=0)   # (H,W)
    mask_ok = np.all(msk_stack > 0, axis=0)              # (H,W)
    valid_2d = finite_ok & mask_ok

    # Features (N,C)
    hwc = np.transpose(arr_stack, (1, 2, 0))             # (H,W,C)
    feats = hwc.reshape(-1, len(model_bands))            # (N,C)
    valid_flat = valid_2d.reshape(-1)

    preds_flat = np.full((feats.shape[0],), np.nan, dtype=np.float32)
    if np.any(valid_flat):
        preds_flat[valid_flat] = model.predict(feats[valid_flat])

    pred_2d = preds_flat.reshape(hwc.shape[0], hwc.shape[1]).astype(np.float32)
    return pred_2d, valid_2d


# =============================================================================
# 5) MAIN
# =============================================================================

def main() -> None:
    print("=== PRISMA → CatBoost TSS Prediction (Hardcoded) ===")
    print(f"📁 BASE_DIR   : {BASE_DIR}")
    print(f"📥 INPUT_TIF  : {INPUT_TIF}")
    print(f"🧠 MODEL_PKL  : {MODEL_PKL}")
    print(f"📤 OUTPUT_TIF : {OUTPUT_TIF}")
    print(f"🧩 MODEL_BANDS (order matters): {MODEL_BANDS}\n")

    # Check files exist
    if not INPUT_TIF.exists():
        raise FileNotFoundError(f"Input not found: {INPUT_TIF}")
    if not MODEL_PKL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PKL}")

    warn_duplicate_indices(BAND_INDICES)

    # Load model
    model = joblib.load(MODEL_PKL)
    print("✅ Model loaded.")
    try_check_feature_names(model, MODEL_BANDS)

    # Open raster, validate mapping, process by blocks
    with rasterio.open(INPUT_TIF) as src:
        print(f"📦 Source bands: {src.count} | size: {src.width} x {src.height}")
        print(f"ℹ️ src.nodata : {src.nodata}\n")

        validate_mapping(MODEL_BANDS, BAND_INDICES, src.count)

        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=OUT_NODATA,
            compress=OUT_COMPRESS,
        )

        total_valid = 0
        total_px = 0

        OUTPUT_TIF.parent.mkdir(parents=True, exist_ok=True)

        with rasterio.open(OUTPUT_TIF, "w", **profile) as dst:
            for _, window in src.block_windows(1):
                pred_2d, valid_2d = predict_block(model, src, window, MODEL_BANDS, BAND_INDICES)

                # NaN -> nodata
                pred_write = np.where(np.isfinite(pred_2d), pred_2d, OUT_NODATA).astype(np.float32)
                dst.write(pred_write, 1, window=window)

                total_valid += int(valid_2d.sum())
                total_px += int(valid_2d.size)

        print(f"✔️ Valid pixels: {total_valid} / {total_px}")
        print("🎉 Done.")


if __name__ == "__main__":
    main()

