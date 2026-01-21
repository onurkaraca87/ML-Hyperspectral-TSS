"""
@author: sokaraca

"""

import os
import re
import numpy as np
import rasterio
import joblib

# =========================================================
# 0) PATHS (EDIT HERE)
# =========================================================
MODEL_DIR  = r"......\CatBoost"
MODEL_PATH = os.path.join(MODEL_DIR, "Catboost_Model.pkl")

EMIT_TIF = r"..........\EMIT_L2A_RFL_20241006.tif"

OUT_DIR  = os.path.join(MODEL_DIR, "Model_Results")
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TIF  = os.path.join(OUT_DIR, "CatBoost_TSS_20241006.tif")

# =========================================================
# 1) MODEL INPUT BANDS (MUST MATCH TRAINING ORDER!)
# =========================================================
model_bands = [
    "X_430",
    "X_611",
    "X_582",
    "X_735",
    "X_797",
    "X_620",
    "X_586",
    "X_407",
    "X_664",
    "X_870",
]

if len(model_bands) != 10:
    raise ValueError("model_bands must contain EXACTLY 10 features.")

# =========================================================
# 2) HELPERS
# =========================================================
def parse_wl_from_desc(desc: str):
    """Extract wavelength from 'Band i (xxxx.xxxx)' -> float nm"""
    if not desc:
        return None
    m = re.search(r"\(([-+]?\d+(?:\.\d+)?)\)", str(desc))
    return float(m.group(1)) if m else None

def parse_target_from_feat(feat: str):
    """'X_690' or 'X_690.0000' -> 690.0"""
    return float(str(feat).split("_", 1)[1])

def find_closest_band_index(target_wl, wl_array, tol_nm=8.0, hard_fail=True):
    """
    wl_array: numpy array of wavelengths aligned to band order (band i -> wl_array[i-1])
    returns: band_num (1-based), matched_wl, diff
    """
    idx0 = int(np.argmin(np.abs(wl_array - target_wl)))  # 0-based index
    matched_wl = float(wl_array[idx0])
    diff = abs(matched_wl - target_wl)

    if diff > tol_nm:
        msg = f"No band within {tol_nm} nm for {target_wl}. Closest is {matched_wl:.4f} (diff={diff:.3f})."
        if hard_fail:
            raise ValueError("❌ " + msg)
        else:
            print("[WARN]", msg)

    band_num = idx0 + 1  # 1-based for rasterio
    return band_num, matched_wl, diff

# =========================================================
# 3) LOAD MODEL
# =========================================================
model = joblib.load(MODEL_PATH)
print(f"✅ CatBoost model loaded: {MODEL_PATH}")

# =========================================================
# 4) OPEN EMIT, READ WAVELENGTH LIST (band order)
# =========================================================
with rasterio.open(EMIT_TIF) as src:
    profile = src.profile.copy()
    height, width = src.height, src.width

    print(f"📏 Raster size: {width} x {height}")
    print(f"📦 Total bands in EMIT file: {src.count}")

    # Build wavelength array in band order
    wl_list = []
    for b in range(1, src.count + 1):
        desc = src.descriptions[b - 1]
        wl = parse_wl_from_desc(desc)
        if wl is None:
            raise RuntimeError(
                f"Band description missing/invalid at band {b}. "
                "Make sure you are using the *_named.tif."
            )
        wl_list.append(wl)

    wl_array = np.array(wl_list, dtype=np.float64)

    # Map model features -> band numbers
    band_indices = {}
    print("\n=== Band matching (feature -> EMIT wavelength) ===")
    for feat in model_bands:
        target = parse_target_from_feat(feat)
        bnum, matched_wl, diff = find_closest_band_index(
            target_wl=target,
            wl_array=wl_array,
            tol_nm=8.0,     # <- if needed set 10.0
            hard_fail=True
        )
        band_indices[feat] = bnum
        print(f"{feat:>8} ({target:7.2f} nm) -> band #{bnum:3d}  ({matched_wl:9.4f} nm)  diff={diff:.3f} nm")

    # =========================================================
    # 5) READ 10 BANDS AND STACK
    # =========================================================
    bands_data = []
    for feat in model_bands:
        bnum = band_indices[feat]
        arr = src.read(bnum).astype(np.float32)
        bands_data.append(arr)

    stacked = np.stack(bands_data, axis=0).transpose(1, 2, 0)  # (H, W, 10)
    print(f"\n✅ Stacked bands shape: {stacked.shape}  (H, W, 10)")

# =========================================================
# 6) PREDICT
# =========================================================
pixels_flat = stacked.reshape(-1, len(model_bands))  # (N, 10)
valid_mask = np.all(np.isfinite(pixels_flat), axis=1)
valid_pixels = pixels_flat[valid_mask]

print(f"✔️ Valid pixels: {valid_pixels.shape[0]} / {pixels_flat.shape[0]}")

preds = np.full((pixels_flat.shape[0],), np.nan, dtype=np.float32)
preds[valid_mask] = model.predict(valid_pixels).astype(np.float32)

tss_raster = preds.reshape(height, width).astype(np.float32)
print(f"✅ Prediction raster generated: {tss_raster.shape}")

# =========================================================
# 7) SAVE OUTPUT
# =========================================================
profile.update(
    dtype=rasterio.float32,
    count=1,
    compress="lzw",
    nodata=np.nan
)

with rasterio.open(OUT_TIF, "w", **profile) as dst:
    dst.write(tss_raster, 1)

print("\n🎉 TSS prediction saved to:")
print(OUT_TIF)
