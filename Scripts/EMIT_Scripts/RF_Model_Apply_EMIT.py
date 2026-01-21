"""
@author: sokaraca

"""

import os
import re
import numpy as np
import rasterio
import joblib

# ========================================
# 1) RF model path (EDIT HERE)
# ========================================
model_dir = r"............\Random_Forest"
model_path = os.path.join(model_dir, "Random_Forest_model.pkl")  # kendi dosya adına göre güncelle

model = joblib.load(model_path)
print(f"✅ Random Forest model loaded: {model_path}")

# ========================================
# 2) EMIT raster path (named tif)
# ========================================
emit_tif = r".............\Random_Forest_TSS_20241006.tif"

# ========================================
# 3) Model bands (MUST match training order!)
#    RF training scriptinden çıkan Selected_Bands ile aynı sıra
# ========================================
model_bands = [
    'X_630',
    'X_625',
    'X_635',
    'X_615',
    'X_495',
    'X_684',
    'X_638',
    'X_585',
    'X_606',
    'X_485'
]

if len(model_bands) != 10:
    raise ValueError("model_bands must contain EXACTLY 10 features.")

# ========================================
# 4) Helpers
# ========================================
def parse_wl_from_desc(desc: str):
    """Extract wavelength from 'Band i (xxxx.xxxx)' -> float nm"""
    if not desc:
        return None
    m = re.search(r"\(([-+]?\d+(?:\.\d+)?)\)", str(desc))
    return float(m.group(1)) if m else None

def parse_target_from_feat(feat: str):
    """'X_690' or 'X_690.0000' -> 690.0"""
    return float(str(feat).split("_", 1)[1])

def find_closest_band_index(target_wl, wl_array, tol_nm=10.0, hard_fail=True):
    """
    wl_array: numpy array of wavelengths aligned to band order (band i -> wl_array[i-1])
    returns: band_num (1-based), matched_wl, diff
    """
    idx0 = int(np.argmin(np.abs(wl_array - target_wl)))
    matched_wl = float(wl_array[idx0])
    diff = abs(matched_wl - target_wl)

    if diff > tol_nm:
        msg = f"No band within {tol_nm} nm for {target_wl}. Closest is {matched_wl:.4f} (diff={diff:.3f})."
        if hard_fail:
            raise ValueError("❌ " + msg)
        else:
            print("[WARN]", msg)

    return idx0 + 1, matched_wl, diff  # band_num is 1-based

# ========================================
# 5) Read EMIT raster, map bands, stack
# ========================================
with rasterio.open(emit_tif) as src:
    profile = src.profile.copy()
    height, width = src.height, src.width

    print(f"🚀 Raster dimensions: {width} x {height}")
    print(f"📏 Total raster bands: {src.count}")

    # wavelength list from descriptions (band order)
    wl_list = []
    for b in range(1, src.count + 1):
        desc = src.descriptions[b - 1]
        wl = parse_wl_from_desc(desc)
        if wl is None:
            raise RuntimeError(
                f"Band description missing/invalid at band {b}. "
                "Use *_named_NONCOG.tif or use CSV-based mapping."
            )
        wl_list.append(wl)

    wl_array = np.array(wl_list, dtype=np.float64)

    # map model features -> band numbers
    band_indices = {}
    print("\n=== Band matching (feature -> EMIT wavelength) ===")
    for feat in model_bands:
        target = parse_target_from_feat(feat)
        bnum, matched_wl, diff = find_closest_band_index(target, wl_array, tol_nm=10.0, hard_fail=True)
        band_indices[feat] = bnum
        print(f"{feat:>8} ({target:7.2f} nm) -> band #{bnum:3d}  ({matched_wl:9.4f} nm)  diff={diff:.3f} nm")

    # read selected bands
    bands_data = []
    for feat in model_bands:
        bnum = band_indices[feat]
        print(f"→ Reading {feat} (band #{bnum})")
        arr = src.read(bnum).astype(np.float32)
        bands_data.append(arr)

    stacked = np.stack(bands_data, axis=0).transpose(1, 2, 0)  # (H, W, 10)
    print(f"✅ Bands stacked shape (H, W, C): {stacked.shape}")

# ========================================
# 6) Predict
# ========================================
pixels_flat = stacked.reshape(-1, len(model_bands))
valid_mask = np.all(np.isfinite(pixels_flat), axis=1)
valid_pixels = pixels_flat[valid_mask]

print(f"Valid pixels: {valid_pixels.shape[0]} / {pixels_flat.shape[0]}")

preds = np.full((pixels_flat.shape[0],), np.nan, dtype=np.float32)
preds[valid_mask] = model.predict(valid_pixels).astype(np.float32)

tss_raster = preds.reshape(height, width).astype(np.float32)
print(f"✅ Prediction raster shape: {tss_raster.shape}")

# ========================================
# 7) Save output
# ========================================
output_dir = os.path.join(model_dir, "Model_sonuclari")
os.makedirs(output_dir, exist_ok=True)

output_tif = os.path.join(output_dir, "EMIT_20241006_RF_TSS_Prediction_10bands.tif")

profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)

with rasterio.open(output_tif, "w", **profile) as dst:
    dst.write(tss_raster, 1)

print(f"\n🎉 TSS prediction saved to:\n{output_tif}")
