# -*- coding: utf-8 -*-
"""
Apply your saved log-log PLSR PKL bundle (Your_VisNIR_400_900_model_refit.pkl)
to PRISMA raster using a band<->wavelength CSV and linear interpolation.

Key fixes:
- Reconstruct FULL feature order from bundle['vip_full_table'] (usually ~501)
  so SimpleImputer.statistics_ matches feature count
- Then select keep_cols (e.g., 143) for pls_refit
- Robust physical masking + log safety + overflow clamp
- Windowed processing

@author: sokaraca
"""

import os
import numpy as np
import pandas as pd
import rasterio
import joblib

# ============================
# 0) PATHS (EDIT)
# ============================
PKL_PATH = r".......\PLSR_VisNIR_400_900_Model.pkl"

PRISMA_TIF = r"......\Prisma_20250310.tif"

# band,wavelength_nm  (band is 1-based)
WL_MAP_CSV = r"......\prisma_wavelengths.csv"

OUT_DIR = r"......\Prisma_2\Model_Results"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_TIF = os.path.join(OUT_DIR, "PRISMA_PLSR_TSS_VisNIR400_900.tif")

# ============================
# 1) SAFETY SETTINGS
# ============================
BLOCK = 256           # window size
MAX_RRS = 0.30        # reflectance sanity upper bound (water typically << 0.3)
MAX_TSS = 5000.0      # sanity bound
NODATA_OUT = np.nan   # keep NaN nodata

# ============================
# 2) LOAD PKL BUNDLE
# ============================
bundle = joblib.load(PKL_PATH)
pls = bundle["pls_refit"]
keep_cols = list(bundle["keep_cols"])
EPS = float(bundle.get("EPS", 1e-6))
log_transform = bool(bundle.get("log_transform", True))

vip_full_table = bundle.get("vip_full_table", None)
imputer = bundle.get("imputer", None)

print(f"✅ PKL loaded: {PKL_PATH}")
print(f"   keep_cols: {len(keep_cols)} bands")
print(f"   EPS={EPS}  log_transform={log_transform}")

if vip_full_table is None:
    raise RuntimeError("❌ bundle['vip_full_table'] missing. Re-save PKL with vip_full_table included.")

# Rebuild FULL columns order from vip_full_table wavelengths
full_wls = np.array(vip_full_table["Wavelength_nm"], dtype=float)
full_cols = [f"X_{int(round(w))}" for w in full_wls]

# imputer statistics must match full feature length (usually 501)
stats = None
if imputer is not None and hasattr(imputer, "statistics_"):
    stats = np.array(imputer.statistics_, dtype=np.float32)
    print(f"   imputer.stats: {len(stats)} features")

# If mismatch, we will fallback to per-window median impute
use_stats_impute = (stats is not None and len(stats) == len(full_cols))

if not use_stats_impute:
    print("⚠️ Imputer statistics do NOT match full feature count. Will use per-window median impute (slower).")

# Indices of keep_cols inside full_cols (order must match keep_cols)
full_index = {c: i for i, c in enumerate(full_cols)}
keep_idx = []
missing_keep = []
for c in keep_cols:
    if c in full_index:
        keep_idx.append(full_index[c])
    else:
        missing_keep.append(c)

if len(missing_keep) > 0:
    print(f"⚠️ keep_cols not found in reconstructed full_cols: {len(missing_keep)} (will be filled as NaN then imputed)")

keep_idx = np.array(keep_idx, dtype=int)

# ============================
# 3) READ PRISMA WL MAP CSV
# ============================
m = pd.read_csv(WL_MAP_CSV)
if not {"band", "wavelength_nm"}.issubset(set(m.columns.str.lower())):
    # allow flexible header casing
    cols_lower = {c.lower(): c for c in m.columns}
    if "band" not in cols_lower or "wavelength_nm" not in cols_lower:
        raise ValueError("❌ WL_MAP_CSV must have columns: band, wavelength_nm")
    band_col = cols_lower["band"]
    wl_col = cols_lower["wavelength_nm"]
else:
    band_col = "band"
    wl_col = "wavelength_nm"

bands_csv = m[band_col].astype(int).to_numpy()
wls_csv = m[wl_col].astype(float).to_numpy()

# Sort by wavelength (needed for interpolation)
order = np.argsort(wls_csv)
bands_sorted = bands_csv[order]
wls_sorted = wls_csv[order]

# ============================
# 4) OPEN PRISMA + PRECOMPUTE INTERP FOR FULL_WLS
# ============================
with rasterio.open(PRISMA_TIF) as src:
    profile = src.profile.copy()
    H, W = src.height, src.width
    n_bands = src.count

    print(f"📏 PRISMA: {W} x {H} | bands: {n_bands}")
    print(f"✅ WL map: {wls_sorted.min():.3f} – {wls_sorted.max():.3f} nm")

    # For each target wl (full_wls), find left/right indices in sorted wl grid
    # idxR = first index where wls_sorted[idxR] >= target
    idxR = np.searchsorted(wls_sorted, full_wls, side="left")
    idxL = idxR - 1

    # clip bounds
    idxR = np.clip(idxR, 0, len(wls_sorted)-1)
    idxL = np.clip(idxL, 0, len(wls_sorted)-1)

    wlL = wls_sorted[idxL]
    wlR = wls_sorted[idxR]
    bL = bands_sorted[idxL]  # 1-based band numbers
    bR = bands_sorted[idxR]

    denom = (wlR - wlL)
    denom[denom == 0] = 1.0
    alpha = (full_wls - wlL) / denom
    alpha = alpha.astype(np.float32)

    # Identify which targets are outside PRISMA wl range (will be NaN then imputed)
    outside = (full_wls < wls_sorted.min()) | (full_wls > wls_sorted.max())
    print(f"   full_wls outside PRISMA range: {outside.sum()} / {len(full_wls)}")

    # Output raster setup
    out_profile = profile.copy()
    out_profile.update(
        dtype=rasterio.float32,
        count=1,
        compress="lzw",
        nodata=NODATA_OUT
    )

    with rasterio.open(OUT_TIF, "w", **out_profile) as dst:

        # Window loop
        for row0 in range(0, H, BLOCK):
            for col0 in range(0, W, BLOCK):
                h = min(BLOCK, H - row0)
                w = min(BLOCK, W - col0)
                win = rasterio.windows.Window(col0, row0, w, h)

                # Read only needed PRISMA bands for this window (unique)
                needed = np.unique(np.concatenate([bL[~outside], bR[~outside]])).astype(int)
                band_data = {}
                for b in needed:
                    band_data[int(b)] = src.read(int(b), window=win).astype(np.float32)

                # Build X_full for this window: (n_pix, n_full)
                n_pix = h * w
                X_full_win = np.empty((n_pix, len(full_wls)), dtype=np.float32)

                # Interpolate each target wl
                for i in range(len(full_wls)):
                    if outside[i]:
                        X_full_win[:, i] = np.nan
                        continue

                    left_band = int(bL[i])
                    right_band = int(bR[i])

                    arrL = band_data[left_band]
                    arrR = band_data[right_band]

                    # linear interpolation
                    a = alpha[i]
                    interp = (1.0 - a) * arrL + a * arrR

                    # physical mask
                    interp[(interp <= 0) | (interp > 1.0)] = np.nan

                    X_full_win[:, i] = interp.reshape(-1)

                # ---- Impute
                if use_stats_impute:
                    # fill NaNs with training medians (statistics_)
                    X_imp = X_full_win.copy()
                    nanmask = ~np.isfinite(X_imp)
                    if nanmask.any():
                        X_imp[nanmask] = np.take(stats, np.where(nanmask)[1])
                else:
                    # per-window median per feature
                    X_imp = X_full_win.copy()
                    med = np.nanmedian(X_imp, axis=0)
                    # if a column is all-NaN, fallback to EPS
                    med[~np.isfinite(med)] = EPS
                    nanmask = ~np.isfinite(X_imp)
                    if nanmask.any():
                        X_imp[nanmask] = np.take(med, np.where(nanmask)[1])

                # Clip reflectance
                X_imp = np.clip(X_imp, EPS, MAX_RRS)

                # Log transform if required
                if log_transform:
                    X_in = np.log(X_imp + EPS).astype(np.float32)
                else:
                    X_in = X_imp.astype(np.float32)

                # Select keep features in correct order
                # If some keep cols were missing in full_cols, they never appear here; but that’s rare.
                X_keep = X_in[:, keep_idx]

                # Predict (log-space -> exp)
                y_log = pls.predict(X_keep).reshape(-1).astype(np.float32)

                if log_transform:
                    y = np.exp(y_log).astype(np.float32)
                else:
                    y = y_log.astype(np.float32)

                # Output sanity clamp
                y[~np.isfinite(y)] = np.nan
                y[(y < 0) | (y > MAX_TSS)] = np.nan

                out_block = y.reshape(h, w).astype(np.float32)
                dst.write(out_block, 1, window=win)

                print(f"✅ wrote window row={row0}:{row0+h} col={col0}:{col0+w}")

print("\n🎉 Done. Saved:")
print(OUT_TIF)
