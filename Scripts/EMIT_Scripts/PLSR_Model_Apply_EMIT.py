"""
@author: sokaraca

"""
# -*- coding: utf-8 -*-
import os, re
import joblib
import numpy as np
import rasterio

PKL_PATH = r"..........\PLSR_VisNIR_400_900_Model.pkl"
EMIT_TIF = r"..........\EMIT_L2A_RFL_20241006.tif"

OUT_DIR = r"..........\Model_Results"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_TIF = os.path.join(OUT_DIR, "EMIT_PLSR_TSS_VisNIR400_900.tif")

SCALE_DIV = 1.0   # senin durumda doğru
EPS_FALLBACK = 1e-6

def parse_wl_from_desc(desc: str):
    if not desc:
        return None
    m = re.search(r"\(([-+]?\d+(?:\.\d+)?)\)", str(desc))
    return float(m.group(1)) if m else None

def nm_from_keepcol(x: str) -> float:
    return float(str(x).split("_", 1)[1])

bundle = joblib.load(PKL_PATH)
pls = bundle["pls_refit"]
keep_cols = bundle["keep_cols"]
log_transform = bool(bundle.get("log_transform", True))
EPS = float(bundle.get("EPS", EPS_FALLBACK))

keep_wl = np.array([nm_from_keepcol(c) for c in keep_cols], dtype=np.float64)

print("✅ PKL loaded:", PKL_PATH)
print("   keep bands:", len(keep_cols), "log_transform=", log_transform, "EPS=", EPS)

with rasterio.open(EMIT_TIF) as src:
    profile = src.profile.copy()
    h, w = src.height, src.width
    nb = src.count
    print(f"📏 Raster {w}x{h} bands={nb}")

    # EMIT wavelengths (length=nb)
    wl = []
    for i in range(nb):
        v = parse_wl_from_desc(src.descriptions[i])
        if v is None:
            raise RuntimeError(f"Missing band description at band {i+1}.")
        wl.append(v)
    wl = np.array(wl, dtype=np.float64)

    # sort wavelengths just in case (usually already sorted)
    sort_idx = np.argsort(wl)
    wl_sorted = wl[sort_idx]

    # read all bands (nb, h, w) as float32
    cube = src.read().astype(np.float32)  # (nb,h,w)
    cube = cube[sort_idx, :, :]           # sort to match wl_sorted

# reshape to pixels: (N, nb)
X_emit = cube.reshape(cube.shape[0], -1).T  # (N, nb)

# scale
X_emit = X_emit / SCALE_DIV

# valid pixels: require finite in all bands (or you can relax if needed)
valid = np.all(np.isfinite(X_emit), axis=1)
Xv = X_emit[valid]

# clip negative
Xv = np.clip(Xv, 0, None)

# ---- INTERPOLATE from EMIT grid -> keep_wl grid ----
# np.interp works 1D, so we do it in chunks for memory
target = keep_wl
Xv_out = np.empty((Xv.shape[0], target.shape[0]), dtype=np.float32)

chunk = 200000  # adjust if RAM limited
for start in range(0, Xv.shape[0], chunk):
    end = min(start + chunk, Xv.shape[0])
    block = Xv[start:end]  # (m, nb)
    # interpolate each pixel row
    # loop is ok per-chunk; faster approach exists but this is robust
    for i in range(block.shape[0]):
        Xv_out[start + i] = np.interp(target, wl_sorted, block[i]).astype(np.float32)

# log
if log_transform:
    Xv_out = np.log(Xv_out + EPS)

# predict (log y) -> linear
pred_log = pls.predict(Xv_out).ravel().astype(np.float32)

pred_lin_e = np.exp(pred_log)            # natural log case
pred_lin_10 = np.power(10.0, pred_log)   # if log10 case (test)

# pick one (şimdilik exp)
pred = pred_lin_e

out = np.full((X_emit.shape[0],), np.nan, dtype=np.float32)
out[valid] = pred.astype(np.float32)
out_img = out.reshape(h, w)

print("✅ Output stats (exp):", np.nanmin(out_img), np.nanmax(out_img))

profile.update(dtype=rasterio.float32, count=1, compress="lzw", nodata=np.nan)
with rasterio.open(OUT_TIF, "w", **profile) as dst:
    dst.write(out_img, 1)

print("🎉 Saved:", OUT_TIF)
print("NOTE: If values still off, try pred = pred_lin_10 (log10 check).")
