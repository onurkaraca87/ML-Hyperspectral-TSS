# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 01:33:09 2025

@author: sokaraca
"""

# -*- coding: utf-8 -*-
"""
PRISMA → CatBoost TSS Prediction (10-band)

Apply a trained CatBoost model (saved via joblib/pickle) to a multiband PRISMA GeoTIFF.
Extracts user-specified bands in the exact training order, predicts per-block (memory-safe),
and writes a single-band GeoTIFF output.

Author: Onur Karaca
"""

# =============================================================================
# IMPORTANT NOTICE – READ BEFORE USING THIS SCRIPT
# =============================================================================
# This script DOES NOT automatically infer wavelengths or band meanings.
#
# 1) Band names (model_bands)
#    - Must EXACTLY match the feature names used during model training.
#    - Names like "X_430" are project-specific and NOT universal.
#    - If your trained model uses different names, you MUST update model_bands
#      or pass --model-bands accordingly.
#
# 2) Band indices (band_indices)
#    - Must correctly map model band names to PRISMA raster band numbers.
#    - Rasterio uses 1-based indexing (band 1 = first band).
#    - Band numbers vary by PRISMA product, processing level, and preprocessing.
#
# 3) Band order
#    - The order of model_bands MUST be identical to the training order.
#    - Wrong order = wrong features = INVALID predictions.
#
# The user is fully responsible for verifying wavelength–band consistency.
# This script may warn about duplicate indices, but cannot validate scientific correctness.
# =============================================================================

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import rasterio
import joblib


DEFAULT_MODEL_BANDS: List[str] = [
    "X_430", "X_611", "X_582", "X_735", "X_797",
    "X_620", "X_586", "X_407", "X_664", "X_870",
]


def _load_band_indices(arg: str) -> Dict[str, int]:
    """
    --band-indices accepts either:
      (a) path to a JSON file
      (b) JSON string e.g. '{"X_430":4,"X_611":27,...}'

    Returns: dict {band_name: 1-based_band_number}
    """
    arg = arg.strip()
    p = Path(arg)
    if p.exists() and p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
    else:
        data = json.loads(arg)

    if not isinstance(data, dict):
        raise ValueError("band_indices must be a JSON object mapping band names to 1-based band numbers.")

    out: Dict[str, int] = {}
    for k, v in data.items():
        if not isinstance(k, str):
            raise ValueError("band_indices keys must be strings (e.g., 'X_430').")
        if not isinstance(v, int):
            raise ValueError(f"band index for '{k}' must be an integer (Rasterio bands are 1-based).")
        out[k] = v
    return out


def _warn_duplicate_indices(band_indices: Dict[str, int]) -> None:
    inv: Dict[int, List[str]] = {}
    for name, idx in band_indices.items():
        inv.setdefault(idx, []).append(name)
    dups = [(idx, names) for idx, names in inv.items() if len(names) > 1]
    if dups:
        print("⚠️ WARNING: Duplicate band indices detected (same raster band used multiple times):")
        for idx, names in dups:
            print(f"   band #{idx}: {names}")
        print("   Please verify your wavelength→band map. This is often a mistake.\n")


def _validate_mapping(
    model_bands: List[str],
    band_indices: Dict[str, int],
    src_count: int
) -> None:
    missing = [b for b in model_bands if b not in band_indices]
    if missing:
        raise ValueError(f"Missing band index for: {missing}")

    bad_range = [(b, band_indices[b]) for b in model_bands if not (1 <= band_indices[b] <= src_count)]
    if bad_range:
        msg = ", ".join([f"{b}={idx}" for b, idx in bad_range])
        raise ValueError(f"Band indices out of range (valid 1..{src_count}): {msg}")


def _try_check_feature_names(model, model_bands: List[str]) -> None:
    """
    Optional: If the loaded model exposes feature names, warn if mismatch.
    (Not all pickled models preserve this reliably across wrappers.)
    """
    feat: Optional[List[str]] = None

    # CatBoostRegressor has feature_names_ or feature_names_ may exist
    if hasattr(model, "feature_names_"):
        try:
            feat = list(getattr(model, "feature_names_"))
        except Exception:
            feat = None

    if feat:
        if feat != model_bands:
            print("⚠️ WARNING: Model feature_names_ != provided model_bands.")
            print(f"   model feature_names_: {feat}")
            print(f"   provided model_bands: {model_bands}")
            print("   If order/names do not match training, predictions will be invalid.\n")


def _predict_block(
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

        # apply source nodata if defined
        if src.nodata is not None:
            arr = np.where(arr == src.nodata, np.nan, arr)

        arrays.append(arr)
        masks.append(msk)

    # Stack to (C,H,W)
    arr_stack = np.stack(arrays, axis=0)   # (C,H,W)
    msk_stack = np.stack(masks, axis=0)    # (C,H,W)

    # Valid if: all bands finite AND all masks > 0
    finite_ok = np.all(np.isfinite(arr_stack), axis=0)         # (H,W)
    mask_ok = np.all(msk_stack > 0, axis=0)                    # (H,W)
    valid_2d = finite_ok & mask_ok

    # Features to (N,C)
    hwc = np.transpose(arr_stack, (1, 2, 0))                   # (H,W,C)
    feats = hwc.reshape(-1, len(model_bands))                  # (N,C)
    valid_flat = valid_2d.reshape(-1)

    preds_flat = np.full((feats.shape[0],), np.nan, dtype=np.float32)
    if np.any(valid_flat):
        preds_flat[valid_flat] = model.predict(feats[valid_flat])

    pred_2d = preds_flat.reshape(hwc.shape[0], hwc.shape[1]).astype(np.float32)
    return pred_2d, valid_2d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a trained CatBoost model to a multiband PRISMA GeoTIFF and write TSS predictions."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input PRISMA multiband GeoTIFF.")
    parser.add_argument("--model", "-m", required=True, help="Path to trained CatBoost model .pkl (joblib/pickle).")
    parser.add_argument(
        "--band-indices",
        "-b",
        required=True,
        help="JSON string OR path to JSON file mapping model band names to 1-based raster band numbers.",
    )
    parser.add_argument(
        "--model-bands",
        default=",".join(DEFAULT_MODEL_BANDS),
        help="Comma-separated model band names in training order (default: 10-band list).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output GeoTIFF path. If omitted, auto-named next to input or in --out-dir.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (used only if --output is omitted). Defaults to input folder.",
    )
    parser.add_argument("--nodata", type=float, default=-9999.0, help="Output nodata value (default: -9999).")
    parser.add_argument("--compress", default="lzw", help="GeoTIFF compression (default: lzw).")

    args = parser.parse_args()

    input_path = Path(args.input)
    model_path = Path(args.model)

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_bands = [x.strip() for x in args.model_bands.split(",") if x.strip()]
    if not model_bands:
        raise ValueError("model_bands is empty. Provide --model-bands or use defaults.")

    band_indices = _load_band_indices(args.band_indices)
    _warn_duplicate_indices(band_indices)

    # Load model
    model = joblib.load(model_path)
    print(f"✅ Model loaded: {model_path}")
    _try_check_feature_names(model, model_bands)

    # Output path
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = Path(args.out_dir) if args.out_dir else input_path.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{input_path.stem}__catboost_tss__{len(model_bands)}bands.tif"

    print(f"📥 Input : {input_path}")
    print(f"📤 Output: {out_path}")
    print(f"🧩 Bands : {model_bands}")

    out_nodata = np.float32(args.nodata)

    with rasterio.open(input_path) as src:
        print(f"📦 Source bands: {src.count} | size: {src.width} x {src.height}")
        print(f"ℹ️ src.nodata: {src.nodata}")

        _validate_mapping(model_bands, band_indices, src.count)

        profile = src.profile.copy()
        profile.update(
            dtype=rasterio.float32,
            count=1,
            nodata=out_nodata,
            compress=args.compress,
        )

        total_valid = 0
        total_px = 0

        with rasterio.open(out_path, "w", **profile) as dst:
            # Memory-safe processing
            for _, window in src.block_windows(1):
                pred_2d, valid_2d = _predict_block(model, src, window, model_bands, band_indices)

                # Write: NaN -> nodata
                pred_write = np.where(np.isfinite(pred_2d), pred_2d, out_nodata).astype(np.float32)
                dst.write(pred_write, 1, window=window)

                total_valid += int(valid_2d.sum())
                total_px += int(valid_2d.size)

        print(f"✔️ Valid pixels: {total_valid} / {total_px}")
        print("🎉 Done.")


if __name__ == "__main__":
    main()
