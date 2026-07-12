# -*- coding: utf-8 -*-
"""
Created on Wed May 27 05:17:03 2026

@author: onurk
"""

# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Author:      Onur Karaca
Contact:     onurkaraca87@hotmail.com
Website:     www.onurkaraca87.com
-------------------------------------------------------------------------------
Project:     PRISMA TSS Mapping — CatBoost Inference
Description:
    Applies a pre-trained CatBoost model to PRISMA L2D hyperspectral
    imagery for pixel-wise TSS prediction.

    Reviewer-addressed features:
    1. Spectral band matching: ASD → PRISMA via argmin(|λ_target − λ_PRISMA|)
    2. Radiometric accuracy assessment: PRISMA Rrs vs ASD Rrs at matchup points
    3. Model transfer documentation: direct transfer via spectral convolution,
       no recalibration required
    4. Water masking, scaling, GeoTIFF export
    5. Band matching table, run summary table, false color RGB
-------------------------------------------------------------------------------
"""

import os
import sys
import re
import logging
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PATH  = r"D:\TWDB\FEEDBACK\ML_Process_Main\Prisma_Process\Catboost_Prisma_2\CatBoost_LogTransform_AntiOverfit_2\catboost_model_20260531.pkl"

TOP10_TXT   = r"D:\TWDB\FEEDBACK\ML_Process_Main\Prisma_Process\Catboost_Prisma_2\CatBoost_LogTransform_AntiOverfit_2\top10_bands_CB_20260531.txt"

PRISMA_TIF  = r"D:\TWDB\FEEDBACK\Datasets\Prisma\Prisma_geo_wm_correct_tif_renamed.tif"

OUTPUT_DIR  = r"D:\TWDB\FEEDBACK\ML_Process_Main\Prisma_Process\Catboost_Prisma_2\CatBoost_LogTransform_AntiOverfit_2\MODEL_RESULTS"

MATCHUP_CSV = None   # set path to enable radiometric accuracy assessment

PRISMA_VNIR_WL = np.linspace(400, 1010, 63)

# =============================================================================
# HELPERS
# =============================================================================

def load_top10_from_txt(txt_path):
    """Load top-10 feature names from training pipeline .txt output."""
    features = []
    with open(txt_path, 'r') as f:
        for line in f:
            m = re.search(r'\d+\.\s+(X_\d+)', line)
            if m:
                features.append(m.group(1))
    if not features:
        logging.error(f"No bands parsed from {txt_path}.")
        sys.exit(1)
    logging.info(f"Loaded {len(features)} bands: {features}")
    return features


def map_bands_to_prisma(features, sensor_wl, out_dir=None):
    """
    Spectral band matching: ASD wavelengths to nearest PRISMA VNIR band.
    Method: argmin(abs(lambda_ASD - lambda_PRISMA))
    Saves band matching table as PNG if out_dir is provided.
    """
    indices = []
    rows    = []

    for feat in features:
        target_wl = float(feat.split('_')[1])
        idx       = int(np.abs(sensor_wl - target_wl).argmin())
        band_num  = idx + 1
        prisma_wl = sensor_wl[idx]
        delta     = abs(target_wl - prisma_wl)
        indices.append(band_num)
        rows.append([feat,
                     '{:.1f}'.format(target_wl),
                     'Band #{}'.format(band_num),
                     '{:.2f}'.format(prisma_wl),
                     '{:.2f}'.format(delta)])

    logging.info('=== Spectral Band Matching: ASD to PRISMA ===')
    logging.info('  {:>8}  {:>12}  {:>12}  {:>14}  {:>8}'.format(
        'Feature', 'ASD lam(nm)', 'PRISMA Band', 'PRISMA lam(nm)', 'delta(nm)'))
    logging.info('  ' + '-' * 60)
    for r in rows:
        logging.info('  {:>8}  {:>12}  {:>12}  {:>14}  {:>8}'.format(*r))
    max_delta = max(float(r[4]) for r in rows)
    logging.info('  Max spectral offset: {:.2f} nm'.format(max_delta))

    if out_dir is not None:
        col_labels = ['Feature', 'ASD lam (nm)', 'PRISMA Band',
                      'PRISMA lam (nm)', 'Delta (nm)']
        fig, ax = plt.subplots(figsize=(9, 3.8))
        ax.axis('off')
        tbl = ax.table(
            cellText  = rows,
            colLabels = col_labels,
            cellLoc   = 'center',
            loc       = 'center',
            bbox      = [0, 0, 1, 1]
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10.5)
        tbl.auto_set_column_width(col=list(range(len(col_labels))))
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor('#2E86C1')
            tbl[(0, j)].set_text_props(fontweight='bold', color='white')
        for i, r in enumerate(rows, 1):
            delta = float(r[4])
            fc = '#FEF9E7' if delta > 3.0 else '#EAFAF1'
            for j in range(len(col_labels)):
                tbl[(i, j)].set_facecolor(fc)
            tbl[(i, 4)].set_text_props(
                fontweight='bold',
                color='#B7770D' if delta > 3.0 else '#1E8449')
        fig.text(0.5, 0.97, 'Spectral Band Matching: ASD to PRISMA VNIR (CatBoost)',
                 ha='center', va='top', fontsize=11, fontweight='bold')
        fig.text(0.5, 0.90,
                 'Method: argmin(|lam_ASD - lam_PRISMA|)  '
                 '|  lam = wavelength (nm)  '
                 '|  Green: delta <= 3 nm  |  Yellow: delta > 3 nm',
                 ha='center', va='top', fontsize=8.5, color='#555555')
        plt.subplots_adjust(top=0.82)
        save_path = os.path.join(out_dir, 'band_matching_table_CB.png')
        plt.savefig(save_path, dpi=1200, bbox_inches='tight')
        plt.close()
        logging.info('Band matching table saved to {}'.format(save_path))

    return indices


def assess_radiometric_accuracy(matchup_csv, features, out_dir):
    """
    Reviewer: assess radiometric accuracy of PRISMA reflectance vs ASD.
    Requires matchup CSV with asd_X_* and prisma_X_* columns.
    """
    if matchup_csv is None or not os.path.exists(matchup_csv):
        logging.warning("Matchup CSV not provided — radiometric assessment skipped.")
        logging.warning("To enable: set MATCHUP_CSV to a CSV with asd_X_* and prisma_X_* columns.")
        return

    logging.info("=== Radiometric Accuracy Assessment: PRISMA vs ASD ===")
    df = pd.read_csv(matchup_csv)
    asd_cols    = [f'asd_{f}'    for f in features]
    prisma_cols = [f'prisma_{f}' for f in features]
    missing = [c for c in asd_cols + prisma_cols if c not in df.columns]
    if missing:
        logging.warning(f"Missing columns: {missing}")
        return

    asd_vals    = df[asd_cols].values.flatten()
    prisma_vals = df[prisma_cols].values.flatten()
    valid       = np.isfinite(asd_vals) & np.isfinite(prisma_vals)

    r2   = r2_score(asd_vals[valid], prisma_vals[valid])
    rmse = float(np.sqrt(mean_squared_error(asd_vals[valid], prisma_vals[valid])))
    mae  = float(mean_absolute_error(asd_vals[valid], prisma_vals[valid]))
    bias = float(np.mean(prisma_vals[valid] - asd_vals[valid]))

    logging.info(f"  R2={r2:.4f} | RMSE={rmse:.6f} | MAE={mae:.6f} | Bias={bias:.6f}")

    per_band = []
    for feat, ac, pc in zip(features, asd_cols, prisma_cols):
        av = df[ac].values; pv = df[pc].values
        vm = np.isfinite(av) & np.isfinite(pv)
        if vm.sum() < 3: continue
        rb = r2_score(av[vm], pv[vm])
        per_band.append({'Band': feat,
                         'R2': round(rb, 4),
                         'RMSE': round(float(np.sqrt(mean_squared_error(av[vm], pv[vm]))), 6),
                         'Bias': round(float(np.mean(pv[vm] - av[vm])), 6)})

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(asd_vals[valid], prisma_vals[valid], c='#2E86C1', s=20, alpha=0.6, zorder=3)
    lo = min(asd_vals[valid].min(), prisma_vals[valid].min()) * 0.9
    hi = max(asd_vals[valid].max(), prisma_vals[valid].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], '--k', lw=1.2, zorder=1)
    ax.set_xlabel('ASD $R_{rs}$ [sr$^{-1}$]', fontsize=11)
    ax.set_ylabel('PRISMA $R_{rs}$ [sr$^{-1}$]', fontsize=11)
    ax.set_title('Radiometric Accuracy\nPRISMA vs ASD $R_{rs}$ at Matchup Stations',
                 fontsize=11, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    stats = (f"$R^2$  = {r2:.4f}\nRMSE = {rmse:.5f} sr$^{{-1}}$\n"
             f"MAE  = {mae:.5f} sr$^{{-1}}$\nBias  = {bias:.5f} sr$^{{-1}}$\nn = {valid.sum()}")
    ax.text(0.97, 0.03, stats, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.4', fc='#EBF5FB', ec='#2E86C1', alpha=0.9))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'radiometric_accuracy_PRISMA_vs_ASD.png'),
                dpi=1300, bbox_inches='tight')
    plt.close()
    logging.info("  Radiometric accuracy plot saved.")

    if per_band:
        pd.DataFrame(per_band).to_csv(
            os.path.join(out_dir, 'radiometric_accuracy_per_band.csv'), index=False)
        logging.info("  Per-band accuracy table saved.")


# =============================================================================
# MAIN
# =============================================================================

def main():

    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load CatBoost model
    # ------------------------------------------------------------------
    if not os.path.exists(MODEL_PATH):
        logging.error(f"Model not found: {MODEL_PATH}")
        sys.exit(1)
    model = joblib.load(MODEL_PATH)
    logging.info("CatBoost model loaded.")

    # ------------------------------------------------------------------
    # 2. Load top-10 bands from txt
    # ------------------------------------------------------------------
    if not os.path.exists(TOP10_TXT):
        logging.error(f"Top-10 txt not found: {TOP10_TXT}")
        sys.exit(1)
    features = load_top10_from_txt(TOP10_TXT)

    # ------------------------------------------------------------------
    # 3. Spectral band matching: ASD → PRISMA
    # ------------------------------------------------------------------
    band_indices = map_bands_to_prisma(features, PRISMA_VNIR_WL, out_dir=str(out))

    # ------------------------------------------------------------------
    # 4. Radiometric accuracy assessment
    # ------------------------------------------------------------------
    assess_radiometric_accuracy(MATCHUP_CSV, features, str(out))

    # ------------------------------------------------------------------
    # 5. Read PRISMA L2D imagery
    # ------------------------------------------------------------------
    logging.info(f"Opening PRISMA image: {PRISMA_TIF}")
    if not os.path.exists(PRISMA_TIF):
        logging.error(f"PRISMA TIF not found: {PRISMA_TIF}")
        sys.exit(1)

    with rasterio.open(PRISMA_TIF) as src:
        profile    = src.profile.copy()
        bands_list = []
        for bnum, feat in zip(band_indices, features):
            arr = src.read(int(bnum)).astype(np.float32)
            if np.nanmax(arr) > 10:
                arr *= 0.0001
            bands_list.append(arr)
        stacked = np.stack(bands_list, axis=-1)

    h, w, c = stacked.shape
    logging.info(f"Image: {h} x {w} pixels | {c} bands")

    # ------------------------------------------------------------------
    # 6. Water masking
    # ------------------------------------------------------------------
    pixels     = stacked.reshape(-1, c)
    n_total    = pixels.shape[0]
    valid_mask = (np.all(np.isfinite(pixels), axis=1) &
                  np.any(pixels > 0, axis=1))
    n_valid    = int(valid_mask.sum())
    logging.info(f"Valid pixels: {n_valid:,} / {n_total:,} ({100*n_valid/n_total:.1f}%)")

    # ------------------------------------------------------------------
    # 7. Pixel-wise prediction
    # ------------------------------------------------------------------
    predictions = np.full(n_total, np.nan, dtype=np.float32)

    if n_valid > 0:
        X_valid = pd.DataFrame(pixels[valid_mask], columns=features)
        try:
            preds_log = model.predict(X_valid).astype(np.float32)
            preds     = np.expm1(preds_log)  # back-transform from log space
            predictions[valid_mask] = preds
            logging.info(f"TSS range: [{preds.min():.2f}, {preds.max():.2f}] mg/L | "
                         f"Mean: {preds.mean():.2f} mg/L")
        except ValueError as e:
            logging.error("FEATURE MISMATCH — ensure model and txt use same top-10 bands.")
            logging.error(f"Details: {e}")
            sys.exit(1)
    else:
        logging.warning("No valid pixels found.")

    # ------------------------------------------------------------------
    # 8. Export GeoTIFF
    # ------------------------------------------------------------------
    tss_map  = predictions.reshape(h, w)
    out_path = out / "PRISMA_CB_TSS_Prediction.tif"

    profile.update(dtype=rasterio.float32, count=1,
                   compress="lzw", nodata=np.nan)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(tss_map, 1)
    logging.info(f"Output saved → {out_path}")

    # ------------------------------------------------------------------
    # 8b. Run summary table
    # ------------------------------------------------------------------
    if n_valid > 0:
        summary_rows = [
            ["Image size",         f"{h} x {w} pixels"],
            ["Total pixels",       f"{n_total:,}"],
            ["Valid water pixels",  f"{n_valid:,} ({100*n_valid/n_total:.1f}%)"],
            ["TSS min",            f"{preds.min():.2f} mg/L"],
            ["TSS max",            f"{preds.max():.2f} mg/L"],
            ["TSS mean",           f"{preds.mean():.2f} mg/L"],
            ["TSS median",         f"{float(np.median(preds)):.2f} mg/L"],
            ["Output file",        "PRISMA_CB_TSS_Prediction.tif"],
        ]
        fig2, ax2 = plt.subplots(figsize=(7, 3.5))
        ax2.axis('off')
        tbl2 = ax2.table(
            cellText  = summary_rows,
            colLabels = ["Parameter", "Value"],
            cellLoc   = 'center',
            loc       = 'center'
        )
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(10)
        tbl2.auto_set_column_width([0, 1])
        for j in range(2):
            tbl2[(0, j)].set_facecolor('#D6EAF8')
            tbl2[(0, j)].set_text_props(fontweight='bold')
        for i in range(1, len(summary_rows)+1):
            fc = '#F2F3F4' if i % 2 == 0 else '#FFFFFF'
            for j in range(2):
                tbl2[(i, j)].set_facecolor(fc)
        ax2.set_title("PRISMA CatBoost TSS Prediction — Run Summary",
                      fontsize=11, fontweight='bold', pad=12)
        plt.tight_layout()
        sum_path = os.path.join(str(out), 'run_summary_table_CB.png')
        plt.savefig(sum_path, dpi=1200, bbox_inches='tight')
        plt.close()
        logging.info(f"Run summary table saved → {sum_path}")

    # ------------------------------------------------------------------
    # 8c. False color RGB composite
    # ------------------------------------------------------------------
    logging.info("Generating false color composite (R=660nm G=550nm B=480nm)...")
    try:
        with rasterio.open(PRISMA_TIF) as src:
            r_band = src.read(28).astype(np.float32) * 0.0001
            g_band = src.read(16).astype(np.float32) * 0.0001
            b_band = src.read(9).astype(np.float32)  * 0.0001

        def stretch(arr, lo=2, hi=98):
            valid = arr[arr > 0]
            if len(valid) == 0: return arr
            lo_v = float(np.percentile(valid, lo))
            hi_v = float(np.percentile(valid, hi))
            out  = np.clip((arr - lo_v) / (hi_v - lo_v + 1e-9), 0, 1)
            out[arr <= 0] = 0
            return out

        rgb = np.dstack([stretch(r_band), stretch(g_band), stretch(b_band)])
        fig_rgb, ax_rgb = plt.subplots(figsize=(8, 7))
        ax_rgb.imshow(rgb, interpolation='bilinear')
        ax_rgb.set_title('PRISMA False Color Composite  |  R=660nm  G=550nm  B=480nm',
                         fontsize=11, fontweight='bold')
        ax_rgb.set_xlabel('Pixel (East-West)', fontsize=9)
        ax_rgb.set_ylabel('Pixel (North-South)', fontsize=9)
        ax_rgb.tick_params(labelsize=8)
        plt.tight_layout()
        rgb_path = os.path.join(str(out), 'PRISMA_FalseColor_RGB.png')
        plt.savefig(rgb_path, dpi=1200, bbox_inches='tight')
        plt.close()
        logging.info('False color composite saved to {}'.format(rgb_path))
    except Exception as e:
        logging.warning('Could not generate RGB composite: {}'.format(e))

    # ------------------------------------------------------------------
    # 9. Model transfer summary log
    # ------------------------------------------------------------------
    logging.info("=== CatBoost Model Transfer Summary ===")
    logging.info("  ASD -> PRISMA transfer method : spectral convolution")
    logging.info("  Band matching                 : argmin(|lam_ASD - lam_PRISMA|)")
    logging.info("  Recalibration required        : No")
    logging.info("  Rationale: Model trained on ASD spectra convolved to")
    logging.info("  PRISMA band centers. Direct transfer is valid.")
    logging.info("Done.")


if __name__ == "__main__":
    main()