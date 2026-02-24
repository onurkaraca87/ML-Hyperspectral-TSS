# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 17:00:15 2026

@author: sokaraca
"""

# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Author:      Onur Karaca
Contact:     onurkaraca87@hotmail.com
Website:     www.onurkaraca87.com
-------------------------------------------------------------------------------
Project:     PLSR with VIP > 1 Feature Selection - TSS Modeling
Description: 
    This script implements a Partial Least Squares Regression (PLSR) pipeline
    enhanced with Variable Importance in Projection (VIP) filtering. 
    It refits the model using only spectral bands with VIP > 1 to improve 
    robustness and interpretability for Total Suspended Solids (TSS) estimation.
-------------------------------------------------------------------------------
"""

import os
import re
import logging
import joblib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

# Logging and Warnings
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
INPUT_FILE = r"path/to/your/Excel_SG_smoothed.xlsx"
OUTPUT_ROOT = r"path/to/your/output/plsr_results"

WAVELENGTH_COL = "Wavelength (nm)"
SPECTRAL_RANGE = (400, 900)
TEST_SIZE, RANDOM_STATE = 0.20, 42
EPS = 1e-6  # To prevent log(0)

def calculate_vip(pls_model):
    """
    Computes Variable Importance in Projection (VIP) scores for PLSR.
    Ref: Wold et al., (1993)
    """
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    u = pls_model.y_scores_
    p, a = w.shape
    ssy = np.sum(u**2, axis=0)
    total_ssy = np.sum(ssy)
    
    if total_ssy <= 0:
        return np.zeros(p)
        
    vip = np.sqrt(p * (w**2 @ ssy) / total_ssy)
    return vip



def export_scatter_plot(y_true_tr, y_pred_tr, y_true_te, y_pred_te, title, output_path):
    """Generates a high-fidelity scatter plot with embedded performance metrics."""
    r2_tr, rmse_tr = r2_score(y_true_tr, y_pred_tr), np.sqrt(mean_squared_error(y_true_tr, y_pred_tr))
    r2_te, rmse_te = r2_score(y_true_te, y_pred_te), np.sqrt(mean_squared_error(y_true_te, y_pred_te))
    
    fig, ax = plt.subplots(figsize=(7, 7), dpi=300)
    ax.grid(True, alpha=0.2, zorder=1)

    ax.scatter(y_true_te, y_pred_te, c="red", s=45, label=f"Test (n={len(y_true_te)})", zorder=2, alpha=0.8)
    ax.scatter(y_true_tr, y_pred_tr, c="black", s=40, label=f"Train (n={len(y_true_tr)})", zorder=3, alpha=0.9)
    
    max_val = max(max(y_true_tr), max(y_true_te)) * 1.05
    ax.plot([0, max_val], [0, max_val], "--", color="gray", zorder=1)
    
    ax.set_xlabel("Measured TSS")
    ax.set_ylabel("Predicted TSS")
    ax.set_title(title)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.legend(loc="upper left")

    metrics_text = (
        f"Train Metrics:\nR² = {r2_tr:.3f}\nRMSE = {rmse_tr:.3f}\n\n"
        f"Test Metrics:\nR² = {r2_te:.3f}\nRMSE = {rmse_te:.3f}"
    )

    ax.text(0.97, 0.03, metrics_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight='medium', bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def main():
    # 1. Workspace Initialization
    tag = datetime.now().strftime("%Y%m%d_%H%M")
    work_dir = Path(OUTPUT_ROOT) / tag
    work_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data Preprocessing
    logging.info("Preprocessing spectral data...")
    raw_df = pd.read_excel(INPUT_FILE)
    
    tmp = raw_df.copy()
    tmp[WAVELENGTH_COL] = pd.to_numeric(tmp[WAVELENGTH_COL], errors="coerce")
    tmp = tmp.dropna(subset=[WAVELENGTH_COL])
    spectral_data = tmp.groupby(WAVELENGTH_COL).mean(numeric_only=True).sort_index()

    band_names = [f"X_{int(wl)}" for wl in spectral_data.index]
    rows = []
    for col in raw_df.columns:
        if col == WAVELENGTH_COL: continue
        match = re.search(r'(?:-|_|\s)(\d+(?:\.\d+)?)\s*$', str(col))
        if match and col in spectral_data.columns:
            rows.append([float(match.group(1))] + spectral_data[col].tolist())

    df = pd.DataFrame(rows, columns=["TSS"] + band_names).dropna().reset_index(drop=True)
    X_cols = [c for c in df.columns if c.startswith("X_") and SPECTRAL_RANGE[0] <= int(c.split("_")[1]) <= SPECTRAL_RANGE[1]]
    
    X, y = df[X_cols].values, df["TSS"].values.astype(float)
    
    # Imputation and Log-Transformation
    imputer = SimpleImputer(strategy="median")
    X_imp = np.clip(imputer.fit_transform(X), EPS, None)
    X_log, y_log = np.log(X_imp), np.log(y + EPS)

    # 3. Initial PLSR for VIP Discovery
    logging.info("Running initial PLSR for VIP score calculation...")
    X_tr, X_te, y_tr, y_te = train_test_split(X_log, y_log, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    
    initial_pls = PLSRegression(n_components=7).fit(X_tr, y_tr)
    vip_scores = calculate_vip(initial_pls)

    # 4. Refit Strategy (VIP > 1.0)
    relevant_idx = np.where(vip_scores >= 1.0)[0]
    filtered_bands = [X_cols[i] for i in relevant_idx]
    
    X_tr_filtered = X_tr[:, relevant_idx]
    X_te_filtered = X_te[:, relevant_idx]

    logging.info(f"Refitting model using {len(filtered_bands)} bands with VIP >= 1.0")
    final_pls = PLSRegression(n_components=min(5, len(relevant_idx))).fit(X_tr_filtered, y_tr)

    # 5. Predictions and Inverse Scaling
    y_tr_pred = np.exp(final_pls.predict(X_tr_filtered).ravel())
    y_te_pred = np.exp(final_pls.predict(X_te_filtered).ravel())
    y_tr_real = np.exp(y_tr)
    y_te_real = np.exp(y_te)

    # 6. Evaluation and Export
    export_scatter_plot(y_tr_real, y_tr_pred, y_te_real, y_te_pred, 
                        f"PLSR Refit (VIP > 1 Strategy)", work_dir / "VIP_Refit_Scatter.png")

    # Metrics
    r2_final = r2_score(y_te_real, y_te_pred)
    rmse_final = np.sqrt(mean_squared_error(y_te_real, y_te_pred))
    mae_final = mean_absolute_error(y_te_real, y_te_pred)
    pearson_final, _ = pearsonr(y_te_real, y_te_pred)

    # Calculate Intercept for Log-Log Space
    intercept = np.mean(y_tr) - np.dot(np.mean(X_tr_filtered, axis=0), final_pls.coef_.ravel())

    # Save Detailed Report
    with open(work_dir / "performance_report.txt", "w") as f:
        f.write("PLSR VIP Performance Report\n" + "="*30 + "\n")
        f.write(f"Refit Bands (VIP > 1): {len(filtered_bands)}\n")
        f.write(f"R² (Test): {r2_final:.4f}\nRMSE (Test): {rmse_final:.4f}\n")
        f.write(f"MAE (Test): {mae_final:.4f}\nPearson: {pearson_final:.4f}\n")
        f.write(f"Log-Space Intercept: {intercept:.6f}\n")

    # 7. Model Serialization
    bundle = {
        "model": final_pls,
        "selected_bands": filtered_bands,
        "imputer": imputer,
        "EPS": EPS
    }
    joblib.dump(bundle, work_dir / "plsr_vip_final_model.pkl")
    
    logging.info(f"Pipeline complete. Results saved at: {work_dir}")

if __name__ == "__main__":
    main()