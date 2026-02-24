# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 17:01:27 2026

@author: sokaraca
"""

# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Author:      Onur Karaca
Contact:     onurkaraca87@hotmail.com
Website:     www.onurkaraca87.com
-------------------------------------------------------------------------------
Project:     Optimized Random Forest Pipeline - TSS Modeling
Description: 
    This script implements an end-to-end Machine Learning pipeline for 
    Total Suspended Solids (TSS) estimation using Random Forest. 
    Key features include:
    - Automated Feature Selection (Top-10 bands).
    - Randomized Hyperparameter Tuning.
    - Model persistence (.pkl export).
    - Interpretability via SHAP and spectral group importance analysis.
-------------------------------------------------------------------------------
"""

import os
import re
import warnings
import logging
import joblib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

# Environment & Thread Management
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# CONFIGURATION & SETTINGS
# =============================================================================
FILE_PATH = r"path/to/your/Excel_SG_smoothed.xlsx"
OUTPUT_ROOT = r"path/to/your/output/RF_Results"
METHOD_TAG = "RandomForest_Final_v2"

# Global Parameters
WL_COL = "Wavelength (nm)"
WL_RANGE = (400, 1000)
TEST_SIZE, RANDOM_STATE = 0.20, 42
CV_SPLITS, N_TOP_BANDS = 3, 10
S_TEST, S_TRAIN = 28, 30

def parse_tss_from_col(col_name):
    """Parses TSS values from column headers using regex."""
    match = re.search(r"(?:-|_|\s)(\d+(?:\.\d+)?)\s*$", str(col_name))
    return float(match.group(1)) if match else None

def main():
    # 1. Initialize Directory
    tag = datetime.now().strftime("%Y%m%d")
    output_dir = Path(OUTPUT_ROOT) / METHOD_TAG
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. Data Preparation
    logging.info("Preprocessing spectral dataset...")
    raw_df = pd.read_excel(FILE_PATH)
    tmp = raw_df.copy()
    tmp[WL_COL] = pd.to_numeric(tmp[WL_COL], errors="coerce")
    tmp = tmp.dropna(subset=[WL_COL])
    df_spectral = tmp.groupby(WL_COL).mean(numeric_only=True).sort_index()

    band_names = [f"X_{int(wl)}" for wl in df_spectral.index]
    sample_cols, tss_vals = [], []
    for col in raw_df.columns:
        if col == WL_COL: continue
        tss = parse_tss_from_col(col)
        if tss is not None and col in df_spectral.columns:
            sample_cols.append(col)
            tss_vals.append(tss)

    df = pd.DataFrame([[tss] + df_spectral[c].tolist() for c, tss in zip(sample_cols, tss_vals)], 
                      columns=["TSS"] + band_names).dropna().reset_index(drop=True)

    X_cols = [c for c in df.columns if c.startswith("X_") and WL_RANGE[0] <= int(c.split("_")[1]) <= WL_RANGE[1]]
    X_full, y = df[X_cols], df["TSS"].astype(float)

    # 3. Feature Selection
    logging.info("Calculating feature importance for band selection...")
    rf_fs = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    rf_fs.fit(X_full, y)
    feat_imp = pd.Series(rf_fs.feature_importances_, index=X_full.columns).sort_values(ascending=False)
    selected_bands = list(feat_imp.index[:N_TOP_BANDS])

    # Save Top-10 Bands List
    with open(output_dir / f"selected_bands_{tag}.txt", "w") as f:
        f.write(f"Top {N_TOP_BANDS} Selected Bands (RF)\n" + "="*35 + "\n")
        for i, band in enumerate(selected_bands, 1):
            f.write(f"{i}. {band} (Score: {feat_imp[band]:.4f})\n")

    # Plot Feature Importance
    plt.figure(figsize=(8, 5))
    sns.barplot(x=feat_imp.values[:N_TOP_BANDS], y=feat_imp.index[:N_TOP_BANDS], palette="magma")
    plt.title(f"Top {N_TOP_BANDS} Feature Importance")
    plt.tight_layout()
    plt.savefig(output_dir / f"feature_importance_{tag}.png", dpi=300)
    plt.close()

    X_top = df[selected_bands]

    # 4. Train/Test Split (Outlier Preservation)
    max_idx = y[y == y.max()].index.tolist()
    other_idx = y.index.difference(max_idx)
    X_train, X_test, y_train, y_test = train_test_split(X_top.loc[other_idx], y.loc[other_idx], 
                                                        test_size=TEST_SIZE, random_state=RANDOM_STATE)
    if max_idx:
        X_train = pd.concat([X_train, X_top.loc[max_idx]])
        y_train = pd.concat([y_train, y.loc[max_idx]])
        X_test = pd.concat([X_test, X_top.loc[max_idx]])
        y_test = pd.concat([y_test, y.loc[max_idx]])

    # 5. Optimized Hyperparameter Tuning
    logging.info("Starting Randomized Search for best parameters...")
    dist = {
        "n_estimators": [500, 800],
        "max_depth": [8, 10, 12],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 3],
        "max_features": ["sqrt"],
        "bootstrap": [True]
    }
    search = RandomizedSearchCV(RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
                                param_distributions=dist, n_iter=25, 
                                cv=KFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
                                scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, refit=True)
    search.fit(X_train, y_train)
    best_rf = search.best_estimator_
    joblib.dump(best_rf, output_dir / f"rf_model_{tag}.pkl")

    # 6. Evaluation & Visualization
    y_pred_test, y_pred_train = best_rf.predict(X_test), best_rf.predict(X_train)
    r2_t, rmse_t = r2_score(y_test, y_pred_test), np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_tr, rmse_tr = r2_score(y_train, y_pred_train), np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_t, mape_t = mean_absolute_error(y_test, y_pred_test), mean_absolute_percentage_error(y_test, y_pred_test)
    p_corr, _ = pearsonr(y_test, y_pred_test)

    # 

    # Scatter Plot
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_test, y_pred_test, c="red", s=S_TEST, label=f"Test (n={len(y_test)})", alpha=0.8, zorder=2)
    ax.scatter(y_train, y_pred_train, c="black", s=S_TRAIN, label=f"Train (n={len(y_train)})", alpha=0.9, zorder=3)
    
    lims = [y.min(), y.max()]
    ax.plot(lims, lims, "--", color="gray", zorder=1)
    ax.set_xlabel("Actual TSS")
    ax.set_ylabel("Predicted TSS")
    ax.set_title("Optimized Random Forest Performance")
    ax.legend(loc="upper left")
    
    stats_text = (f"Train Metrics:\nR² = {r2_tr:.3f}\nRMSE = {rmse_tr:.3f}\n\n"
                  f"Test Metrics:\nR² = {r2_t:.3f}\nRMSE = {rmse_t:.3f}")
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, fontweight='medium', bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / f"performance_scatter_rf_{tag}.png", dpi=300)

    # 7. SHAP and Spectral Zone Analysis
    explainer = shap.TreeExplainer(best_rf)
    shap_values = explainer.shap_values(X_train)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)

    # 

    zones = {"Blue (400-500)": 0.0, "Green (501-600)": 0.0, "Red (601-800)": 0.0}
    for bname, val in zip(X_train.columns, mean_abs_shap):
        wl = int(bname.split("_")[1])
        if 400 <= wl <= 500: zones["Blue (400-500)"] += val
        elif 501 <= wl <= 600: zones["Green (501-600)"] += val
        elif 601 <= wl <= 800: zones["Red (601-800)"] += val

    plt.figure(figsize=(6,6))
    plt.pie(zones.values(), labels=zones.keys(), autopct='%1.1f%%', startangle=140, colors=['skyblue', 'lightgreen', 'salmon'])
    plt.title("Importance Distribution by Spectral Region")
    plt.savefig(output_dir / f"importance_pie_rf_{tag}.png", dpi=300)

    # 8. Report Export
    report_path = output_dir / f"model_report_rf_{tag}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Model Summary: Random Forest\n{'='*30}\n")
        f.write(f"R² (Test): {r2_t:.4f}\nRMSE (Test): {rmse_t:.4f}\n")
        f.write(f"MAE (Test): {mae_t:.4f}\nMAPE (Test): {mape_t:.4f}\n")
        f.write(f"Pearson (Test): {p_corr:.4f}\n\n")
        f.write(f"Best Parameters: {search.best_params_}\n")

    logging.info(f"Pipeline finished. Train R2: {r2_tr:.3f} | Test R2: {r2_t:.3f}")

if __name__ == "__main__":
    main()