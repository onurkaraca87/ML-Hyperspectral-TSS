# -*- coding: utf-8 -*-
"""
Created on Tue Feb 24 17:03:32 2026

@author: sokaraca
"""

# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Author:      Onur Karaca
Contact:     onurkaraca87@hotmail.com
Website:     www.onurkaraca87.com
-------------------------------------------------------------------------------
Project:     Balanced XGBoost Pipeline - TSS Regression
Description: 
    This script implements a robust XGBoost regression pipeline for Total 
    Suspended Solids (TSS) estimation. It uses heavy L1/L2 regularization 
    to balance training/testing performance and includes SHAP-based 
    spectral importance analysis.
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
from xgboost import XGBRegressor

from sklearn.metrics import (mean_squared_error, r2_score, 
                             mean_absolute_error, mean_absolute_percentage_error)
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from scipy.stats import randint, pearsonr

# Environment & Performance Optimization
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# CONFIGURATION & HYPERPARAMETERS
# =============================================================================
INPUT_FILE = r"path/to/your/Excel_SG_smoothed.xlsx"
OUTPUT_ROOT = r"path/to/your/output/XGBoost_Results"
METHOD_TAG = "XGBoost_Final_Balanced"

# Tuning Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 3
N_TOP_BANDS = 10
WL_RANGE = (400, 1000)

def parse_concentration(column_name):
    """Extracts target concentration (TSS) from column headers via Regex."""
    match = re.search(r"(?:-|_|\s)(\d+(?:\.\d+)?)\s*$", str(column_name))
    return float(match.group(1)) if match else None

def main():
    # 1. Initialize Workspace
    tag = datetime.now().strftime("%Y%m%d")
    output_dir = Path(OUTPUT_ROOT) / METHOD_TAG
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Data Ingestion & Preprocessing
    logging.info("Reading spectral data...")
    raw_df = pd.read_excel(INPUT_FILE)
    wl_col = "Wavelength (nm)"
    
    tmp = raw_df.copy()
    tmp[wl_col] = pd.to_numeric(tmp[wl_col], errors="coerce")
    tmp = tmp.dropna(subset=[wl_col])
    spectral_pivot = tmp.groupby(wl_col).mean(numeric_only=True).sort_index()
    
    band_names = [f"X_{int(wl)}" for wl in spectral_pivot.index]
    sample_list, target_vals = [], []
    
    for col in raw_df.columns:
        if col == wl_col: continue
        val = parse_concentration(col)
        if val is not None and col in spectral_pivot.columns:
            sample_list.append(col)
            target_vals.append(val)

    df = pd.DataFrame(
        [[t] + spectral_pivot[s].tolist() for s, t in zip(sample_list, target_vals)], 
        columns=["TSS"] + band_names
    ).dropna().reset_index(drop=True)

    feat_cols = [c for c in df.columns if c.startswith("X_") and 
                 WL_RANGE[0] <= int(c.split("_")[1]) <= WL_RANGE[1]]
    X_full, y = df[feat_cols], df["TSS"].astype(float)

    # 3. Feature Selection (XGB Importance)
    logging.info("Selecting most significant bands...")
    selector = XGBRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1)
    selector.fit(X_full, y)
    importances = pd.Series(selector.feature_importances_, index=X_full.columns).sort_values(ascending=False)
    selected_features = list(importances.index[:N_TOP_BANDS])

    # 4. Train-Test Split (Ensuring max value inclusion)
    max_val_idx = y[y == y.max()].index.tolist()
    rest_idx = y.index.difference(max_val_idx)
    X_train, X_test, y_train, y_test = train_test_split(
        df[selected_features].loc[rest_idx], y.loc[rest_idx], 
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    if max_val_idx:
        X_train = pd.concat([X_train, df[selected_features].loc[max_val_idx]])
        y_train = pd.concat([y_train, y.loc[max_val_idx]])
        X_test = pd.concat([X_test, df[selected_features].loc[max_val_idx]])
        y_test = pd.concat([y_test, y.loc[max_val_idx]])

    # 5. Randomized Hyperparameter Tuning
    logging.info("Optimizing model parameters...")
    param_grid = {
        "n_estimators": randint(600, 1000),
        "max_depth": [3], # Shallow trees for better generalization
        "learning_rate": [0.01, 0.02, 0.03],
        "reg_lambda": [50, 80, 100], # Strong L2 regularization
        "reg_alpha": [1, 5, 10],     # L1 regularization
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8]
    }

    search = RandomizedSearchCV(
        XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=1),
        param_distributions=param_grid, n_iter=40, 
        cv=KFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, refit=True
    )
    search.fit(X_train, y_train)
    best_xgb = search.best_estimator_
    joblib.dump(best_xgb, output_dir / f"xgboost_balanced_model_{tag}.pkl")

    # 6. Evaluation & Visualization
    
    
    y_pred_test, y_pred_train = best_xgb.predict(X_test), best_xgb.predict(X_train)
    r2_te, rmse_te = r2_score(y_test, y_pred_test), np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_tr, rmse_tr = r2_score(y_train, y_pred_train), np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_te = mean_absolute_error(y_test, y_pred_test)
    mape_te = mean_absolute_percentage_error(y_test, y_pred_test)
    p_corr, _ = pearsonr(y_test, y_pred_test)

    # Performance Plot (Scatter)
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_test, y_pred_test, c="red", s=30, label=f"Test (n={len(y_test)})", alpha=0.8, zorder=2)
    ax.scatter(y_train, y_pred_train, c="black", s=30, label=f"Train (n={len(y_train)})", alpha=0.9, zorder=3)
    
    limits = [y.min(), y.max()]
    ax.plot(limits, limits, "--", color="gray", zorder=1)
    ax.set_xlabel("Actual TSS")
    ax.set_ylabel("Predicted TSS")
    ax.set_title("XGBoost Model: Training vs Testing")
    ax.legend(loc="upper left")
    
    stats_box = (f"Train Metrics:\nR² = {r2_tr:.3f}\nRMSE = {rmse_tr:.3f}\n\n"
                 f"Test Metrics:\nR² = {r2_te:.3f}\nRMSE = {rmse_te:.3f}")
    ax.text(0.98, 0.02, stats_box, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / f"XGB_performance_scatter_{tag}.png", dpi=300)

    # 7. SHAP Importance Pie Chart
    
    
    explainer = shap.TreeExplainer(best_xgb)
    shap_vals = explainer.shap_values(X_train)
    abs_shap = np.abs(shap_vals).mean(axis=0)
    
    zones = {"Blue (400-500)": 0.0, "Green (501-600)": 0.0, "Red (601-800)": 0.0}
    for feat, val in zip(X_train.columns, abs_shap):
        wl = int(feat.split("_")[1])
        if 400 <= wl <= 500: zones["Blue (400-500)"] += val
        elif 501 <= wl <= 600: zones["Green (501-600)"] += val
        elif 601 <= wl <= 800: zones["Red (601-800)"] += val

    plt.figure(figsize=(6, 6))
    plt.pie(zones.values(), labels=zones.keys(), autopct='%1.1f%%', startangle=140)
    plt.title("Importance Distribution by Color Group")
    plt.savefig(output_dir / f"XGB_importance_pie_{tag}.png", dpi=300)

    # 8. Report Export
    report_path = output_dir / f"performance_report_xgb_{tag}.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Model: XGBoost (Balanced Pipeline)\n" + "="*40 + "\n")
        f.write(f"R² (Test): {r2_te:.4f}\nRMSE (Test): {rmse_te:.4f}\n")
        f.write(f"MAE (Test): {mae_te:.4f}\nMAPE (Test): {mape_te:.4f}\n")
        f.write(f"Pearson Correlation: {p_corr:.4f}\n\n")
        f.write(f"Selected Bands: {', '.join([s.split('_')[1] for s in selected_features])}\n")
        f.write(f"Best Parameters: {search.best_params_}\n")

    logging.info(f"Success! Results saved to {output_dir}")

if __name__ == "__main__":
    main()