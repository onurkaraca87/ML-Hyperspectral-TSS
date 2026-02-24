# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
Author:      Onur Karaca
Contact:     +1 (346) 719-259 | onurkaraca87@hotmail.com
Website:     www.onurkaraca87.com
Profiles:    LinkedIn: linkedin.com/in/onurkaraca | GitHub: github.com/onurkaraca
-------------------------------------------------------------------------------
Project:     LightGBM Balanced Pipeline for TSS Prediction
Description: 
    This script implements a robust machine learning pipeline for estimating 
    Total Suspended Solids (TSS) from hyperspectral data. It includes:
    1. Automated feature selection based on LightGBM gain.
    2. Hyperparameter optimization using RandomizedSearchCV.
    3. Regularized model training to prevent overfitting.
    4. Comprehensive evaluation and SHAP-based importance visualization.
-------------------------------------------------------------------------------
"""

import os
import re
import warnings
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
from lightgbm import LGBMRegressor

from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from scipy.stats import randint, uniform, pearsonr

# Environment setup to control threading
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# =============================================================================
# CONFIGURATION & HYPERPARAMETERS
# =============================================================================
FILE_PATH = r"path/to/your/input_data/Excel_SG_smoothed.xlsx"
OUTPUT_ROOT = r"path/to/your/output_directory"
METHOD_TAG = "LGBM_Balanced_TSS"

# Hyperparameters and Constants
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 3
N_TOP_BANDS = 10
WL_RANGE = (400, 1000)

# Visual Settings
S_TEST, S_TRAIN = 28, 30
COLOR_TRAIN, COLOR_TEST = "black", "red"

def parse_tss_from_column(column_name):
    """Extracts numeric TSS value from column strings using regex."""
    match = re.search(r"(?:-|_|\s)(\d+(?:\.\d+)?)\s*$", str(column_name))
    return float(match.group(1)) if match else None

def prepare_dataset(path):
    """Reads raw spectral data and transforms it into a TSS-Feature dataframe."""
    raw_df = pd.read_excel(path)
    wl_col = "Wavelength (nm)"
    
    # Process wavelengths
    tmp = raw_df.copy()
    tmp[wl_col] = pd.to_numeric(tmp[wl_col], errors="coerce")
    tmp = tmp.dropna(subset=[wl_col])
    df_spectral = tmp.groupby(wl_col).mean(numeric_only=True).sort_index()
    
    band_names = [f"X_{int(wl)}" for wl in df_spectral.index]
    
    sample_cols, tss_values = [], []
    for col in raw_df.columns:
        if col == wl_col: continue
        tss = parse_tss_from_column(col)
        if tss is not None and col in df_spectral.columns:
            sample_cols.append(col)
            tss_values.append(tss)
            
    df = pd.DataFrame(
        [[tss] + df_spectral[c].tolist() for c, tss in zip(sample_cols, tss_values)],
        columns=["TSS"] + band_names
    ).dropna().reset_index(drop=True)
    
    return df

def main():
    # 1. Initialize Paths
    tag = datetime.now().strftime("%Y%m%d")
    final_output_dir = Path(OUTPUT_ROOT) / METHOD_TAG
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Data Preparation
    logging.info("Preparing dataset...")
    df = prepare_dataset(FILE_PATH)
    
    # Filter wavelengths within range
    X_cols = [c for c in df.columns if c.startswith("X_") and WL_RANGE[0] <= int(c.split("_")[1]) <= WL_RANGE[1]]
    X_full, y = df[X_cols], df["TSS"].astype(float)
    
    # 3. Feature Selection
    logging.info("Performing feature selection...")
    selector = LGBMRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=1, verbose=-1)
    selector.fit(X_full, y)
    
    feat_importance = pd.Series(selector.feature_importances_, index=X_full.columns).sort_values(ascending=False)
    selected_bands = list(feat_importance.index[:N_TOP_BANDS])
    X_top = df[selected_bands]
    
    # 4. Train-Test Split with Outlier Preservation
    max_idx = y[y == y.max()].index.tolist()
    other_idx = y.index.difference(max_idx)
    X_train, X_test, y_train, y_test = train_test_split(
        X_top.loc[other_idx], y.loc[other_idx], 
        test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    
    # Ensure extreme values are present in both sets if possible
    if max_idx:
        X_train = pd.concat([X_train, X_top.loc[max_idx]])
        y_train = pd.concat([y_train, y.loc[max_idx]])
        X_test = pd.concat([X_test, X_top.loc[max_idx]])
        y_test = pd.concat([y_test, y.loc[max_idx]])

    # 5. Hyperparameter Tuning
    logging.info("Starting RandomizedSearchCV for LightGBM...")
    param_dist = {
        "n_estimators": randint(500, 1000),
        "max_depth": [3, 4],
        "num_leaves": [7, 15],
        "learning_rate": [0.01, 0.02, 0.03],
        "lambda_l2": [50, 100, 150],
        "lambda_l1": [1, 5, 10],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8]
    }

    search = RandomizedSearchCV(
        LGBMRegressor(random_state=RANDOM_STATE, n_jobs=1, importance_type='gain', verbose=-1),
        param_distributions=param_dist, 
        n_iter=40, 
        cv=KFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
        scoring="neg_root_mean_squared_error", 
        random_state=RANDOM_STATE, 
        refit=True
    )
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    
    # Save Model
    joblib.dump(best_model, final_output_dir / f"lgbm_balanced_model_{tag}.pkl")

    # 6. Evaluation & Visualization
    y_pred_test = best_model.predict(X_test)
    y_pred_train = best_model.predict(X_train)
    
    # Metrics calculation
    r2_t, rmse_t = r2_score(y_test, y_pred_test), np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2_tr, rmse_tr = r2_score(y_train, y_pred_train), np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_t = mean_absolute_error(y_test, y_pred_test)
    mape_t = mean_absolute_percentage_error(y_test, y_pred_test)
    p_corr, _ = pearsonr(y_test, y_pred_test)

    # Plot Scatter
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_test, y_pred_test, c=COLOR_TEST, s=S_TEST, label=f"Test (n={len(y_test)})", alpha=0.8, zorder=2)
    ax.scatter(y_train, y_pred_train, c=COLOR_TRAIN, s=S_TRAIN, label=f"Train (n={len(y_train)})", alpha=0.9, zorder=3)
    
    limits = [y.min(), y.max()]
    ax.plot(limits, limits, "--", color="gray", zorder=1)
    ax.set_xlabel("Actual TSS (mg/L)")
    ax.set_ylabel("Predicted TSS (mg/L)")
    ax.set_title("LightGBM Model: Prediction Accuracy")
    ax.legend()
    
    stats_text = (f"Train R²: {r2_tr:.3f}\nTrain RMSE: {rmse_tr:.3f}\n\n"
                  f"Test R²: {r2_t:.3f}\nTest RMSE: {rmse_t:.3f}")
    ax.text(0.95, 0.05, stats_text, transform=ax.transAxes, ha="right", va="bottom",
            bbox=dict(boxstyle="round", fc="white", ec="black", alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(final_output_dir / f"LGBM_performance_scatter_{tag}.png", dpi=600)

    # 7. Performance Report Export
    report_path = final_output_dir / f"performance_report_{tag}.txt"
    with open(report_path, "w") as f:
        f.write(f"TSS Prediction Model Report - LightGBM\n{'='*40}\n")
        f.write(f"Test R2: {r2_t:.4f}\nTest RMSE: {rmse_t:.4f}\n")
        f.write(f"Test MAE: {mae_t:.4f}\nTest MAPE: {mape_t:.4f}\n")
        f.write(f"Pearson Correlation: {p_corr:.4f}\n\n")
        f.write(f"Best Parameters: {search.best_params_}\n")
        f.write(f"Selected Wavelengths: {', '.join([b.split('_')[1] for b in selected_bands])}\n")

    # 8. Importance Visualization (SHAP)
    explainer = shap.TreeExplainer(best_model)
    shap_v = explainer.shap_values(X_train)
    
    # Importance by Color Bands
    zones = {"Blue (400-500nm)": 0.0, "Green (501-600nm)": 0.0, "Red (601-800nm)": 0.0}
    mean_shap = np.abs(shap_v).mean(axis=0)
    for band, val in zip(X_train.columns, mean_shap):
        w = int(band.split("_")[1])
        if 400 <= w <= 500: zones["Blue (400-500nm)"] += val
        elif 501 <= w <= 600: zones["Green (501-600nm)"] += val
        elif 601 <= w <= 800: zones["Red (601-800nm)"] += val

    plt.figure(figsize=(6,6))
    plt.pie(zones.values(), labels=zones.keys(), autopct='%1.1f%%', startangle=140)
    plt.title("Spectral Importance Distribution")
    plt.savefig(final_output_dir / f"LGBM_importance_pie_{tag}.png", dpi=300)

    logging.info(f"Pipeline finished successfully. Results saved in: {final_output_dir}")

if __name__ == "__main__":
    main()