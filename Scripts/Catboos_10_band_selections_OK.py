# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 01:20:01 2025

@author: sokaraca
"""

# -*- coding: utf-8 -*-
"""
CatBoost band selection + TSS prediction (hyperspectral in situ reflectance)

Purpose
-------
End-to-end pipeline to:
1) Read an Excel table of hyperspectral reflectance (rows = wavelengths, cols = samples)
2) Parse TSS from column names (TSS value after the last hyphen/underscore/space)
3) Select wavelength range (default: 400–1000 nm)
4) Train an initial CatBoost model and rank feature importance
5) Select top-N bands (default: 10)
6) Train/test split (NO leakage)
7) Hyperparameter search (RandomizedSearchCV)
8) Evaluate (R², RMSE, MAE, MAPE, RSS, Pearson r)
9) Save outputs (plots, csv/xlsx summaries, trained model)

Input format (expected)
-----------------------
- One column: "Wavelength (nm)"
- Remaining columns: sample spectra
- Sample column names must include the TSS value at the end, e.g.:
  "TB5-33.7" or "Sample_65.03"  -> parses trailing number

Example usage
-------------
python scripts/catboost_band_selection.py \
  --input data/raw/Excel_SG_smoothed.xlsx \
  --output results/prisma \
  --method catboost_10band \
  --wl-min 400 --wl-max 1000 \
  --top-bands 10 \
  --test-size 0.2 \
  --cv-splits 5 \
  --n-iter 40 \
  --run-shap 0

Author
------
Şükrü Onur Karaca (University of Houston)
"""

import os
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Optional: seaborn for nicer plots
try:
    import seaborn as sns
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

import joblib

from scipy.stats import pearsonr, randint, uniform
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from catboost import CatBoostRegressor


# -----------------------------
# Helpers
# -----------------------------
def parse_tss_from_col(colname: str) -> float | None:
    """
    Extract trailing number at end of column name (after -/_/space).
    Example: 'TB5-33.7' -> 33.7
    """
    m = re.search(r"(?:-|_|\s)(\d+(?:\.\d+)?)\s*$", str(colname))
    return float(m.group(1)) if m else None


def nm_from_feature(colname: str) -> int:
    """Convert feature name like 'X_655' -> 655"""
    return int(str(colname).split("_")[1])


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def mape_eps(y_true, y_pred, eps=1e-6) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100.0)


# -----------------------------
# Main pipeline
# -----------------------------
def main(
    input_path: str,
    output_dir: str,
    method_tag: str = "catboost_10band",
    wl_col: str = "Wavelength (nm)",
    wl_min: int = 400,
    wl_max: int = 1000,
    top_bands: int = 10,
    test_size: float = 0.2,
    random_state: int = 42,
    cv_splits: int = 5,
    n_iter: int = 40,
    threads: int = -1,
    run_shap: bool = False,
) -> None:
    tag = datetime.now().strftime("%Y%m%d")
    out_root = Path(output_dir) / method_tag
    safe_mkdir(out_root)

    # -----------------------------
    # 1) Read Excel
    # -----------------------------
    raw_df = pd.read_excel(input_path)
    if wl_col not in raw_df.columns:
        raise ValueError(f"Expected column '{wl_col}' not found. Available: {list(raw_df.columns)}")

    # -----------------------------
    # 2) Unique wavelength index (handle duplicates)
    # -----------------------------
    tmp = raw_df.copy()
    tmp[wl_col] = pd.to_numeric(tmp[wl_col], errors="coerce")
    tmp = tmp.dropna(subset=[wl_col])

    df_idx = (
        tmp.groupby(wl_col, as_index=True)
           .mean(numeric_only=True)
           .sort_index()
    )
    wavelengths = df_idx.index.to_numpy()
    band_names = [f"X_{int(wl)}" for wl in wavelengths]

    # -----------------------------
    # 3) Parse TSS from column names
    # -----------------------------
    sample_cols, tss_values = [], []
    for col in raw_df.columns:
        if col == wl_col:
            continue
        tss = parse_tss_from_col(col)
        if tss is not None and col in df_idx.columns:
            sample_cols.append(col)
            tss_values.append(tss)

    if len(sample_cols) < 5:
        raise ValueError(
            "Too few sample columns were parsed. "
            "Check that sample column names end with '-<TSS>' or '_<TSS>' or ' <TSS>'."
        )

    # -----------------------------
    # 4) Build samples x bands matrix
    # -----------------------------
    rows = []
    for col, tss in zip(sample_cols, tss_values):
        vec = df_idx[col].to_numpy()
        if np.all(np.isnan(vec)):
            continue
        rows.append([tss] + vec.tolist())

    df = pd.DataFrame(rows, columns=["TSS"] + band_names).dropna(axis=0).reset_index(drop=True)

    # -----------------------------
    # 5) Select wavelength range
    # -----------------------------
    X_cols_full = [c for c in df.columns if c.startswith("X_")]
    X_cols_full = [c for c in X_cols_full if wl_min <= nm_from_feature(c) <= wl_max]

    X_full = df[X_cols_full].copy()
    y = df["TSS"].astype(float).copy()

    print(f"[INFO] Samples: {len(df)} | Bands ({wl_min}–{wl_max} nm): {len(X_cols_full)}")

    # -----------------------------
    # 6) Initial CatBoost for feature importance
    # -----------------------------
    print("[INFO] Initial CatBoost fit (feature importance)...")
    cat_init = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        iterations=800,
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=10.0,
        subsample=0.8,
        rsm=0.8,
        random_seed=random_state,
        thread_count=threads,
        bootstrap_type="Bernoulli",
        verbose=False,
        allow_writing_files=False,
    )
    cat_init.fit(X_full, y)

    imp = cat_init.get_feature_importance(type="PredictionValuesChange")
    feat_imp = pd.Series(imp, index=list(X_full.columns)).sort_values(ascending=False)

    # Save full importance
    full_imp_csv = out_root / f"feature_importance_full_{method_tag}_{tag}.csv"
    feat_imp.to_csv(full_imp_csv, header=["Importance"])
    print(f"[INFO] Saved importance -> {full_imp_csv}")

    # Plot top-N importance
    topN = min(top_bands, len(feat_imp))
    top_feat = feat_imp.head(topN)

    plt.figure(figsize=(7, 5))
    if _HAS_SEABORN:
        sns.barplot(x=top_feat.values[::-1], y=top_feat.index[::-1], orient="h")
    else:
        plt.barh(top_feat.index[::-1], top_feat.values[::-1])
    plt.title(f"Top {topN} Wavelengths (CatBoost Importance)")
    plt.xlabel("Importance (PredictionValuesChange)")
    plt.ylabel("Wavelength band")
    plt.tight_layout()
    feat_png = out_root / f"feature_importance_top{topN}_{method_tag}_{tag}.png"
    plt.savefig(feat_png, dpi=300)
    plt.close()

    # Select exactly top-N bands
    selected_bands = list(top_feat.index)
    X_top = df[selected_bands].copy()
    print(f"[INFO] Selected bands (top {topN}): {selected_bands}")

    # -----------------------------
    # 7) Train/test split (NO leakage)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_top, y, test_size=test_size, random_state=random_state
    )
    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")

    # -----------------------------
    # 8) RandomizedSearchCV
    # -----------------------------
    print("[INFO] RandomizedSearchCV (CatBoost)...")
    dist = {
        "iterations": randint(500, 1500),
        "depth": randint(4, 10),
        "learning_rate": uniform(0.01, 0.19),
        "l2_leaf_reg": uniform(1.0, 9.0),
        "subsample": uniform(0.6, 0.4),
        "rsm": uniform(0.6, 0.4),
        "random_strength": uniform(0.0, 2.0),
        "border_count": randint(64, 256),
    }
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        CatBoostRegressor(
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=random_state,
            thread_count=threads,
            bootstrap_type="Bernoulli",
            verbose=False,
            allow_writing_files=False,
        ),
        param_distributions=dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=1,  # CatBoost internal threading
        verbose=1,
        random_state=random_state,
    )

    cv_df = None
    try:
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_cv_rmse = -search.best_score_

        hp_txt = out_root / f"best_hyperparameters_{method_tag}_{tag}.txt"
        with open(hp_txt, "w", encoding="utf-8") as f:
            f.write("Best parameters (RandomizedSearchCV - CatBoost)\n")
            for k, v in best_params.items():
                f.write(f"{k}: {v}\n")
            f.write(f"\nBest CV RMSE: {best_cv_rmse:.6f} (CV={cv_splits})\n")

        cv_df = pd.DataFrame(search.cv_results_)
        cv_df["rmse"] = -cv_df["mean_test_score"]
        cv_df = cv_df.sort_values("rmse")
        cv_csv = out_root / f"cv_results_{method_tag}_{tag}.csv"
        cv_df.to_csv(cv_csv, index=False)

        print(f"[INFO] Best CV RMSE: {best_cv_rmse:.3f} | Saved -> {hp_txt}")
    except Exception as e:
        warnings.warn(f"RandomizedSearchCV failed: {e}\nUsing initial model.")
        best_model = cat_init

    # -----------------------------
    # 9) Evaluation
    # -----------------------------
    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    mape = mape_eps(y_test, y_pred)
    rss = float(np.sum((np.asarray(y_test) - np.asarray(y_pred)) ** 2))
    pear, _ = pearsonr(y_test, y_pred)

    print(f"[INFO] Test: R²={r2:.3f} RMSE={rmse:.3f} MAE={mae:.3f} MAPE={mape:.2f}% r={pear:.3f}")

    # Save predictions
    pred_df = pd.DataFrame({"Actual_TSS": y_test.to_numpy(), "Predicted_TSS": np.asarray(y_pred)})
    pred_csv = out_root / f"actual_vs_predicted_{method_tag}_{tag}.csv"
    pred_df.to_csv(pred_csv, index=False)

    # Scatter plot
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, y_pred, s=25)
    lims = [min(y_test.min(), np.min(y_pred)), max(y_test.max(), np.max(y_pred))]
    plt.plot(lims, lims, "r--", linewidth=1)
    plt.xlabel("Actual TSS (Test)")
    plt.ylabel("Predicted TSS (Test)")
    plt.title(f"CatBoost ({topN} bands)\nR²={r2:.3f} RMSE={rmse:.2f} r={pear:.3f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    scatter_png = out_root / f"scatter_{method_tag}_{tag}.png"
    plt.savefig(scatter_png, dpi=300)
    plt.close()

    # Save metrics
    scores_txt = out_root / f"scores_{method_tag}_{tag}.txt"
    with open(scores_txt, "w", encoding="utf-8") as f:
        f.write(f"Number of bands: {topN}\n")
        f.write(f"Selected bands: {selected_bands}\n\n")
        f.write(f"R2 (Test): {r2:.6f}\n")
        f.write(f"RMSE (Test): {rmse:.6f}\n")
        f.write(f"MAE (Test): {mae:.6f}\n")
        f.write(f"MAPE (Test): {mape:.6f}\n")
        f.write(f"RSS (Test): {rss:.6f}\n")
        f.write(f"Pearson r (Test): {pear:.6f}\n")

    # Save model
    model_path = out_root / f"catboost_{topN}bands_model_{method_tag}_{tag}.pkl"
    joblib.dump(best_model, model_path)

    # Excel summary
    excel_path = out_root / f"CatBoost_results_summary_{method_tag}_{tag}.xlsx"
    scores_df = pd.DataFrame(
        {
            "Metric": ["Number of bands", "Selected bands", "R2", "RMSE", "MAE", "MAPE (%)", "RSS", "Pearson r"],
            "Value": [topN, ", ".join(selected_bands), r2, rmse, mae, mape, rss, pear],
        }
    )
    sel_imp_df = feat_imp.loc[selected_bands].reset_index()
    sel_imp_df.columns = ["Band", "Importance"]

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        scores_df.to_excel(writer, sheet_name="Scores", index=False)
        sel_imp_df.to_excel(writer, sheet_name="Selected_Bands", index=False)
        pred_df.to_excel(writer, sheet_name="Actual_vs_Predicted", index=False)
        if cv_df is not None:
            cv_df.to_excel(writer, sheet_name="CV_Results", index=False)

    print(f"[INFO] Saved outputs -> {out_root}")
    print(f"[INFO] Model -> {model_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CatBoost band selection for TSS using hyperspectral reflectance.")
    parser.add_argument("--input", required=True, help="Path to input Excel file (e.g., data/raw/Excel_SG_smoothed.xlsx)")
    parser.add_argument("--output", default="results", help="Output directory root (default: results)")
    parser.add_argument("--method", default="catboost_10band", help="Method tag for outputs")
    parser.add_argument("--wl-min", type=int, default=400, help="Min wavelength (nm)")
    parser.add_argument("--wl-max", type=int, default=1000, help="Max wavelength (nm)")
    parser.add_argument("--top-bands", type=int, default=10, help="Number of top bands to select")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--cv-splits", type=int, default=5, help="CV folds")
    parser.add_argument("--n-iter", type=int, default=40, help="Random search iterations")
    parser.add_argument("--threads", type=int, default=-1, help="CatBoost thread_count")
    parser.add_argument("--run-shap", type=int, default=0, help="Run SHAP (0/1). Requires shap.")

    args = parser.parse_args()

    main(
        input_path=args.input,
        output_dir=args.output,
        method_tag=args.method,
        wl_min=args.wl_min,
        wl_max=args.wl_max,
        top_bands=args.top_bands,
        test_size=args.test_size,
        random_state=args.random_state,
        cv_splits=args.cv_splits,
        n_iter=args.n_iter,
        threads=args.threads,
        run_shap=bool(args.run_shap),
    )
