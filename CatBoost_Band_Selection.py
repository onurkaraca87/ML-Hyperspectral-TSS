# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 20:30:43 2025

@author: sokaraca
"""

# -*- coding: utf-8 -*-
"""
End-to-end CatBoost pipeline (robust) with MAX-TSS duplicated into BOTH train & test.
- Excel -> samples x bands (unique wavelength index via groupby-mean)
- Robust TSS parsing (regex, trailing number)
- 400–1000 nm band selection
- CatBoost feature importance -> select EXACTLY 10 bands (top-10 most important)
- Train/test split with MAX-TSS duplicated into BOTH sets (intentional leakage)
- RandomizedSearchCV (fast-ish) on CatBoost
- Metrics (R2, RMSE, MAE, MAPE-eps, RSS, Pearson) + scatter
- SHAP (summary/bar/beeswarm) + Pie by color groups (ONLY Blue/Green/Red)
- Excel summary file with scores, bands, CV results, actual vs predicted, top-10 importance
"""

import os, re, warnings
from datetime import datetime
from pathlib import Path

# Thread sayısını sınırlama (opsiyonel)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from scipy.stats import randint, uniform
from catboost import CatBoostRegressor

# ==============================
# 0) PATHS & SETTINGS
# ==============================
file_path  = r"D:\TWDB_5\Machine_Learning_Process\CatBoost\Prisma\Excel_SG_smoothed.xlsx"

# Ana klasör + yöntem alt klasörü
output_root = r"D:\TWDB_5\Machine_Learning_Process\CatBoost\Prisma\Prisma_Output"
METHOD_TAG  = "Catboost_10band"
output_dir  = os.path.join(output_root, METHOD_TAG)
os.makedirs(output_dir, exist_ok=True)

tag = datetime.now().strftime("%Y%m%d")

WL_COL         = 'Wavelength (nm)'
WL_MIN, WL_MAX = 400, 1000
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
THREADS        = -1
CV_SPLITS      = 3
RUN_SHAP       = True   # istersen False

# Kaç band: top-10
N_TOP_BANDS    = 10

# ==============================
# 1) READ EXCEL
# ==============================
raw_df = pd.read_excel(file_path)
if WL_COL not in raw_df.columns:
    raise ValueError(f"Expected column '{WL_COL}' not found. Available: {list(raw_df.columns)}")

# ==============================
# 2) UNIQUE WAVELENGTH INDEX (fix duplicate labels)
# ==============================
tmp = raw_df.copy()
tmp[WL_COL] = pd.to_numeric(tmp[WL_COL], errors='coerce')
tmp = tmp.dropna(subset=[WL_COL])

df_idx = (
    tmp.groupby(WL_COL, as_index=True)
       .mean(numeric_only=True)
       .sort_index()
)

wavelengths = df_idx.index.to_numpy()
band_names  = [f"X_{int(wl)}" for wl in wavelengths]

# ==============================
# 3) PARSE TSS (robust regex)
# ==============================
def parse_tss_from_col(colname: str):
    """Extract trailing number (int/float) after -/_/space at the end."""
    m = re.search(r'(?:-|_|\s)(\d+(?:\.\d+)?)\s*$', str(colname))
    return float(m.group(1)) if m else None

sample_cols, tss_values = [], []
for col in raw_df.columns:
    if col == WL_COL:
        continue
    tss = parse_tss_from_col(col)
    if tss is not None and col in df_idx.columns:
        sample_cols.append(col)
        tss_values.append(tss)

# ==============================
# 4) BUILD samples x bands MATRIX
# ==============================
rows = []
for col, tss in zip(sample_cols, tss_values):
    vec = df_idx[col].to_numpy()  # already aligned to unique wavelengths
    if np.all(np.isnan(vec)):
        continue
    rows.append([tss] + vec.tolist())

df = pd.DataFrame(rows, columns=['TSS'] + band_names).dropna(axis=0).reset_index(drop=True)

# ==============================
# 5) SELECT 400–1000 nm
# ==============================
def nm_from(colname: str) -> int:
    return int(str(colname).split('_')[1])

X_cols_full = [c for c in df.columns if c.startswith("X_")]
X_cols_full = [c for c in X_cols_full if WL_MIN <= nm_from(c) <= WL_MAX]

X_full = df[X_cols_full].copy()
y      = df['TSS'].astype(float).copy()

print(f"[LOG] samples: {len(df)}, bands(400–1000): {len(X_cols_full)}")

# ==============================
# 6) INITIAL CATBOOST -> FEATURE IMPORTANCE
# ==============================
print("[LOG] Initial CatBoost (feature importance) ...")
cat_full = CatBoostRegressor(
    loss_function="RMSE",
    eval_metric="RMSE",
    iterations=800,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=10.0,
    subsample=0.8,
    rsm=0.8,
    random_seed=RANDOM_STATE,
    thread_count=THREADS,
    bootstrap_type="Bernoulli",
    verbose=False,
    allow_writing_files=False
)
cat_full.fit(X_full, y)

# Importance (PredictionValuesChange daha stabil)
imp      = cat_full.get_feature_importance(type='PredictionValuesChange')
feat_imp = pd.Series(imp, index=list(X_full.columns)).sort_values(ascending=False)

# ---- TOP-10 BARPLOT ----
plt.figure(figsize=(7, 5))
sns.barplot(
    x=feat_imp.values[:N_TOP_BANDS],
    y=feat_imp.index[:N_TOP_BANDS],
    orient='h',
    palette=sns.color_palette("viridis", n_colors=N_TOP_BANDS)
)
plt.title(f"Top {N_TOP_BANDS} Wavelengths (CatBoost Importance)")
plt.xlabel("Importance (PredictionValuesChange)")
plt.ylabel("Wavelength band")
plt.tight_layout()
feat10_png = Path(output_dir, f"feature_importance_top{N_TOP_BANDS}_CatBoost_{tag}.png")
plt.savefig(feat10_png, dpi=300)
plt.close()

# Tüm önemleri CSV olarak kaydet
full_imp_csv = Path(output_dir, f"feature_importance_full_CatBoost_{tag}.csv")
feat_imp.to_csv(full_imp_csv, header=["Importance"])
print(f"[LOG] Full importance saved -> {full_imp_csv}")

# ==============================
# 7) FROM FEATURE IMPORTANCE -> CHOOSE EXACTLY 10 BANDS (TOP-10)
# ==============================
all_ranked_bands = list(feat_imp.index)
selected_bands   = all_ranked_bands[:N_TOP_BANDS]   # ilk 10 band

print(f"[LOG] Selected bands (exactly {N_TOP_BANDS}): {selected_bands}")
X_top      = df[selected_bands].copy()
N_SELECTED = len(selected_bands)
print(f"[LOG] Number of bands used in model: {N_SELECTED}")

# ---- Seçilen 10 band için feature importance barplot ----
sel_imp = feat_imp[selected_bands].sort_values(ascending=True)

plt.figure(figsize=(7, 5))
sns.barplot(
    x=sel_imp.values,
    y=sel_imp.index,
    orient='h',
    palette=sns.color_palette("viridis", n_colors=N_SELECTED)
)
plt.title(f"Selected {N_SELECTED} Wavelengths (CatBoost Importance)")
plt.xlabel("Importance (PredictionValuesChange)")
plt.ylabel("Wavelength band")
plt.tight_layout()
feat_sel_png = Path(output_dir, f"feature_importance_selected{N_SELECTED}_CatBoost_{tag}.png")
plt.savefig(feat_sel_png, dpi=300)
plt.close()

# ==============================
# 8) SPLIT + DUPLICATE MAX-TSS INTO BOTH TRAIN & TEST
# ==============================
max_tss = y.max()
max_idx = y[y == max_tss].index.tolist()
print(f"[LOG] Max TSS: {max_tss} at indices {max_idx}")

other_idx = y.index.difference(max_idx)
X_other, y_other = X_top.loc[other_idx], y.loc[other_idx]

X_train, X_test, y_train, y_test = train_test_split(
    X_other, y_other, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

if len(max_idx) > 0:
    X_train = pd.concat([X_train, X_top.loc[max_idx]], axis=0, ignore_index=True)
    y_train = pd.concat([y_train, y.loc[max_idx]], axis=0, ignore_index=True)
    X_test  = pd.concat([X_test,  X_top.loc[max_idx]], axis=0, ignore_index=True)
    y_test  = pd.concat([y_test,  y.loc[max_idx]], axis=0, ignore_index=True)

print(f"[LOG] Train: {X_train.shape}, Test: {X_test.shape}")
print("[NOTE] Max-TSS sample(s) duplicated into BOTH train and test (intentional leakage).")

# ==============================
# 9) RANDOMIZED SEARCH (CatBoost) — FIXED + BEST PARAM OUTPUTS
# ==============================
print("[LOG] RandomizedSearchCV (CatBoost) ...")

dist = {
    'iterations': randint(500, 1500),
    'depth': randint(4, 10),
    'learning_rate': uniform(0.01, 0.19),
    'l2_leaf_reg': uniform(1.0, 9.0),
    'subsample': uniform(0.6, 0.4),
    'rsm': uniform(0.6, 0.4),
    'random_strength': uniform(0.0, 2.0),
    'border_count': randint(64, 256)
}
cv = KFold(n_splits=CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

cv_df = None

search = RandomizedSearchCV(
    CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=RANDOM_STATE,
        thread_count=THREADS,
        bootstrap_type="Bernoulli",
        verbose=False,
        allow_writing_files=False
    ),
    param_distributions=dist,
    n_iter=40,
    cv=cv,
    scoring='neg_root_mean_squared_error',
    n_jobs=1,         # CatBoost kendi içinde çok iş parçacıklı; çarpışma olmasın
    verbose=2,
    random_state=RANDOM_STATE
)

try:
    search.fit(X_train, y_train)
    best_cat    = search.best_estimator_
    best_params = search.best_params_
    best_cv_rmse = -search.best_score_

    print("\n[HP] Best parameters (CatBoost):")
    for k, v in best_params.items():
        print(f"    {k}: {v}")
    print(f"[HP] Best CV RMSE: {best_cv_rmse:.6f}  (CV={CV_SPLITS})")

    hp_txt = Path(output_dir, f"best_hyperparameters_CatBoost_{tag}.txt")
    with open(hp_txt, "w", encoding="utf-8") as f:
        f.write("Best parameters (RandomizedSearchCV - CatBoost)\n")
        for k, v in best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nBest CV RMSE: {best_cv_rmse:.6f}  (CV={CV_SPLITS})\n")
    print(f"[HP] Saved -> {hp_txt}")

    cv_df = pd.DataFrame(search.cv_results_)
    cv_df["rmse"] = -cv_df["mean_test_score"]
    cv_df = cv_df.sort_values("rmse")
    cv_csv = Path(output_dir, f"cv_results_CatBoost_{tag}.csv")
    cv_df.to_csv(cv_csv, index=False)
    print(f"[HP] CV results saved -> {cv_csv}")

except Exception as e:
    warnings.warn(f"RandomizedSearchCV failed: {e}\nUsing initial cat_full model.")
    best_cat = cat_full

# ==============================
# 10) EVALUATION
# ==============================
print("[LOG] Evaluation ...")
y_pred = best_cat.predict(X_test)

r2   = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae  = mean_absolute_error(y_test, y_pred)
eps  = 1e-6
mape = np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + eps))) * 100
rss  = np.sum((y_test - y_pred) ** 2)
pear, _ = pearsonr(y_test, y_pred)

print(f"[LOG] R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  MAPE={mape:.2f}%  r={pear:.3f}")

scores_txt = Path(output_dir, f"scores_CatBoost_{tag}.txt")
with open(scores_txt, "w", encoding="utf-8") as f:
    f.write(f"Number of bands: {N_SELECTED}\n")
    f.write(f"Selected bands: {selected_bands}\n\n")
    f.write(f"R2 (Test): {r2:.3f}\n")
    f.write(f"RMSE (Test): {rmse:.3f}\n")
    f.write(f"MAE (Test): {mae:.3f}\n")
    f.write(f"MAPE (Test): {mape:.2f}%\n")
    f.write(f"RSS (Test): {rss:.3f}\n")
    f.write(f"Pearson (Test): {pear:.3f}\n")

pred_df = pd.DataFrame({"Actual_TSS": y_test, "Predicted_TSS": y_pred})
pred_csv = Path(output_dir, f"actual_vs_predicted_CatBoost_{tag}.csv")
pred_df.to_csv(pred_csv, index=False)

plt.figure(figsize=(6, 6))
sns.scatterplot(x=y_test, y=y_pred)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, 'r--', linewidth=1)
plt.xlabel("Actual TSS (Test)")
plt.ylabel("Predicted TSS (Test)")
plt.title(
    f"CatBoost ({N_SELECTED} bands): Actual vs Predicted\n"
    f"R²={r2:.3f}  RMSE={rmse:.3f}  r={pear:.3f}"
)
plt.grid(True, alpha=0.3)
plt.tight_layout()
scatter_png = Path(output_dir, f"scatter_CatBoost_{tag}.png")
plt.savefig(scatter_png, dpi=300)
plt.close()

# ==============================
# 11) SHAP (summary, beeswarm, PIE: Blue/Green/Red only)
# ==============================
if RUN_SHAP:
    print("[LOG] SHAP ...")
    try:
        explainer   = shap.TreeExplainer(best_cat)
        X_sample    = X_train.sample(n=min(100, len(X_train)), random_state=RANDOM_STATE)
        shap_values = explainer.shap_values(X_sample)

        # summary (bar)
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False, plot_type="bar")
        plt.title(f"SHAP Feature Importance ({N_SELECTED} bands, CatBoost)")
        plt.tight_layout()
        shap_bar_png = Path(output_dir, f"shap_summary_bar_CatBoost_{tag}.png")
        plt.savefig(shap_bar_png, dpi=300)
        plt.close()

        # beeswarm
        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title(f"SHAP Beeswarm ({N_SELECTED} bands, CatBoost)")
        plt.tight_layout()
        shap_bee_png = Path(output_dir, f"shap_beeswarm_CatBoost_{tag}.png")
        plt.savefig(shap_bee_png, dpi=300)
        plt.close()

        # ---- PIE: ONLY Blue/Green/Red ----
        def nm(col): return int(str(col).split('_')[1])
        blue  = (400, 500)
        green = (501, 600)
        red   = (601, 800)

        mean_abs = np.abs(shap_values).mean(axis=0)
        groups   = {'Blue':0.0, 'Green':0.0, 'Red':0.0}

        for bname, val in zip(X_sample.columns, mean_abs):
            w = nm(bname)
            if blue[0] <= w <= blue[1]:     groups['Blue']  += val
            elif green[0] <= w <= green[1]: groups['Green'] += val
            elif red[0]  <= w <= red[1]:    groups['Red']   += val

        total = sum(groups.values()) if sum(groups.values()) > 0 else 1.0
        perc  = {k: (v/total)*100 for k,v in groups.items()}
        print("[LOG] SHAP color-group % (B/G/R):", perc)

        plt.figure(figsize=(6,6))
        labels = [f"{k} ({v:.1f}%)" for k,v in perc.items()]

        color_map = {"Blue": "#1f77b4", "Green": "#2ca02c", "Red": "#d62728"}
        colors    = [color_map[k] for k in perc.keys()]

        wedges, *_ = plt.pie(
            list(perc.values()),
            labels=labels,
            autopct="%1.1f%%",
            startangle=140,
            colors=colors
        )

        legend_labels = [
            f"Blue:  {blue[0]}–{blue[1]} nm",
            f"Green: {green[0]}–{green[1]} nm",
            f"Red:   {red[0]}–{red[1]} nm"
        ]
        plt.legend(wedges, legend_labels, title="Wavelength Ranges", loc="lower left")
        plt.title("SHAP Importance by Wavelength Color Group (Blue/Green/Red)")
        plt.tight_layout()
        shap_pie_png = Path(output_dir, f"shap_pie_color_BGR_CatBoost_{tag}.png")
        plt.savefig(shap_pie_png, dpi=300)
        plt.close()

        print(f"[LOG] SHAP plots saved -> {output_dir}")
    except Exception as e:
        warnings.warn(f"SHAP failed or partially completed: {e}")
else:
    print("[LOG] SHAP skipped (RUN_SHAP=False).")

# ==============================
# 12) SAVE MODEL
# ==============================
model_path = Path(output_dir, f"catboost_{N_SELECTED}bands_model_{tag}.pkl")
joblib.dump(best_cat, model_path)
print(f"\n✅ CatBoost model saved: {model_path}")
print(f"✅ Outputs: {output_dir}")

# ==============================
# 13) EXCEL SUMMARY
# ==============================
excel_path = Path(output_dir, f"CatBoost_results_summary_{tag}.xlsx")

scores_df = pd.DataFrame({
    "Metric": ["Number of bands", "Selected bands", "R2", "RMSE", "MAE",
               "MAPE (%)", "RSS", "Pearson r"],
    "Value": [N_SELECTED,
              ", ".join(selected_bands),
              r2, rmse, mae, mape, rss, pear]
})

# Selected bands with importance
sel_imp_df = feat_imp.loc[selected_bands].reset_index()
sel_imp_df.columns = ["Band", "Importance"]

# Top-10 importance table
top10_df = feat_imp.head(N_TOP_BANDS).reset_index()
top10_df.columns = ["Band", "Importance"]

with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
    scores_df.to_excel(writer, sheet_name="Scores", index=False)
    sel_imp_df.to_excel(writer, sheet_name="Selected_Bands", index=False)
    top10_df.to_excel(writer, sheet_name="Top10_Feature_Importance", index=False)
    pred_df.to_excel(writer, sheet_name="Actual_vs_Predicted", index=False)
    if cv_df is not None:
        cv_df.to_excel(writer, sheet_name="CV_Results", index=False)

print(f"✅ Excel summary saved: {excel_path}")
