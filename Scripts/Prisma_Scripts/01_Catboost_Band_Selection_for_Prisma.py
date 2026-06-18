# -*- coding: utf-8 -*-
"""
CatBoost Stable Pipeline - Final Visuals & Model Export
- Added: .pkl model saving (joblib).
- Added: Feature Importance Top-10 list, plot, and TXT export.
- Visuals: Bottom-right label with BOTH Train & Test R2/RMSE.
"""

import os, re, warnings
from datetime import datetime
from pathlib import Path

# Thread yönetimi
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import shap
import seaborn as sns

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from scipy.stats import randint
from catboost import CatBoostRegressor

# ==============================
# 0) SETTINGS
# ==============================
file_path = r"...................\Hyperspectral_Water_Reflectance_Spectra_and_TSS_in_Matagorda_and_Trinity_Bays.xlsx"
output_root = r".................\Prisma_Result\Catboost_Output"
METHOD_TAG = "CatBoost_Model"
output_dir = os.path.join(output_root, METHOD_TAG)
os.makedirs(output_dir, exist_ok=True)

tag = datetime.now().strftime("%Y%m%d")
WL_COL, WL_MIN, WL_MAX = "Wavelength (nm)", 400, 1000
TEST_SIZE, RANDOM_STATE = 0.20, 42
CV_SPLITS, N_TOP_BANDS = 3, 10
S_TEST, S_TRAIN = 28, 30

# ==============================
# 1) DATA PREPARATION
# ==============================
raw_df = pd.read_excel(file_path)
tmp = raw_df.copy()
tmp[WL_COL] = pd.to_numeric(tmp[WL_COL], errors="coerce")
tmp = tmp.dropna(subset=[WL_COL])
df_idx = tmp.groupby(WL_COL).mean(numeric_only=True).sort_index()

band_names = [f"X_{int(wl)}" for wl in df_idx.index]
def parse_tss(col):
    m = re.search(r"(?:-|_|\s)(\d+(?:\.\d+)?)\s*$", str(col))
    return float(m.group(1)) if m else None

sample_cols, tss_vals = [], []
for col in raw_df.columns:
    if col == WL_COL: continue
    tss = parse_tss(col)
    if tss is not None and col in df_idx.columns:
        sample_cols.append(col); tss_vals.append(tss)

df = pd.DataFrame([[tss] + df_idx[c].tolist() for c, tss in zip(sample_cols, tss_vals)], 
                  columns=["TSS"] + band_names).dropna().reset_index(drop=True)

X_cols = [c for c in df.columns if c.startswith("X_") and WL_MIN <= int(c.split("_")[1]) <= WL_MAX]
X_full, y = df[X_cols], df["TSS"].astype(float)

# ==============================
# 2) FEATURE SELECTION & TXT EXPORT
# ==============================
cat_fs = CatBoostRegressor(iterations=200, random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False)
cat_fs.fit(X_full, y)
feat_imp = pd.Series(cat_fs.get_feature_importance(), index=X_full.columns).sort_values(ascending=False)

selected_bands = list(feat_imp.index[:N_TOP_BANDS])

# ✅ YENİ: Seçilen band isimlerini .txt dosyasına kaydet
txt_path = os.path.join(output_dir, f"selected_bands_{tag}.txt")
with open(txt_path, "w") as f:
    f.write(f"Top {N_TOP_BANDS} Selected Bands for CatBoost\n")
    f.write("="*30 + "\n")
    for i, band in enumerate(selected_bands, 1):
        score = feat_imp[band]
        f.write(f"{i}. {band} (Score: {score:.4f})\n")

print(f"✅ Band isimleri kaydedildi: {txt_path}")

# Feature Importance Grafiği
plt.figure(figsize=(8, 5))
sns.barplot(x=feat_imp.values[:N_TOP_BANDS], y=feat_imp.index[:N_TOP_BANDS], palette="viridis")
plt.title(f"Top {N_TOP_BANDS} Feature Importance")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig(Path(output_dir, f"feature_importance_top10_{tag}.png"), dpi=300)
plt.close()

X_top = df[selected_bands]

# ==============================
# 3) SPLIT
# ==============================
max_idx = y[y == y.max()].index.tolist()
other_idx = y.index.difference(max_idx)
X_train, X_test, y_train, y_test = train_test_split(X_top.loc[other_idx], y.loc[other_idx], test_size=TEST_SIZE, random_state=RANDOM_STATE)
if len(max_idx) > 0:
    X_train = pd.concat([X_train, X_top.loc[max_idx]])
    y_train = pd.concat([y_train, y.loc[max_idx]])
    X_test = pd.concat([X_test, X_top.loc[max_idx]])
    y_test = pd.concat([y_test, y.loc[max_idx]])

# ==============================
# 4) STABLE PARAMETER TUNING
# ==============================
dist = {
    "iterations": randint(800, 1400),
    "depth": [2, 3],                         
    "learning_rate": [0.01, 0.015, 0.02, 0.03], 
    "l2_leaf_reg": [30, 50, 70, 90],         
    "subsample": [0.6, 0.7, 0.8],
    "random_strength": [1, 2, 5, 10]
}

search = RandomizedSearchCV(
    CatBoostRegressor(loss_function='RMSE', random_seed=RANDOM_STATE, bootstrap_type='Bernoulli', 
                      verbose=False, allow_writing_files=False),
    param_distributions=dist, n_iter=35, cv=KFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE),
    scoring="neg_root_mean_squared_error", random_state=RANDOM_STATE, refit=True
)
search.fit(X_train, y_train)
best_cat = search.best_estimator_

# ✅ Model Kaydetme (.pkl)
model_path = Path(output_dir, f"catboost_model_{tag}.pkl")
joblib.dump(best_cat, model_path)
print(f"✅ Model kaydedildi: {model_path}")

# ==============================
# 5) SCATTER PLOT & METRICS
# ==============================
y_pred_test, y_pred_train = best_cat.predict(X_test), best_cat.predict(X_train)
r2_t, rmse_t = r2_score(y_test, y_pred_test), np.sqrt(mean_squared_error(y_test, y_pred_test))
r2_tr, rmse_tr = r2_score(y_train, y_pred_train), np.sqrt(mean_squared_error(y_train, y_pred_train))

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(y_test, y_pred_test, c="red", s=S_TEST, label=f"Test (n={len(y_test)})", zorder=2, alpha=0.8)
ax.scatter(y_train, y_pred_train, c="black", s=S_TRAIN, label=f"Train (n={len(y_train)})", zorder=3, alpha=0.9)

lims = [min(y.min(), y_pred_test.min()), max(y.max(), y_pred_test.max())]
ax.plot(lims, lims, "--", color="gray", zorder=1)

ax.set_xlabel("Actual TSS")
ax.set_ylabel("Predicted TSS")
ax.set_title(f"CatBoost Model: Train vs Test Balance")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.2)

dual_label = (
    f"Train Metrics:\n"
    f"R² = {r2_tr:.3f}\n"
    f"RMSE = {rmse_tr:.3f}\n\n"
    f"Test Metrics:\n"
    f"R² = {r2_t:.3f}\n"
    f"RMSE = {rmse_t:.3f}"
)

ax.text(0.98, 0.02, dual_label, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, fontweight='medium',
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))

plt.tight_layout()
plt.savefig(Path(output_dir, f"stable_scatter_labeled_{tag}.png"), dpi=1200)

# ==============================
# 6) SHAP & PIE
# ==============================
explainer = shap.TreeExplainer(best_cat)
shap_values = explainer.shap_values(X_train)
blue, green, red = (400, 500), (501, 600), (601, 800)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
groups = {"Blue": 0.0, "Green": 0.0, "Red": 0.0}
for bname, val in zip(X_train.columns, mean_abs_shap):
    w = int(bname.split("_")[1])
    if blue[0] <= w <= blue[1]: groups["Blue"] += val
    elif green[0] <= w <= green[1]: groups["Green"] += val
    elif red[0] <= w <= red[1]: groups["Red"] += val

plt.figure(figsize=(6,6))
plt.pie(groups.values(), labels=[f"{k} ({v/sum(groups.values())*100:.1f}%)" for k,v in groups.items()], autopct='%1.1f%%', startangle=140)
plt.title("Importance by Color Group")
plt.savefig(Path(output_dir, f"color_pie_{tag}.png"), dpi=300)

print(f"✅ Başarılı! \nTrain R2: {r2_tr:.3f} | Test R2: {r2_t:.3f}")
