# -*- coding: utf-8 -*-
"""
PLSR (Partial Least Squares Regression) pipeline for Total Suspended Solids (TSS)
retrieval from hyperspectral reflectance data.

Author: Onur Karaca
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import (r2_score, mean_squared_error,
                             mean_absolute_error, mean_absolute_percentage_error)
from sklearn.model_selection import (train_test_split, KFold, cross_val_score)
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
MODEL_NAME = 'PLSR'

# =============================================================================
# CONFIGURATION
# =============================================================================
FILE_PATH        = "data/spectral_dataset.xlsx"   # in-situ spectra + TSS labels
OUTPUT_DIR       = "outputs"
METHOD_TAG       = "PLSR_v1"

PRISMA_TIF       = "data/prisma_scene.tif"        # optional: PRISMA GeoTIFF for RGB composite

WL_COL           = "Wavelength (nm)"
WL_RANGE         = (400, 900)
TEST_SIZE        = 0.20
RANDOM_STATE     = 42
CV_SPLITS        = 5
N_TOP_BANDS      = 10
N_RANDOM_SPLITS  = 100
TEMPORAL_CUTOFF  = 202503
MAX_COMPONENTS   = 20

S_TRAIN          = 30
S_TEST           = 60

PRISMA_RGB_BANDS = {'R': 28, 'G': 16, 'B': 9}
PRISMA_VNIR_WL   = np.linspace(400, 1010, 63)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def parse_tss(col_name):
    m = re.search(r'(?:-|_|\s)(\d+(?:\.\d+)?)\s*$', str(col_name))
    return float(m.group(1)) if m else None

def get_location(col):
    if 'MB' in col or 'Matagorda' in col.lower(): return 'Matagorda'
    if 'TB' in col or 'Trinity' in col.lower():   return 'Trinity'
    return 'Matagorda'

def get_date_order(col):
    months = {
        'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
        'July':7,'August':8,'September':9,'October':10,'November':11,'December':12,
        'Jan':1,'Feb':2,'Mar':3,'Apr':4,'Jun':6,'Jul':7,'Aug':8,
        'Sep':9,'Oct':10,'Nov':11,'Dec':12
    }
    year = 2024; month = 1
    for y in ['2024','2025']:
        if y in col: year = int(y)
    for k,v in months.items():
        if k in col: month = v; break
    return year * 100 + month

def compute_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    bias = float(np.mean(y_pred - y_true))
    r, _ = pearsonr(y_true, y_pred)
    return dict(R2=r2, RMSE=rmse, MAE=mae, MAPE=mape, Bias=bias, Pearson=r)

def compute_vip(model, X):
    t = model.x_scores_; w = model.x_weights_; q = model.y_loadings_
    p, h = w.shape; vip = np.zeros(p)
    s = np.diag(t.T @ t @ q.T @ q); total_s = np.sum(s)
    for i in range(p):
        weight = np.array([(w[i,j]/np.linalg.norm(w[:,j]))**2 for j in range(h)])
        vip[i] = np.sqrt(p * np.dot(weight, s) / total_s)
    return vip

def select_n_components_cv(X_tr, y_tr, max_comp, cv_splits, random_state):
    kf = KFold(cv_splits, shuffle=True, random_state=random_state)
    rmse_cv = []; max_comp = min(max_comp, X_tr.shape[0]-1, X_tr.shape[1])
    for n in range(1, max_comp+1):
        pls = PLSRegression(n_components=n, scale=True)
        scores = -cross_val_score(pls, X_tr, y_tr, cv=kf, scoring='neg_root_mean_squared_error')
        rmse_cv.append(scores.mean())
    best_n = int(np.argmin(rmse_cv)) + 1
    logging.info(f"n_components CV RMSE: {[round(v,2) for v in rmse_cv]}")
    logging.info(f"Optimal n_components = {best_n} (RMSE={rmse_cv[best_n-1]:.4f})")
    return best_n, rmse_cv

def save_ncomp_selection(rmse_cv, best_n, save_path):
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(range(1, len(rmse_cv)+1), rmse_cv, 'o-', color='#2E86C1', lw=1.8, ms=5, alpha=0.85)
    ax.axvline(best_n, color='#C0392B', lw=2, linestyle='--', label=f'Optimal n={best_n}')
    ax.set_xlabel('Number of Components', fontsize=11); ax.set_ylabel('CV RMSE [mg/L]', fontsize=11)
    ax.set_title('PLSR — n_components Selection (5-fold CV)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, framealpha=0.9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(save_path, dpi=1300, bbox_inches='tight'); plt.close()
    logging.info(f"n_components plot saved \u2192 {save_path}")

def save_scatter(y_tr, yp_tr, y_te, yp_te, m_tr, m_te,
                 cv_r2_mean, cv_r2_std, cv_rmse_mean, cv_rmse_std, save_path):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(y_tr, yp_tr, c='#5B9BD5', s=S_TRAIN, alpha=0.75, zorder=3, label=f'Train  (n={len(y_tr)})')
    ax.scatter(y_te, yp_te, c='#E8604C', s=S_TEST,  alpha=0.85, zorder=4, label=f'Test   (n={len(y_te)})')
    lo = min(float(y_tr.min()), float(y_te.min())) * 0.92
    hi = max(float(y_tr.max()), float(y_te.max())) * 1.05
    ax.plot([lo,hi],[lo,hi],'--',color='black',lw=1.2,zorder=1,label='1:1 line')
    slope,intercept = np.polyfit(y_te.values, yp_te, 1)
    xf = np.linspace(lo,hi,200)
    ax.plot(xf,slope*xf+intercept,'-',color='#E8604C',lw=1.5,zorder=2,label=f'Test fit  (slope={slope:.2f})')
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel('Observed TSS [mg/L]',fontsize=12); ax.set_ylabel('Predicted TSS [mg/L]',fontsize=12)
    ax.set_title('PLSR  --  Predicted vs. Observed TSS\n(Combined Dataset)',fontsize=13,fontweight='bold')
    ax.legend(loc='upper left',fontsize=9,framealpha=0.9)
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    stats=(f"Train Metrics:\n  $R^2$  = {m_tr['R2']:.4f}\n  RMSE = {m_tr['RMSE']:.2f} mg/L\n\n"
           f"Test Metrics:\n  $R^2$  = {m_te['R2']:.4f}\n  RMSE = {m_te['RMSE']:.2f} mg/L\n"
           f"  MAE  = {m_te['MAE']:.2f} mg/L\n  Bias  = {m_te['Bias']:.2f} mg/L\n\n"
           f"5-fold CV (train only):\n  $R^2$  = {cv_r2_mean:.4f} \u00b1 {cv_r2_std:.4f}\n"
           f"  RMSE = {cv_rmse_mean:.2f} \u00b1 {cv_rmse_std:.2f} mg/L")
    ax.text(0.97,0.03,stats,transform=ax.transAxes,ha='right',va='bottom',fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.5',fc='#FFFFF0',ec='black',alpha=0.9))
    plt.tight_layout(); plt.savefig(save_path,dpi=1300,bbox_inches='tight'); plt.close()
    logging.info(f"Scatter saved \u2192 {save_path}")

def save_vip_plot(vip_norm, wl_arr, top10_wl_set, save_path):
    colors=['#E8604C' if w in top10_wl_set else '#5B9BD5' for w in wl_arr]
    fig,ax=plt.subplots(figsize=(11,5))
    ax.bar(wl_arr,vip_norm,width=1.0,color=colors,alpha=0.85,linewidth=0)
    ax.axhline(1.0/vip_norm.max(),color='#7F8C8D',lw=1.0,linestyle='--',label='VIP = 1.0 threshold (normalized)')
    for wv in sorted(top10_wl_set):
        idx=np.where(wl_arr==wv)[0]
        if len(idx): ax.text(wv,vip_norm[idx[0]]+0.02,str(wv),ha='center',va='bottom',
                             fontsize=7.5,color='#C0392B',fontweight='bold',rotation=90)
    legend_el=[mpatches.Patch(facecolor='#E8604C',label='Top 10 Bands (VIP)'),
               mpatches.Patch(facecolor='#5B9BD5',label='Other Bands')]
    ax.legend(handles=legend_el,loc='upper right',fontsize=10,framealpha=0.9)
    ax.set_xlim(WL_RANGE[0]-5,WL_RANGE[1]+5); ax.set_ylim(0,1.15)
    ax.set_xlabel('Wavelength [nm]',fontsize=12); ax.set_ylabel('Normalized VIP Score',fontsize=12)
    ax.set_title('PLSR VIP Scores \u2014 All Bands\n(Red = Top 10)',fontsize=13,fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout(); plt.savefig(save_path,dpi=1300,bbox_inches='tight'); plt.close()
    logging.info(f"VIP plot saved \u2192 {save_path}")

def save_spectral_region_pie(top10_bands, vip_series, save_path):
    region_names=['Blue (400-499 nm)','Green (500-599 nm)','Red (600-699 nm)','NIR (700-900 nm)']
    region_colors=['#2E86C1','#27AE60','#C0392B','#8E44AD']
    region_imp=[0.0,0.0,0.0,0.0]; region_bands=[[],[],[],[]]
    for b in top10_bands:
        wl=int(b.split('_')[1]); imp=float(vip_series[b])
        if wl<500:   region_imp[0]+=imp; region_bands[0].append(str(wl))
        elif wl<600: region_imp[1]+=imp; region_bands[1].append(str(wl))
        elif wl<700: region_imp[2]+=imp; region_bands[2].append(str(wl))
        else:        region_imp[3]+=imp; region_bands[3].append(str(wl))
    total=sum(region_imp); labels,sizes,colors,explode=[],[],[],[]
    for i in range(4):
        if region_imp[i]>0:
            labels.append(region_names[i]+'  |  '+', '.join(region_bands[i])+' nm')
            sizes.append(region_imp[i]/total*100); colors.append(region_colors[i]); explode.append(0.04)
    fig,ax=plt.subplots(figsize=(7,6.5))
    wedges,texts,autotexts=ax.pie(sizes,colors=colors,explode=explode,autopct='%1.1f%%',
                                   startangle=140,pctdistance=0.72,wedgeprops=dict(linewidth=1.5,edgecolor='white'))
    for at in autotexts: at.set_fontsize(12); at.set_fontweight('bold'); at.set_color('white')
    ax.legend(wedges,labels,loc='lower center',bbox_to_anchor=(0.5,-0.22),fontsize=9,framealpha=0.9)
    ax.set_title(MODEL_NAME+' Top 10 Bands - Spectral Region Distribution (slice size = cumulative VIP score)',
                 fontsize=11,fontweight='bold',pad=14)
    plt.tight_layout(); plt.savefig(save_path,dpi=1300,bbox_inches='tight'); plt.close()

def save_repeated_splits_figure(r2_list, rmse_list, save_path):
    r2_arr=np.array(r2_list); rmse_arr=np.array(rmse_list)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    fig.suptitle(f'PLSR Uncertainty Analysis \u2014 {len(r2_list)} Repeated Random Splits',fontsize=13,fontweight='bold')
    ax=axes[0]
    ax.hist(r2_arr,bins=20,color='#5B9BD5',edgecolor='white',alpha=0.85,linewidth=0.5)
    ax.axvline(r2_arr.mean(),color='#C0392B',lw=2,linestyle='--')
    ax.axvline(r2_arr.mean()-r2_arr.std(),color='#E8604C',lw=1.2,linestyle=':')
    ax.axvline(r2_arr.mean()+r2_arr.std(),color='#E8604C',lw=1.2,linestyle=':')
    ax.set_xlabel('$R^2$',fontsize=12); ax.set_ylabel('Count',fontsize=12)
    ax.set_title('$R^2$ Distribution',fontsize=11,fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.text(0.97,0.97,f"Mean = {r2_arr.mean():.4f}\nStd   = {r2_arr.std():.4f}\n"
            f"Min   = {r2_arr.min():.4f}\nMax  = {r2_arr.max():.4f}\n\u2014 dashed = Mean\n\u00b7\u00b7\u00b7 dotted = \u00b11 SD",
            transform=ax.transAxes,ha='right',va='top',fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.4',fc='#FFFFF0',ec='black',alpha=0.9))
    ax=axes[1]
    ax.hist(rmse_arr,bins=20,color='#E8604C',edgecolor='white',alpha=0.85,linewidth=0.5)
    ax.axvline(rmse_arr.mean(),color='#C0392B',lw=2,linestyle='--')
    ax.axvline(rmse_arr.mean()-rmse_arr.std(),color='#5B9BD5',lw=1.2,linestyle=':')
    ax.axvline(rmse_arr.mean()+rmse_arr.std(),color='#5B9BD5',lw=1.2,linestyle=':')
    ax.set_xlabel('RMSE [mg/L]',fontsize=12); ax.set_ylabel('Count',fontsize=12)
    ax.set_title('RMSE Distribution',fontsize=11,fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    ax.text(0.97,0.97,f"Mean = {rmse_arr.mean():.2f} mg/L\nStd   = {rmse_arr.std():.2f} mg/L\n"
            f"Min   = {rmse_arr.min():.2f} mg/L\nMax  = {rmse_arr.max():.2f} mg/L\n\u2014 dashed = Mean\n\u00b7\u00b7\u00b7 dotted = \u00b11 SD",
            transform=ax.transAxes,ha='right',va='top',fontsize=8.5,
            bbox=dict(boxstyle='round,pad=0.4',fc='#FFFFF0',ec='black',alpha=0.9))
    plt.tight_layout(); plt.savefig(save_path,dpi=1300,bbox_inches='tight'); plt.close()
    logging.info(f"Repeated splits figure saved \u2192 {save_path}")

def save_validation_figure(spatial_results, temp_result, save_path):
    fig,axes=plt.subplots(1,3,figsize=(16,5.5))
    fig.suptitle('PLSR Validation Strategy Comparison\n(Spatial Blocking & Temporal Blocking)',
                 fontsize=13,fontweight='bold',y=1.01)
    spatial_configs=[
        ('Matagorda\u2192Trinity','Spatial: Train=Matagorda\nTest=Trinity','#C0392B','#FDEDEC'),
        ('Trinity\u2192Matagorda','Spatial: Train=Trinity\nTest=Matagorda','#1A5276','#D6EAF8'),
    ]
    for ax,(key,title,color,fc) in zip(axes[:2],spatial_configs):
        r=spatial_results[key]; y_obs=r['y_te']; y_pred=r['yp_te']
        lo=min(y_obs.min(),y_pred.min())*0.88; hi=max(y_obs.max(),y_pred.max())*1.08
        ax.scatter(y_obs,y_pred,c=color,s=55,alpha=0.80,zorder=3)
        ax.plot([lo,hi],[lo,hi],'--k',lw=1.2,zorder=1)
        slope,intercept=np.polyfit(y_obs.values,y_pred,1); xf=np.linspace(lo,hi,200)
        ax.plot(xf,slope*xf+intercept,'-',color=color,lw=1.5,zorder=2)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_xlabel('Observed TSS [mg/L]',fontsize=10); ax.set_ylabel('Predicted TSS [mg/L]',fontsize=10)
        ax.set_title(title,fontsize=10,fontweight='bold')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        stats=(f"n_train={r['n_train']} | n_test={r['n_test']}\n$R^2$ = {r['R2']:.4f}\n"
               f"RMSE = {r['RMSE']:.2f} mg/L\nMAE  = {r['MAE']:.2f} mg/L\nBias = {r['Bias']:.2f} mg/L")
        ax.text(0.97,0.03,stats,transform=ax.transAxes,ha='right',va='bottom',fontsize=8,
                bbox=dict(boxstyle='round,pad=0.4',fc=fc,ec=color,alpha=0.9))
    ax=axes[2]; color='#1E8449'; fc='#EAFAF1'
    y_obs=temp_result['y_te']; y_pred=temp_result['yp_te']
    lo=min(y_obs.min(),y_pred.min())*0.88; hi=max(y_obs.max(),y_pred.max())*1.08
    ax.scatter(y_obs,y_pred,c=color,s=55,alpha=0.80,zorder=3)
    ax.plot([lo,hi],[lo,hi],'--k',lw=1.2,zorder=1)
    slope,intercept=np.polyfit(y_obs.values,y_pred,1); xf=np.linspace(lo,hi,200)
    ax.plot(xf,slope*xf+intercept,'-',color=color,lw=1.5,zorder=2)
    ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
    ax.set_xlabel('Observed TSS [mg/L]',fontsize=10); ax.set_ylabel('Predicted TSS [mg/L]',fontsize=10)
    ax.set_title('Temporal: Train=Aug24\u2013Feb25\nTest=Mar25\u2013Jul25',fontsize=10,fontweight='bold')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    stats=(f"n_train={temp_result['n_train']} | n_test={temp_result['n_test']}\n"
           f"$R^2$ = {temp_result['R2']:.4f}\nRMSE = {temp_result['RMSE']:.2f} mg/L\n"
           f"MAE  = {temp_result['MAE']:.2f} mg/L\nBias = {temp_result['Bias']:.2f} mg/L")
    ax.text(0.97,0.03,stats,transform=ax.transAxes,ha='right',va='bottom',fontsize=8,
            bbox=dict(boxstyle='round,pad=0.4',fc=fc,ec=color,alpha=0.9))
    plt.tight_layout(); plt.savefig(save_path,dpi=1300,bbox_inches='tight'); plt.close()
    logging.info(f"Validation figure saved \u2192 {save_path}")

def save_false_color_rgb(prisma_tif, rgb_bands, sensor_wl, save_path):
    if prisma_tif is None or not os.path.exists(str(prisma_tif)):
        logging.warning("PRISMA TIF not found — RGB composite skipped."); return
    try:
        import rasterio
    except ImportError:
        logging.warning("rasterio not installed — RGB composite skipped."); return
    def pct_stretch(arr,lo=2,hi=98):
        valid=arr[arr>0]
        if len(valid)==0: return arr
        lo_v=float(np.percentile(valid,lo)); hi_v=float(np.percentile(valid,hi))
        out=np.clip((arr-lo_v)/(hi_v-lo_v+1e-9),0,1); out[arr<=0]=0; return out
    try:
        with rasterio.open(str(prisma_tif)) as src:
            r_arr=src.read(rgb_bands['R']).astype(np.float32)*0.0001
            g_arr=src.read(rgb_bands['G']).astype(np.float32)*0.0001
            b_arr=src.read(rgb_bands['B']).astype(np.float32)*0.0001
        r_wl=sensor_wl[rgb_bands['R']-1]; g_wl=sensor_wl[rgb_bands['G']-1]; b_wl=sensor_wl[rgb_bands['B']-1]
        rgb=np.dstack([pct_stretch(r_arr),pct_stretch(g_arr),pct_stretch(b_arr)])
        fig,ax=plt.subplots(figsize=(8,7)); ax.imshow(rgb,interpolation='bilinear')
        ax.set_title(f'PRISMA False Color Composite\nR={r_wl:.0f} nm  G={g_wl:.0f} nm  B={b_wl:.0f} nm',
                     fontsize=10,fontweight='bold')
        ax.set_xlabel('Pixel (East-West)',fontsize=9); ax.set_ylabel('Pixel (North-South)',fontsize=9)
        ax.tick_params(labelsize=8); plt.tight_layout()
        plt.savefig(save_path,dpi=1200,bbox_inches='tight'); plt.close()
        logging.info(f"False color RGB saved \u2192 {save_path}")
    except Exception as e:
        logging.warning(f"RGB composite failed: {e}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    tag = datetime.now().strftime('%Y%m%d')
    out = Path(OUTPUT_DIR) / METHOD_TAG
    out.mkdir(parents=True, exist_ok=True)

    # STEP 1 — Load & parse data
    logging.info("Loading dataset...")
    df_raw = pd.read_excel(FILE_PATH, sheet_name=0)
    tmp = df_raw.copy()
    tmp[WL_COL] = pd.to_numeric(tmp[WL_COL], errors='coerce')
    tmp = tmp.dropna(subset=[WL_COL])
    df_sp = tmp.groupby(WL_COL).mean(numeric_only=True).sort_index()
    band_names = [f'X_{int(wl)}' for wl in df_sp.index]
    sample_cols, tss_vals = [], []
    for col in df_raw.columns:
        if col == WL_COL: continue
        tss = parse_tss(col)
        if tss is not None and col in df_sp.columns:
            sample_cols.append(col); tss_vals.append(tss)
    data_rows = [[tss]+df_sp[c].tolist() for c,tss in zip(sample_cols,tss_vals)]
    df = (pd.DataFrame(data_rows, columns=['TSS']+band_names).dropna().reset_index(drop=True))
    df['location']   = [get_location(c)   for c in sample_cols]
    df['date_order'] = [get_date_order(c) for c in sample_cols]
    X_cols = [c for c in df.columns if c.startswith('X_') and WL_RANGE[0]<=int(c.split('_')[1])<=WL_RANGE[1]]
    X_full = df[X_cols].values.astype(float)
    y      = df['TSS'].astype(float).values
    logging.info(f"Dataset: {len(df)} samples | {len(X_cols)} bands")

    # STEP 2 — Train / Test split
    max_idx   = list(np.where(y == y.max())[0])
    other_idx = [i for i in range(len(y)) if i not in max_idx]
    X_other = X_full[other_idx]; y_other = y[other_idx]
    X_tr_o, X_te, y_tr_o, y_te = train_test_split(
        X_other, y_other, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    X_tr = np.vstack([X_tr_o, X_full[max_idx]])
    y_tr = np.concatenate([y_tr_o, y[max_idx]])
    y_tr_s = pd.Series(y_tr)
    y_te_s = pd.Series(y_te)
    logging.info(f"Train: {len(y_tr)} | Test: {len(y_te)} | Total: {len(y_tr)+len(y_te)}")

    # STEP 3 — n_components selection
    logging.info("Selecting optimal n_components via 5-fold CV...")
    best_n, rmse_cv = select_n_components_cv(X_tr, y_tr, MAX_COMPONENTS, CV_SPLITS, RANDOM_STATE)
    save_ncomp_selection(rmse_cv, best_n, out/f'ncomp_selection_PLSR_{tag}.png')

    # STEP 4 — Fit final PLSR model
    logging.info(f"Fitting PLSR with n_components={best_n}...")
    best_pls = PLSRegression(n_components=best_n, scale=True)
    best_pls.fit(X_tr, y_tr)
    joblib.dump(best_pls, out/f'plsr_model_{tag}.pkl')

    # STEP 5 — VIP scores
    logging.info("Computing VIP scores...")
    vip        = compute_vip(best_pls, X_tr)
    vip_series = pd.Series(vip, index=X_cols)
    vip_norm   = vip / vip.max()
    top10_bands = vip_series.nlargest(N_TOP_BANDS).index.tolist()
    top10_wl    = {int(b.split('_')[1]) for b in top10_bands}
    wl_arr      = np.array([int(c.split('_')[1]) for c in X_cols])
    logging.info(f"Top 10 VIP bands: {sorted(top10_wl)}")

    top10_path = out / f'top10_bands_PLSR_{tag}.txt'
    with open(top10_path, 'w') as f:
        f.write("Top 10 Selected Bands — PLSR (by VIP score)\n")
        f.write("=" * 46 + "\n")
        for i,b in enumerate(top10_bands,1):
            f.write(f"{i:2d}. {b}  ({b.split('_')[1]} nm)  |  VIP = {vip_series[b]:.6f}\n")
    logging.info(f"Top-10 txt saved \u2192 {top10_path}")
    save_vip_plot(vip_norm, wl_arr, top10_wl, out/f'vip_scores_PLSR_{tag}.png')

    # STEP 6 — 5-Fold CV on TRAIN only
    logging.info("5-fold CV on train data...")
    kf      = KFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    cv_r2   = cross_val_score(best_pls, X_tr, y_tr, cv=kf, scoring='r2')
    cv_rmse = -cross_val_score(best_pls, X_tr, y_tr, cv=kf, scoring='neg_root_mean_squared_error')
    cv_r2_mean,cv_r2_std     = cv_r2.mean(),cv_r2.std()
    cv_rmse_mean,cv_rmse_std = cv_rmse.mean(),cv_rmse.std()
    logging.info(f"CV R2   = {cv_r2_mean:.4f} +/- {cv_r2_std:.4f}")
    logging.info(f"CV RMSE = {cv_rmse_mean:.4f} +/- {cv_rmse_std:.4f}")

    # STEP 7 — Test set evaluation
    yp_tr = best_pls.predict(X_tr).flatten()
    yp_te = best_pls.predict(X_te).flatten()
    m_tr  = compute_metrics(y_tr, yp_tr)
    m_te  = compute_metrics(y_te, yp_te)
    logging.info(f"Train R2 = {m_tr['R2']:.4f} | RMSE = {m_tr['RMSE']:.4f}")
    logging.info(f"Test  R2 = {m_te['R2']:.4f} | RMSE = {m_te['RMSE']:.4f}")

    # ── TAYLOR DIAGRAM STATS ─────────────────────────────────────────────
    # Note: y_te and yp_te are numpy arrays (no .values needed)
    obs_std  = float(np.std(y_te))
    pred_std = float(np.std(yp_te))
    pearson  = float(m_te['Pearson'])
    logging.info(f"Taylor | Obs Std (REF_STD) = {obs_std:.4f} mg/L")
    logging.info(f"Taylor | Pred Std           = {pred_std:.4f} mg/L")
    logging.info(f"Taylor | Pearson (corr)     = {pearson:.4f}")

    # STEP 8 — Main scatter plot
    save_scatter(y_tr_s, yp_tr, y_te_s, yp_te, m_tr, m_te,
                 cv_r2_mean, cv_r2_std, cv_rmse_mean, cv_rmse_std,
                 out/f'scatter_PLSR_{tag}.png')

    # STEP 9 — Repeated random splits
    logging.info(f"Running {N_RANDOM_SPLITS} repeated random splits...")
    r2_list, rmse_list = [], []
    for seed in range(N_RANDOM_SPLITS):
        X_tr_s2,X_te_s2,y_tr_s2,y_te_s2 = train_test_split(
            X_other, y_other, test_size=TEST_SIZE, random_state=seed)
        X_tr_s2 = np.vstack([X_tr_s2, X_full[max_idx]])
        y_tr_s2 = np.concatenate([y_tr_s2, y[max_idx]])
        pls_s = PLSRegression(n_components=best_n, scale=True)
        pls_s.fit(X_tr_s2, y_tr_s2)
        yp_s = pls_s.predict(X_te_s2).flatten()
        r2_list.append(r2_score(y_te_s2, yp_s))
        rmse_list.append(float(np.sqrt(mean_squared_error(y_te_s2, yp_s))))
    r2_arr=np.array(r2_list); rmse_arr=np.array(rmse_list)
    logging.info(f"Repeated R2   = {r2_arr.mean():.4f} +/- {r2_arr.std():.4f}")
    logging.info(f"Repeated RMSE = {rmse_arr.mean():.4f} +/- {rmse_arr.std():.4f}")
    save_repeated_splits_figure(r2_list, rmse_list, out/f'uncertainty_repeated_splits_PLSR_{tag}.png')

    # STEP 10 — Spatial blocking
    logging.info("Running spatial blocking validation...")
    spatial_results = {}
    for train_loc,test_loc in [('Matagorda','Trinity'),('Trinity','Matagorda')]:
        tr_idx=df[df['location']==train_loc].index.tolist()
        te_idx=df[df['location']==test_loc].index.tolist()
        pls_b=PLSRegression(n_components=best_n, scale=True)
        pls_b.fit(X_full[tr_idx], df.loc[tr_idx,'TSS'].astype(float).values)
        yp_b=pls_b.predict(X_full[te_idx]).flatten()
        y_te_b=df.loc[te_idx,'TSS'].astype(float)
        m_b=compute_metrics(y_te_b.values, yp_b)
        key=f'{train_loc}\u2192{test_loc}'
        spatial_results[key]={'n_train':len(tr_idx),'n_test':len(te_idx),'y_te':y_te_b,'yp_te':yp_b,**m_b}
        logging.info(f"Spatial {key}: R2={m_b['R2']:.4f} | RMSE={m_b['RMSE']:.4f}")

    # STEP 11 — Temporal blocking
    logging.info("Running temporal blocking validation...")
    tr_idx_t=df[df['date_order']<TEMPORAL_CUTOFF].index.tolist()
    te_idx_t=df[df['date_order']>=TEMPORAL_CUTOFF].index.tolist()
    pls_t=PLSRegression(n_components=best_n, scale=True)
    pls_t.fit(X_full[tr_idx_t], df.loc[tr_idx_t,'TSS'].astype(float).values)
    yp_t=pls_t.predict(X_full[te_idx_t]).flatten()
    y_te_t=df.loc[te_idx_t,'TSS'].astype(float)
    m_t=compute_metrics(y_te_t.values, yp_t)
    temp_result={'n_train':len(tr_idx_t),'n_test':len(te_idx_t),'y_te':y_te_t,'yp_te':yp_t,**m_t}
    logging.info(f"Temporal: R2={m_t['R2']:.4f} | RMSE={m_t['RMSE']:.4f}")

    # STEP 12 — Validation figure
    save_validation_figure(spatial_results, temp_result, out/f'validation_spatial_temporal_PLSR_{tag}.png')

    # STEP 13 — Pie chart
    save_spectral_region_pie(top10_bands, vip_series, out/f'spectral_region_pie_PLSR_{tag}.png')

    # STEP 14 — PRISMA RGB
    logging.info("Generating PRISMA false color RGB composite...")
    save_false_color_rgb(PRISMA_TIF, PRISMA_RGB_BANDS, PRISMA_VNIR_WL, out/f'PRISMA_FalseColor_RGB_{tag}.png')

    # STEP 15 — Full report + Taylor Diagram Stats
    rpt_path = out / f'model_report_PLSR_{tag}.txt'
    with open(rpt_path, 'w', encoding='utf-8') as f:
        f.write(f"PLSR - Model Report\n{'='*42}\n")
        f.write(f"Date           : {tag}\n")
        f.write(f"Total          : {len(df)} samples\n")
        f.write(f"Train          : {len(y_tr)} samples\n")
        f.write(f"Test           : {len(y_te)} samples\n")
        f.write(f"n_components   : {best_n}\n")
        f.write(f"Spectral bands : {len(X_cols)} (all bands used)\n\n")
        f.write(f"Train Metrics\n{'-'*25}\n")
        for k,v in m_tr.items(): f.write(f"  {k:8s}: {v:.4f}\n")
        f.write(f"\nTest Metrics\n{'-'*25}\n")
        for k,v in m_te.items(): f.write(f"  {k:8s}: {v:.4f}\n")
        f.write(f"\n5-Fold CV - TRAIN only\n{'-'*25}\n")
        f.write(f"  R2   : {cv_r2_mean:.4f} +/- {cv_r2_std:.4f}\n")
        f.write(f"  RMSE : {cv_rmse_mean:.4f} +/- {cv_rmse_std:.4f}\n")
        f.write(f"\nRepeated Random Splits ({N_RANDOM_SPLITS}x)\n{'-'*25}\n")
        f.write(f"  R2   : {r2_arr.mean():.4f} +/- {r2_arr.std():.4f}\n")
        f.write(f"  RMSE : {rmse_arr.mean():.4f} +/- {rmse_arr.std():.4f}\n")
        f.write(f"  R2 min/max : {r2_arr.min():.4f} / {r2_arr.max():.4f}\n")
        f.write(f"\nSpatial Blocking Validation\n{'-'*25}\n")
        for key,r in spatial_results.items():
            f.write(f"  {key}: n_train={r['n_train']} | n_test={r['n_test']}\n")
            f.write(f"    R2={r['R2']:.4f} | RMSE={r['RMSE']:.4f} | Bias={r['Bias']:.4f}\n")
        f.write(f"\nTemporal Blocking (Aug24-Feb25 -> Mar25-Jul25)\n{'-'*25}\n")
        f.write(f"  n_train={temp_result['n_train']} | n_test={temp_result['n_test']}\n")
        f.write(f"  R2={temp_result['R2']:.4f} | RMSE={temp_result['RMSE']:.4f} | Bias={temp_result['Bias']:.4f}\n")
        f.write(f"\nTop-10 Bands by VIP Score\n{'-'*25}\n")
        for i,b in enumerate(top10_bands,1): f.write(f"  {i:2d}. {b}  (VIP={vip_series[b]:.6f})\n")
        f.write(f"\nHyperparameter\n{'-'*25}\n")
        f.write(f"  n_components: {best_n}\n")
        # ── Taylor Diagram Statistics ──────────────────────────────────
        f.write(f"\nTaylor Diagram Statistics (Test Set)\n{'-'*25}\n")
        f.write(f"  Obs  Std  (REF_STD) : {obs_std:.4f} mg/L\n")
        f.write(f"  Pred Std            : {pred_std:.4f} mg/L\n")
        f.write(f"  Pearson (corr)      : {pearson:.4f}\n")
        f.write(f"\n  --> Values for the Taylor diagram script:\n")
        f.write(f"      REF_STD = {obs_std:.3f}\n")
        f.write(f"      PLSR: std={pred_std:.3f}, corr={pearson:.4f}\n")
    logging.info(f"Report saved \u2192 {rpt_path}")
    logging.info("=" * 55)
    logging.info(f"Train        R2 = {m_tr['R2']:.4f}")
    logging.info(f"Test         R2 = {m_te['R2']:.4f}")
    logging.info(f"5-fold CV    R2 = {cv_r2_mean:.4f} +/- {cv_r2_std:.4f}")
    logging.info(f"Repeated     R2 = {r2_arr.mean():.4f} +/- {r2_arr.std():.4f}")
    logging.info(f"Spatial M->T R2 = {spatial_results['Matagorda\u2192Trinity']['R2']:.4f}")
    logging.info(f"Spatial T->M R2 = {spatial_results['Trinity\u2192Matagorda']['R2']:.4f}")
    logging.info(f"Temporal     R2 = {temp_result['R2']:.4f}")
    logging.info(f"n_components    = {best_n}")
    logging.info(f"Taylor REF_STD  = {obs_std:.4f} mg/L")
    logging.info(f"Taylor Pred Std = {pred_std:.4f} mg/L")
    logging.info(f"Taylor Pearson  = {pearson:.4f}")
    logging.info(f"Outputs      -> {out}")
    logging.info("=" * 55)


if __name__ == "__main__":
    main()