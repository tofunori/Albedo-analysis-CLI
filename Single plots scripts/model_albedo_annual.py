#!/usr/bin/env python3
"""
Standalone Multiple Regression for Haig Glacier with Rainfall integration only
- Baseline model: OLS Albedo ~ Temperature + BC_AOD
- Extended model: OLS Albedo ~ Temperature + BC_AOD + Rainfall (if available)
- Standardized coefficients (z-scored predictors) for interpretability
- Centered interaction model (secondary): Albedo ~ T_c + B_c + T_c×B_c (explicitly without precip)
- Residual diagnostics (residuals vs fitted, Q–Q)
- Influence check: Cook's distance (report high-D points, optional refit)
- Robustness: trimmed windows and median aggregation (baseline predictors)
- Plots: pairs (base), pairs with Rainfall, partial regressions (including Rainfall)
- Model comparison: Extended vs Baseline (ΔadjR², AIC/BIC, nested F-test)
- LMG relative importance analysis (average ΔR² over all entry orders)
- Summary table: standardized coef, p, partial R², LMG share (%)
- Stability add-on: annotate high Cook’s D with years; replicate LMG/partial R² and summary table after excluding high-D points
- Outputs appended to the current run folder's summary.txt and plots saved there
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from output_manager import OutputManager  # provided by your main project

# =============================================================================
# CONFIG
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("haig_multiple_regression")

CONFIG = {
    'data_paths': {
        'haig': {
            'modis': "D:/Downloads/MODIS_Terra_Aqua_MultiProduct_2002-01-01_to_2025-01-01.csv",
            'aws': "D:/Documents/Projects/Haig_analysis/data/csv/HaigAWS_daily_2002_2015_gapfilled.csv",
            'temperature': "D:/Downloads/Haig_Glacier_Climate_JuneSept_Daily_MERRA2_Speciated_AOD - Haig_Glacier_Climate_JuneSept_Daily_MERRA2_Speciated_AOD.csv"
        }
    },
    'aws_stations': {'haig': {'lat': 50.7186, 'lon': -115.3433, 'name': 'Haig AWS'}},
    'methods': ['MOD09GA'],
    'method_mapping': {
        'mcd43a3': 'MCD43A3', 'MCD43A3': 'MCD43A3',
        'mod09ga': 'MOD09GA', 'MOD09GA': 'MOD09GA',
        'myd09ga': 'MYD09GA', 'MYD09GA': 'MYD09GA',
        'mod10a1': 'MOD10A1', 'MOD10A1': 'MOD10A1',
        'myd10a1': 'MYD10A1', 'MYD10A1': 'MYD10A1'
    },
    'trend_analysis': {
        'alpha': 0.05,
        'seasonal_analysis': True,
        'prewhitening': True,
        'min_years': 5,
        'trend_period': 'annual',
        'season_months': [6, 7, 8, 9],
        'annual_agg': 'mean',   # 'mean' or 'median'
        'weighted_annual': False
    },
    'quality_filters': {'min_glacier_fraction': 0.1, 'min_observations': 10},
    'output': {
        'analysis_name': 'haig_trend_analysis',
        'base_dir': 'outputs',
        'summary_template': {}
    },
}

SUMMARY_TXT_NAME = "summary.txt"
PAIRS_PLOT = "regression_pairs.png"
PAIRS_PLOT_WITH_PRECIP = "regression_pairs_with_rain.png"
PARTIAL_TEMP_PLOT = "partial_regression_temperature.png"
PARTIAL_BC_PLOT = "partial_regression_bc_aod.png"
PARTIAL_PRECIP_PLOT = "partial_regression_rainfall.png"
RESIDUALS_PLOT = "residuals_vs_fitted.png"
QQ_PLOT = "residuals_qq_plot.png"
INTERACTION_PLOT = "interaction_simple_slopes.png"

# =============================================================================
# HELPERS
# =============================================================================

def enforce_albedo_bounds(df: pd.DataFrame, col: str = 'Albedo') -> pd.DataFrame:
    if col not in df.columns:
        return df
    return df[(df[col] >= 0) & (df[col] <= 1)].copy()

def prep_ts(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df[['date', value_col]].dropna().sort_values('date').copy()
    out['year'] = out['date'].dt.year
    out['month'] = out['date'].dt.month
    out['day_of_year'] = out['date'].dt.dayofyear
    return out

def seasonal_annual_agg(df: pd.DataFrame, date_col: str, value_col: str,
                        months: Optional[List[int]], agg: str,
                        return_counts: bool = True) -> pd.DataFrame:
    d = df.copy()
    if 'month' not in d.columns:
        d['month'] = d[date_col].dt.month
    if months:
        d = d[d['month'].isin(months)]
    if d.empty:
        cols = ['date', value_col] + (['n_obs'] if return_counts else [])
        return pd.DataFrame(columns=cols)
    grp = d.groupby(d[date_col].dt.year)[value_col]
    out = grp.agg(agg).to_frame(value_col)
    if return_counts:
        out['n_obs'] = grp.size()
    out = out.reset_index().rename(columns={date_col: 'year'})
    out['date'] = pd.to_datetime(out['year'], format='%Y')
    cols = ['date', value_col] + (['n_obs'] if return_counts else [])
    return out[cols]

def get_value_col(df: pd.DataFrame) -> Optional[str]:
    for c in ('Albedo', 'Temperature', 'BC_AOD', 'Rainfall'):
        if c in df.columns:
            return c
    return None

def find_latest_run_dir(base_dir: Path, analysis_name: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(analysis_name)]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else base

# =============================================================================
# DATA LOADING
# =============================================================================

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def load_haig_data_complete(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        paths = self.config['data_paths']['haig']
        modis = self._load_modis_data(paths['modis'])
        aws = self._load_aws_data(paths['aws'])
        temp = self._load_temperature_data(paths['temperature'])
        precip = self._load_precip_data(paths['temperature'])
        return modis, aws, temp, precip

    def _load_modis_data(self, file_path: str) -> pd.DataFrame:
        data = pd.read_csv(file_path)
        data.columns = [c.strip() for c in data.columns]
        if 'date' not in data.columns:
            raise ValueError("Expected 'date' column in MODIS data")
        data['date'] = pd.to_datetime(data['date'])
        if 'method' in data.columns:
            data['method'] = data['method'].map(self.config['method_mapping']).fillna(data['method'])
            long_df = data.rename(columns={'albedo': 'Albedo'}) if 'albedo' in data.columns else data
        else:
            long_df = self._convert_to_long_format(data)
        if 'Albedo' not in long_df.columns and 'albedo' in long_df.columns:
            long_df.rename(columns={'albedo': 'Albedo'}, inplace=True)
        return enforce_albedo_bounds(long_df, 'Albedo')

    def _convert_to_long_format(self, data: pd.DataFrame) -> pd.DataFrame:
        method_cols: Dict[str, List[str]] = {'MOD09GA': [], 'MYD09GA': [], 'MOD10A1': [], 'MYD10A1': [], 'MCD43A3': []}
        for col in data.columns:
            cl = col.lower()
            if 'albedo' in cl:
                if 'mod09ga' in cl: method_cols['MOD09GA'].append(col)
                elif 'myd09ga' in cl: method_cols['MYD09GA'].append(col)
                elif 'mod10a1' in cl: method_cols['MOD10A1'].append(col)
                elif 'myd10a1' in cl: method_cols['MYD10A1'].append(col)
                elif 'mcd43a3' in cl: method_cols['MCD43A3'].append(col)
        common = [c for c in ['pixel_id', 'date', 'longitude', 'latitude'] if c in data.columns]
        gf_cols = [c for c in data.columns if 'glacier_fraction' in c.lower()]
        if gf_cols: common.append(gf_cols[0])
        rows = []
        for method, cols in method_cols.items():
            for col_name in cols:
                sub = data[common + [col_name]].copy()
                sub = sub[sub[col_name].notna()]
                if sub.empty: continue
                sub['method'] = method
                sub['Albedo'] = pd.to_numeric(sub[col_name], errors='coerce')
                sub.drop(columns=[col_name], inplace=True)
                rows.append(sub)
        if not rows:
            return pd.DataFrame(columns=['pixel_id', 'date', 'longitude', 'latitude', 'glacier_fraction', 'method', 'Albedo'])
        long_df = pd.concat(rows, ignore_index=True)
        gf_cols2 = [c for c in long_df.columns if 'glacier_fraction' in c.lower()]
        if gf_cols2 and 'glacier_fraction' not in long_df.columns:
            long_df.rename(columns={gf_cols2[0]: 'glacier_fraction'}, inplace=True)
        return long_df.dropna(subset=['date', 'Albedo']).sort_values(['pixel_id', 'date']).reset_index(drop=True)

    def _load_aws_data(self, file_path: str) -> pd.DataFrame:
        aws = pd.read_csv(file_path, sep=';', skiprows=6, decimal=',')
        aws.columns = aws.columns.str.strip()
        aws = aws.dropna(subset=['Year', 'Day'])
        aws['Year'] = aws['Year'].astype(int)
        aws['Day'] = aws['Day'].astype(int)
        aws['date'] = pd.to_datetime(aws['Year'].astype(str) + '-01-01') + pd.to_timedelta(aws['Day'] - 1, unit='D')
        albedo_cols = [c for c in aws.columns if 'albedo' in c.lower()]
        if not albedo_cols:
            raise ValueError("No albedo column found in Haig AWS data")
        aws['Albedo'] = pd.to_numeric(aws[albedo_cols[0]], errors='coerce')
        aws = aws[['date', 'Albedo']].dropna()
        aws = enforce_albedo_bounds(aws, 'Albedo')
        aws = aws[aws['Albedo'] > 0].drop_duplicates().sort_values('date').reset_index(drop=True)
        return aws

    def _load_temperature_data(self, file_path: str) -> pd.DataFrame:
        temp = pd.read_csv(file_path)
        temp.columns = temp.columns.str.strip()
        if 'date' in temp.columns:
            temp['date'] = pd.to_datetime(temp['date'])
        else:
            date_cols = [c for c in temp.columns if 'date' in c.lower() or 'time' in c.lower()]
            if not date_cols:
                raise ValueError("No date column found in temperature data")
            temp['date'] = pd.to_datetime(temp[date_cols[0]])
        if 'temperature_c' in temp.columns:
            temp['Temperature'] = pd.to_numeric(temp['temperature_c'], errors='coerce')
        else:
            temp_cols = [c for c in temp.columns if 'temp' in c.lower()]
            if not temp_cols:
                raise ValueError("No temperature column found in temperature data")
            temp['Temperature'] = pd.to_numeric(temp[temp_cols[0]], errors='coerce')
        if 'bc_aod_regional' in temp.columns:
            temp['BC_AOD'] = pd.to_numeric(temp['bc_aod_regional'], errors='coerce')
            temp = temp[['date', 'Temperature', 'BC_AOD']].copy()
        else:
            temp = temp[['date', 'Temperature']].copy()
        return temp.dropna(subset=['Temperature']).drop_duplicates().sort_values('date').reset_index(drop=True)

    def _load_precip_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if not date_cols:
                raise ValueError("No date column found in precip data")
            df['date'] = pd.to_datetime(df[date_cols[0]])

        col_map = {c.lower(): c for c in df.columns}
        rain_col = col_map.get('rainfall_mm') or col_map.get('rain_mm') or None

        out = pd.DataFrame({'date': df['date']})
        if rain_col:
            out['Rainfall_mm'] = pd.to_numeric(df[rain_col], errors='coerce')

        return out.dropna(subset=['date']).sort_values('date').reset_index(drop=True)

# =============================================================================
# PIXEL SELECTION
# =============================================================================

class PixelSelector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1 = np.radians(lat1); lon1 = np.radians(lon1)
        lat2 = np.radians(lat2); lon2 = np.radians(lon2)
        dlat = lat2 - lat1; dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2*R*np.arcsin(np.sqrt(a))

    def select_best_pixels(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        if modis_data.empty:
            return modis_data
        station = self.config['aws_stations'][glacier_id]
        if not {'latitude', 'longitude'}.issubset(modis_data.columns):
            return modis_data
        agg = {'Albedo': 'count', 'latitude': 'first', 'longitude': 'first'}
        if 'glacier_fraction' in modis_data.columns:
            agg['glacier_fraction'] = 'mean'
        summary = modis_data.groupby('pixel_id').agg(agg).reset_index()
        summary.rename(columns={'Albedo': 'n_observations', 'glacier_fraction': 'avg_glacier_fraction'}, inplace=True)
        q = self.config['quality_filters']
        quality = summary[
            (summary.get('avg_glacier_fraction', 1) > q['min_glacier_fraction']) &
            (summary['n_observations'] > q['min_observations'])
        ].copy()
        if quality.empty:
            return modis_data
        quality['distance_to_aws'] = self._haversine_distance(
            quality['latitude'].values, quality['longitude'].values, station['lat'], station['lon']
        )
        best = quality.sort_values(['avg_glacier_fraction', 'distance_to_aws'], ascending=[False, True]).head(1)
        pix_ids = best['pixel_id'].tolist()
        return modis_data[modis_data['pixel_id'].isin(pix_ids)].copy()

# =============================================================================
# DATA PROCESSOR
# =============================================================================

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def prepare_time_series_data(self, modis: pd.DataFrame, aws: pd.DataFrame, temp: pd.DataFrame, glacier_id: str,
                                 precip: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        out: Dict[str, pd.DataFrame] = {}
        if not aws.empty:
            out['aws_albedo'] = prep_ts(aws, 'Albedo')
        if not temp.empty:
            if 'Temperature' in temp.columns:
                out['temperature'] = prep_ts(temp, 'Temperature')
            if 'BC_AOD' in temp.columns:
                out['bc_aod'] = prep_ts(temp, 'BC_AOD')
        if precip is not None and not precip.empty:
            if 'Rainfall_mm' in precip.columns:
                out['rainfall'] = prep_ts(precip.rename(columns={'Rainfall_mm': 'Rainfall'}), 'Rainfall')
        if modis is not None and not modis.empty:
            for method in [m for m in modis['method'].unique() if m in self.config['methods']]:
                df = modis[modis['method'] == method][['date', 'Albedo']].dropna().sort_values('date')
                df = prep_ts(df, 'Albedo')
                out[f'modis_{method.lower()}'] = df
        return out

    def create_annual_series(self, ts_data: Dict[str, pd.DataFrame], months: Optional[List[int]], agg: str) -> Dict[str, pd.DataFrame]:
        annual: Dict[str, pd.DataFrame] = {}
        for name, df in ts_data.items():
            if df.empty or 'date' not in df.columns:
                continue
            val = get_value_col(df)
            if not val:
                continue
            out = seasonal_annual_agg(df, 'date', val, months, agg, return_counts=True)
            if out.empty:
                continue
            annual[f"{name}_annual"] = out
        return annual

# =============================================================================
# REGRESSION + DIAGNOSTICS
# =============================================================================

def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    vifs = []
    if X.shape[1] == 1:
        return pd.DataFrame([{'variable': X.columns[0], 'VIF': 1.0}])
    for col in X.columns:
        y = X[col].values
        X_others = X.drop(columns=[col]).values
        X_others_const = np.column_stack([np.ones(len(X_others)), X_others])
        beta, _, _, _, _, _, _, _ = ols_with_stats(y, X_others_const)
        y_hat = X_others_const @ beta
        sst = np.sum((y - y.mean())**2)
        sse = np.sum((y - y_hat)**2)
        r2 = 1 - sse / sst if sst > 0 else 0.0
        vif = np.inf if (1 - r2) <= 0 else 1.0 / (1.0 - r2)
        vifs.append((col, vif))
    return pd.DataFrame(vifs, columns=['variable', 'VIF'])

def safe_inv(mat: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(mat)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(mat)

def ols_with_stats(y: np.ndarray, X: np.ndarray):
    n, k = X.shape
    XtX = X.T @ X
    Xty = X.T @ y
    beta = np.linalg.solve(XtX, Xty) if np.linalg.matrix_rank(XtX) == XtX.shape[0] else (safe_inv(XtX) @ Xty)
    y_hat = X @ beta
    resid = y - y_hat

    sse = float(resid.T @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - sse / sst if sst > 0 else np.nan
    dof = n - k
    sigma2 = sse / dof if dof > 0 else np.nan
    cov_beta = sigma2 * safe_inv(XtX)
    se = np.sqrt(np.diag(cov_beta))
    tvals = beta / se
    pvals = 2 * (1 - stats.t.cdf(np.abs(tvals), df=max(dof, 1)))
    p = k - 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if (n - p - 1) > 0 else np.nan
    return beta, se, tvals, pvals, r2, adj_r2, resid, y_hat

def aic_bic(y: np.ndarray, y_hat: np.ndarray, k: int) -> Tuple[float, float]:
    n = len(y)
    resid = y - y_hat
    sse = float(resid.T @ resid)
    sigma2 = sse / n if n > 0 else np.nan
    ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1) if sigma2 > 0 else np.nan
    aic = 2 * k - 2 * ll if ll == ll else np.nan
    bic = np.log(n) * k - 2 * ll if ll == ll else np.nan
    return aic, bic

def nested_f_test(sse_restricted: float, sse_full: float, df_restricted: int, df_full: int, p_full: int, p_restricted: int) -> Tuple[float, float]:
    num_df = p_full - p_restricted
    den_df = df_full
    if num_df <= 0 or den_df <= 0 or sse_full <= 0:
        return np.nan, np.nan
    F = ((sse_restricted - sse_full) / num_df) / (sse_full / den_df)
    p = 1 - stats.f.cdf(F, num_df, den_df)
    return F, p

def standardize(series: pd.Series) -> pd.Series:
    sd = series.std(ddof=1)
    return (series - series.mean()) / sd if sd > 0 else series * 0.0

def cooks_distance(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    n, k = X.shape
    _, _, _, _, _, _, resid, _ = ols_with_stats(y, X)
    XtX_inv = safe_inv(X.T @ X)
    H_diag = np.einsum('ij,jk,ik->i', X, XtX_inv, X)
    mse = np.sum(resid**2) / (n - k)
    cook = (resid**2 / (k * mse)) * (H_diag / (1 - H_diag)**2)
    return cook

# ---------------- LMG relative importance ----------------

def _r2_from_design(y: np.ndarray, X: np.ndarray) -> float:
    _, _, _, _, r2, _, _, _ = ols_with_stats(y, X)
    return 0.0 if not np.isfinite(r2) else float(r2)

def _design_from_df(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    M = df[cols].values.astype(float)
    return np.column_stack([np.ones(len(df)), M])

def lmg_importance(y: np.ndarray, df_pred: pd.DataFrame, predictors: List[str]) -> Dict[str, Tuple[float, float]]:
    """
    Returns {var: (absolute_R2_contribution, relative_share)} for each predictor.
    absolute_R2_contribution = average (over all permutations) of the R² increase when the variable enters.
    relative_share = contribution / sum(contributions).
    """
    p = len(predictors)
    if p == 0:
        return {}
    contrib = {v: 0.0 for v in predictors}
    perms = list(itertools.permutations(predictors, p))
    for order in perms:
        used = []
        r2_prev = 0.0
        for v in order:
            used.append(v)
            X_used = _design_from_df(df_pred, used)
            r2_curr = _r2_from_design(y, X_used)
            delta = max(r2_curr - r2_prev, 0.0)
            contrib[v] += delta
            r2_prev = r2_curr
    nperm = len(perms)
    for v in contrib:
        contrib[v] /= nperm
    total = sum(contrib.values())
    out = {v: (contrib[v], (contrib[v] / total if total > 0 else np.nan)) for v in predictors}
    return out

def partial_r2_for_var(y: np.ndarray, X_full: np.ndarray, cols: List[str], var: str, df_pred: pd.DataFrame) -> float:
    cols_wo = [c for c in cols if c != var]
    X_res = _design_from_df(df_pred, cols_wo)
    _, _, _, _, r2_full, _, _, _ = ols_with_stats(y, X_full)
    _, _, _, _, r2_res,  _, _, _ = ols_with_stats(y, X_res)
    return max((0.0 if not np.isfinite(r2_full) else r2_full) - (0.0 if not np.isfinite(r2_res) else r2_res), 0.0)

# =============================================================================
# REPORTING AND PLOTS
# =============================================================================

def append_report(run_dir: Path,
                  title: str,
                  names: List[str],
                  beta: np.ndarray, se: np.ndarray, tvals: np.ndarray, pvals: np.ndarray,
                  r2: float, adj_r2: float, n: int,
                  vif_df: Optional[pd.DataFrame] = None,
                  extra_lines: Optional[List[str]] = None):
    lines = []
    lines.append(f"\n{title}")
    lines.append("-" * max(50, len(title)))
    lines.append(f"n = {n}, R^2 = {r2:.3f}, adj. R^2 = {adj_r2:.3f}")
    if names and beta.size:
        lines.append("\nCoefficients:")
        lines.append("  term            coef         SE           t           p")
        for name, b, s, t, p in zip(names, beta, se, tvals, pvals):
            lines.append(f"  {name:<12} {b:>+12.6f} {s:>12.6f} {t:>12.3f} {p:>12.3g}")
    if vif_df is not None and len(vif_df) > 0:
        lines.append("\nVariance Inflation Factors (VIF):")
        for _, row in vif_df.iterrows():
            v = row['VIF']
            v_str = f"{v:.3f}" if np.isfinite(v) else "inf"
            lines.append(f"  {row['variable']:<12} VIF = {v_str}")
    if extra_lines:
        lines.append("")
        lines.extend(extra_lines)
    lines.append("")
    summary_path = run_dir / SUMMARY_TXT_NAME
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logger.info(f"Appended report to {summary_path}")

def plot_pairs_and_save(df: pd.DataFrame, run_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=180)
    fig.suptitle("Pairwise relationships (annual)", fontsize=14, fontweight='bold', y=0.98)

    pairs = [
        ('Temperature', 'Albedo', 'Temperature (°C)', 'Albedo (unitless)'),
        ('BC_AOD', 'Albedo', 'BC AOD', 'Albedo (unitless)'),
        ('Temperature', 'BC_AOD', 'Temperature (°C)', 'BC AOD'),
    ]
    colors = ['#3498DB', '#E74C3C', '#27AE60']

    for ax, (xcol, ycol, xlabel, ylabel), color in zip(axes, pairs, colors):
        if xcol not in df.columns or ycol not in df.columns:
            ax.axis('off')
            continue
        x = df[xcol].values
        y = df[ycol].values
        ax.scatter(x, y, color=color, s=50, alpha=0.75)
        if len(x) > 1 and np.ptp(x) > 0:
            slope, intercept, r, p, _ = stats.linregress(x, y)
            xx = np.linspace(x.min(), x.max(), 100)
            yy = slope * xx + intercept
            ax.plot(xx, yy, color='#2C3E50', lw=2, label=f'fit: R²={r**2:.3f}, p={p:.3f}')
            ax.legend(loc='best')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if 'Albedo' in ycol:
            ax.set_ylim(0, 1)

    out = run_dir / PAIRS_PLOT
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved pairwise plot: {out}")

def plot_pairs_with_rain(df: pd.DataFrame, run_dir: Path):
    if 'Rainfall' not in df.columns or not df['Rainfall'].notna().any():
        return
    cols = [
        ('Rainfall', 'Albedo', 'Rainfall (mm)', 'Albedo'),
        ('Rainfall', 'Temperature', 'Rainfall (mm)', 'Temperature (°C)'),
        ('Rainfall', 'BC_AOD', 'Rainfall (mm)', 'BC AOD'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=160)
    for ax, (xcol, ycol, xlabel, ylabel) in zip(axes, cols):
        x = df[xcol].values
        y = df[ycol].values
        ax.scatter(x, y, s=50, alpha=0.8, color='#8E44AD')
        if len(x) > 1 and np.ptp(x) > 0:
            slope, intercept, r, p, _ = stats.linregress(x, y)
            xx = np.linspace(x.min(), x.max(), 100)
            yy = slope * xx + intercept
            ax.plot(xx, yy, color='#2C3E50', lw=2, label=f'fit: R²={r**2:.3f}, p={p:.3f}')
            ax.legend(loc='best')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        if ycol == 'Albedo':
            ax.set_ylim(0, 1)
    out = run_dir / PAIRS_PLOT_WITH_PRECIP
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved rainfall pairs plot: {out}")

def partial_regression_plot_multi(df: pd.DataFrame, y_col: str, x_col: str, other_cols: List[str], out_path: Path):
    y = df[y_col].values.astype(float)
    X_other = df[other_cols].values.astype(float) if other_cols else np.empty((len(df), 0))
    Xo_const = np.column_stack([np.ones(len(df)), X_other]) if X_other.size else np.ones((len(df), 1))
    beta_y, _, _, _, _, _, _, _ = ols_with_stats(y, Xo_const)
    y_resid = y - (Xo_const @ beta_y)
    x = df[x_col].values.astype(float)
    beta_x, _, _, _, _, _, _, _ = ols_with_stats(x, Xo_const)
    x_resid = x - (Xo_const @ beta_x)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    ax.scatter(x_resid, y_resid, color='#7D3C98', alpha=0.8, s=50)
    if len(x_resid) > 1 and np.ptp(x_resid) > 0:
        slope, intercept, r, p, _ = stats.linregress(x_resid, y_resid)
        xx = np.linspace(x_resid.min(), x_resid.max(), 100)
        yy = slope * xx + intercept
        ax.plot(xx, yy, color='#2C3E50', lw=2, label=f'partial fit: R²={r**2:.3f}, p={p:.3f}')
        ax.legend(loc='best')
    others_label = ", ".join(other_cols) if other_cols else "none"
    ax.set_xlabel(f"{x_col} residuals (controlling {others_label})")
    ax.set_ylabel(f"{y_col} residuals (controlling {others_label})")
    ax.set_title(f"Partial regression: {y_col} ~ {x_col} | {others_label}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved partial regression plot: {out_path}")

def residual_diagnostics_plots(y: np.ndarray, y_hat: np.ndarray, run_dir: Path):
    resid = y - y_hat
    # Residuals vs fitted
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=180)
    ax.scatter(y_hat, resid, color='#34495E', alpha=0.8, s=50)
    ax.axhline(0, color='red', lw=1)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / RESIDUALS_PLOT, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved residuals vs fitted: {run_dir / RESIDUALS_PLOT}")

    # Q-Q plot
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=180)
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title("Residuals Q-Q Plot")
    fig.tight_layout()
    fig.savefig(run_dir / QQ_PLOT, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved Q-Q plot: {run_dir / QQ_PLOT}")

def interaction_simple_slopes_plot(df: pd.DataFrame, beta: np.ndarray, run_dir: Path):
    # Expect beta for model: Intercept, T_c, B_c, T_c×B_c
    temp = df['Temperature']
    bc = df['BC_AOD']
    T_c = temp - temp.mean()
    B_c = bc - bc.mean()
    temp_grid = np.linspace(temp.min(), temp.max(), 100)
    T_c_grid = temp_grid - temp.mean()
    bc_vals = np.percentile(bc, [10, 50, 90])
    Bc_levels = bc_vals - bc.mean()
    colors = ['#1F618D', '#117864', '#A93226']
    labels = [f"BC_AOD p10={bc_vals[0]:.3f}", f"BC_AOD p50={bc_vals[1]:.3f}", f"BC_AOD p90={bc_vals[2]:.3f}"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=180)
    for c, bc_c, lab in zip(colors, Bc_levels, labels):
        y_hat = beta[0] + beta[1]*T_c_grid + beta[2]*bc_c + beta[3]*(T_c_grid*bc_c)
        ax.plot(temp_grid, y_hat, color=c, lw=2, label=lab)
    sc = ax.scatter(temp, df['Albedo'], c=bc, cmap='viridis', edgecolor='k', alpha=0.8)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Albedo (unitless)")
    ax.set_title("Interaction: Temperature × BC_AOD (centered simple slopes)")
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    fig.colorbar(sc, ax=ax, label='BC_AOD')
    fig.tight_layout()
    fig.savefig(run_dir / INTERACTION_PLOT, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved interaction simple slopes plot: {run_dir / INTERACTION_PLOT}")

# =============================================================================
# MAIN
# =============================================================================

def main():
    # Initialize OutputManager and choose the latest run directory
    output_manager = OutputManager(CONFIG['output']['analysis_name'], CONFIG['output']['base_dir'])
    run_dir = find_latest_run_dir(Path(output_manager.base_dir), CONFIG['output']['analysis_name'])
    logger.info(f"Using run directory: {run_dir}")

    # Load/prepare
    loader = DataLoader(CONFIG)
    selector = PixelSelector(CONFIG)
    processor = DataProcessor(CONFIG)

    logger.info("Loading raw datasets...")
    modis, aws, temp, precip = loader.load_haig_data_complete()
    modis_sel = selector.select_best_pixels(modis, 'haig')

    logger.info("Preparing time series and annual series...")
    ts = processor.prepare_time_series_data(modis_sel, aws, temp, 'haig', precip)
    months = CONFIG['trend_analysis']['season_months'] if CONFIG['trend_analysis']['seasonal_analysis'] else None
    annual_mean = processor.create_annual_series(ts, months, CONFIG['trend_analysis']['annual_agg'])

    modis_key = next((k for k in annual_mean if k.startswith('modis_') and k.endswith('_annual')), None)
    if not modis_key:
        raise RuntimeError("No MODIS annual series found.")
    for req in ['temperature_annual', 'bc_aod_annual']:
        if req not in annual_mean:
            raise RuntimeError(f"Missing series: {req}")

    # Build analysis dataframe with optional Rainfall regressor
    df = (annual_mean[modis_key][['date', 'Albedo']]
          .merge(annual_mean['temperature_annual'][['date', 'Temperature']], on='date', how='inner')
          .merge(annual_mean['bc_aod_annual'][['date', 'BC_AOD']], on='date', how='inner'))

    if 'rainfall_annual' in annual_mean:
        df = df.merge(annual_mean['rainfall_annual'][['date', 'Rainfall']], on='date', how='left')

    df = (df.dropna(subset=['Albedo', 'Temperature', 'BC_AOD'])
            .sort_values('date')
            .reset_index(drop=True))

    n = len(df)
    if n < 6:
        logger.warning("Fewer than 6 overlapping annual observations; results may be unstable.")

    # Predictors: baseline and rain-only extension
    baseline_predictors = ['Temperature', 'BC_AOD']
    predictors = baseline_predictors.copy()
    added = []
    if 'Rainfall' in df.columns and df['Rainfall'].notna().any():
        predictors.append('Rainfall')
        added.append('Rainfall')

    # Fit baseline model
    y = df['Albedo'].values.astype(float)
    Xb_pred = df[baseline_predictors].astype(float)
    Xb = np.column_stack([np.ones(n), Xb_pred.values])
    beta_b, se_b, t_b, p_b, r2_b, adj_b, resid_b, yhat_b = ols_with_stats(y, Xb)
    aic_b, bic_b = aic_bic(y, yhat_b, k=Xb.shape[1])
    append_report(run_dir,
                  "Baseline model (no precip): Albedo ~ Temperature + BC_AOD",
                  ['Intercept'] + baseline_predictors,
                  beta_b, se_b, t_b, p_b, r2_b, adj_b, n, compute_vif(Xb_pred))

    # Fit extended model (rain-only, if present)
    X_pred = df[predictors].astype(float).copy()
    X = np.column_stack([np.ones(n), X_pred.values])
    beta, se, tvals, pvals, r2, adj_r2, resid, y_hat = ols_with_stats(y, X)
    aic_e, bic_e = aic_bic(y, y_hat, k=X.shape[1])

    title = "Multiple regression: Albedo ~ " + " + ".join(predictors)
    names = ['Intercept'] + predictors
    append_report(run_dir, title, names, beta, se, tvals, pvals, r2, adj_r2, n, compute_vif(X_pred))

    # Model comparison (extended vs baseline) if Rainfall added
    if len(predictors) > len(baseline_predictors):
        sse_b = float(((y - yhat_b) ** 2).sum())
        sse_e = float(((y - y_hat) ** 2).sum())
        df_b = n - Xb.shape[1]
        df_e = n - X.shape[1]
        F, pF = nested_f_test(sse_b, sse_e, df_b, df_e, X.shape[1]-1, Xb.shape[1]-1)
        extra_lines = [
            "Model comparison (Extended vs Baseline):",
            f"  Δadj R² = {adj_r2 - adj_b:+.3f} (Extended: {adj_r2:.3f}, Baseline: {adj_b:.3f})",
            f"  ΔAIC = {aic_e - aic_b:+.2f} (lower is better)",
            f"  ΔBIC = {bic_e - bic_b:+.2f} (lower is better)",
            f"  Nested F-test: F = {F:.3f}, p = {pF:.3f} (tests if added terms improve fit)"
        ]
        append_report(run_dir, "Extended vs Baseline comparison", [], np.array([]), np.array([]), np.array([]), np.array([]),
                      r2, adj_r2, n, None, extra_lines=extra_lines)

    # Standardized effect sizes for the extended model
    Xz_pred = X_pred.copy()
    for c in Xz_pred.columns:
        Xz_pred[c] = standardize(Xz_pred[c])
    Xz = np.column_stack([np.ones(n), Xz_pred.values])
    b_z, se_z, t_z, p_z, r2_z, adj_r2_z, _, _ = ols_with_stats(y, Xz)
    z_names = ['Intercept'] + [f"{c}_z" for c in X_pred.columns]
    append_report(run_dir,
                  "Standardized coefficients (predictors z-scored)",
                  z_names, b_z, se_z, t_z, p_z, r2_z, adj_r2_z, n, None)

    # Influence: Cook's distance for the extended model
    D = cooks_distance(y, X)
    threshold = 4.0 / n
    idx_high = np.where(D > threshold)[0].tolist()
    # Annotate years for high-influence indices
    years = df['date'].dt.year.values
    high_years = [int(years[i]) for i in idx_high] if idx_high else []
    extra = [f"Cook's distance: threshold = 4/n = {threshold:.3f}",
             f"High-influence indices (0-based) = {idx_high}",
             f"High-influence years = {high_years}"] if idx_high else \
            [f"Cook's distance: threshold = 4/n = {threshold:.3f}", "No high-influence points flagged."]
    append_report(run_dir,
                  "Influence check (Cook's distance) for additive model",
                  ['Intercept'] + predictors,
                  beta, se, tvals, pvals, r2, adj_r2, n, None, extra_lines=extra)

    # Optional refit excluding high-influence points
    b_r = se_r = t_r = p_r = None
    r2_r = adj_r2_r = None
    Xp_refit = None
    if idx_high:
        mask = np.ones(n, dtype=bool)
        mask[idx_high] = False
        y_refit = y[mask]
        Xp_refit = X_pred.iloc[mask].copy()
        X_refit = np.column_stack([np.ones(len(y_refit)), Xp_refit.values])
        b_r, se_r, t_r, p_r, r2_r, adj_r2_r, _, _ = ols_with_stats(y_refit, X_refit)
        append_report(run_dir,
                      "Additive model refit (excluding high Cook's D)",
                      ['Intercept'] + list(Xp_refit.columns),
                      b_r, se_r, t_r, p_r, r2_r, adj_r2_r, len(y_refit), compute_vif(Xp_refit))

    # Residual diagnostics for extended model
    residual_diagnostics_plots(y, y_hat, run_dir)

    # Pairwise plots
    plot_pairs_and_save(df, run_dir)
    plot_pairs_with_rain(df, run_dir)

    # Partial regression plots controlling all other predictors
    other_for_temp = [c for c in predictors if c != 'Temperature']
    other_for_bc = [c for c in predictors if c != 'BC_AOD']
    partial_regression_plot_multi(df, 'Albedo', 'Temperature', other_for_temp, run_dir / PARTIAL_TEMP_PLOT)
    partial_regression_plot_multi(df, 'Albedo', 'BC_AOD', other_for_bc, run_dir / PARTIAL_BC_PLOT)
    # Rainfall partial (if present)
    if 'Rainfall' in predictors:
        others = [c for c in predictors if c != 'Rainfall']
        partial_regression_plot_multi(df, 'Albedo', 'Rainfall', others, run_dir / PARTIAL_PRECIP_PLOT)

    # Centered interaction (secondary check) only for T and BC
    T_c = df['Temperature'].astype(float) - df['Temperature'].astype(float).mean()
    B_c = df['BC_AOD'].astype(float) - df['BC_AOD'].astype(float).mean()
    X_int = pd.DataFrame({'T_c': T_c, 'B_c': B_c})
    X_int['Txc'] = X_int['T_c'] * X_int['B_c']
    X_int_mat = np.column_stack([np.ones(n), X_int['T_c'], X_int['B_c'], X_int['Txc']])
    b_i, se_i, t_i, p_i, r2_i, adj_r2_i, _, _ = ols_with_stats(y, X_int_mat)
    vif_int = compute_vif(X_int[['T_c', 'B_c', 'Txc']])
    append_report(run_dir,
                  "Centered interaction (controls: none beyond T and BC): Albedo ~ T_c + B_c + T_c×B_c",
                  ['Intercept', 'T_c', 'B_c', 'T_c×B_c'],
                  b_i, se_i, t_i, p_i, r2_i, adj_r2_i, n, vif_int)
    interaction_simple_slopes_plot(df, b_i, run_dir)

    # Robustness: trimmed windows using the baseline predictors
    def trim_and_fit_baseline(df_full: pd.DataFrame, drop_first: int = 0, drop_last: int = 0):
        d = df_full.copy()
        if drop_first > 0:
            d = d.iloc[drop_first:].copy()
        if drop_last > 0:
            d = d.iloc[:len(d)-drop_last].copy()
        y_ = d['Albedo'].values.astype(float)
        Xp = d[baseline_predictors].astype(float)
        Xm = np.column_stack([np.ones(len(d)), Xp.values])
        return ols_with_stats(y_, Xm) + (len(d),)

    beta_e, se_e, t_e, p_e, r2_e, adj_e, _, _, n_e = trim_and_fit_baseline(df, drop_first=3, drop_last=0)
    beta_l, se_l, t_l, p_l, r2_l, adj_l, _, _, n_l = trim_and_fit_baseline(df, drop_first=0, drop_last=3)
    append_report(run_dir,
                  "Robustness: trimmed windows (drop first 3 years, baseline predictors)",
                  ['Intercept'] + baseline_predictors,
                  beta_e, se_e, t_e, p_e, r2_e, adj_e, n_e, None)
    append_report(run_dir,
                  "Robustness: trimmed windows (drop last 3 years, baseline predictors)",
                  ['Intercept'] + baseline_predictors,
                  beta_l, se_l, t_l, p_l, r2_l, adj_l, n_l, None)

    # Robustness: median aggregation (baseline predictors)
    annual_median = processor.create_annual_series(ts, months, 'median')
    if modis_key in annual_median and 'temperature_annual' in annual_median and 'bc_aod_annual' in annual_median:
        df_med = (annual_median[modis_key][['date', 'Albedo']]
                  .merge(annual_median['temperature_annual'][['date', 'Temperature']], on='date', how='inner')
                  .merge(annual_median['bc_aod_annual'][['date', 'BC_AOD']], on='date', how='inner')
                  .dropna()
                  .sort_values('date')
                  .reset_index(drop=True))
        if len(df_med) >= 4:
            y_m = df_med['Albedo'].values.astype(float)
            Xp_m = df_med[baseline_predictors].astype(float)
            Xm_m = np.column_stack([np.ones(len(df_med)), Xp_m.values])
            b_m, s_m, t_m, p_m, r2_m, adj_m, _, _ = ols_with_stats(y_m, Xm_m)
            append_report(run_dir,
                          "Robustness: median annual aggregation (JJAS, baseline predictors)",
                          ['Intercept'] + baseline_predictors,
                          b_m, s_m, t_m, p_m, r2_m, adj_m, len(df_med), None)
        else:
            logger.warning("Median-aggregation overlap too small for robustness OLS.")
    else:
        logger.warning("Median-aggregation series missing for robustness OLS.")

    # ---------------- LMG and Partial R² reporting ----------------
    lmg = {}
    pr2_map = {}
    try:
        lmg = lmg_importance(y, X_pred, predictors)
        lines = ["Relative importance (LMG): average ΔR² over all entry orders"]
        for var, (abs_r2, share) in sorted(lmg.items(), key=lambda kv: kv[1][0], reverse=True):
            lines.append(f"  {var:<12} ΔR² = {abs_r2:.3f}  |  share = {share:.1%}")
        append_report(run_dir,
                      "LMG relative importance (additive model)",
                      [], np.array([]), np.array([]), np.array([]), np.array([]),
                      r2, adj_r2, n, None, extra_lines=lines)
    except Exception as e:
        logger.warning(f"LMG computation failed: {e}")

    try:
        X_full = np.column_stack([np.ones(n), X_pred.values])
        for v in predictors:
            pr2_map[v] = partial_r2_for_var(y, X_full, predictors, v, X_pred)
        pr_lines = ["Partial R² (unique contribution):"]
        for v in predictors:
            pr_lines.append(f"  {v:<12} partial R² = {pr2_map.get(v, np.nan):.3f}")
        append_report(run_dir,
                      "Partial R² by predictor (additive model)",
                      [], np.array([]), np.array([]), np.array([]), np.array([]),
                      r2, adj_r2, n, None, extra_lines=pr_lines)
    except Exception as e:
        logger.warning(f"Partial R² computation failed: {e}")

    # ---------------- Summary table: standardized coef + p + partial R² + LMG (%) ----------------
    try:
        # b_z corresponds to: ['Intercept'] + [f"{c}_z" for c in predictors]
        std_map = {c: (b, p) for c, b, p in zip(predictors, b_z[1:], p_z[1:])}
        lines = ["Summary table (standardized effects + Partial R² + LMG share)"]
        lines.append("  predictor     beta_std       p-value    partial R²    LMG share (%)")
        for v in predictors:
            beta_std, pval = std_map.get(v, (np.nan, np.nan))
            pr2 = pr2_map.get(v, np.nan)
            lmg_share_pct = (lmg.get(v, (np.nan, np.nan))[1] * 100.0) if v in lmg else np.nan
            lines.append(f"  {v:<12} {beta_std:>+11.6f}   {pval:>9.3g}     {pr2:>10.3f}     {lmg_share_pct:>11.1f}")
        append_report(run_dir,
                      "Standardized effects + Partial R² + LMG (summary)",
                      [], np.array([]), np.array([]), np.array([]), np.array([]),
                      r2, adj_r2, n, None, extra_lines=lines)
    except Exception as e:
        logger.warning(f"Failed to build standardized + partial R² + LMG summary: {e}")

    # ---------------- Stability section: exclude high Cook's D and recompute LMG/Partial R² and summary table ----------
    if idx_high:
        try:
            lines = []
            lines.append("Stability check (excluding high Cook's D observations)")
            lines.append(f"  Excluded indices = {idx_high}")
            lines.append(f"  Excluded years   = {high_years}")

            # Recompute LMG and Partial R² on the refit sample
            y_refit = y[mask]
            Xp_refit = X_pred.iloc[mask].copy()
            X_refit = np.column_stack([np.ones(len(y_refit)), Xp_refit.values])

            # Standardized coefficients on refit sample
            Xz_refit = Xp_refit.copy()
            for c in Xz_refit.columns:
                Xz_refit[c] = standardize(Xz_refit[c])
            Xz_refit_mat = np.column_stack([np.ones(len(y_refit)), Xz_refit.values])
            b_z_r, se_z_r, t_z_r, p_z_r, r2_z_r, adj_r2_z_r, _, _ = ols_with_stats(y_refit, Xz_refit_mat)

            # LMG refit
            lmg_r = lmg_importance(y_refit, Xp_refit, list(Xp_refit.columns))
            # Partial R² refit
            pr2_r = {}
            X_full_r = np.column_stack([np.ones(len(y_refit)), Xp_refit.values])
            cols_r = list(Xp_refit.columns)
            for v in cols_r:
                pr2_r[v] = partial_r2_for_var(y_refit, X_full_r, cols_r, v, Xp_refit)

            # Build summary table for refit
            std_map_r = {c: (b, p) for c, b, p in zip(cols_r, b_z_r[1:], p_z_r[1:])}
            lines.append("")
            lines.append("  predictor     beta_std       p-value    partial R²    LMG share (%)")
            for v in cols_r:
                beta_std_r, pval_r = std_map_r.get(v, (np.nan, np.nan))
                pr2_val = pr2_r.get(v, np.nan)
                lmg_share_pct_r = (lmg_r.get(v, (np.nan, np.nan))[1] * 100.0) if v in lmg_r else np.nan
                lines.append(f"  {v:<12} {beta_std_r:>+11.6f}   {pval_r:>9.3g}     {pr2_val:>10.3f}     {lmg_share_pct_r:>11.1f}")

            append_report(run_dir,
                          "Stability: standardized effects + Partial R² + LMG (excluding high Cook's D)",
                          [], np.array([]), np.array([]), np.array([]), np.array([]),
                          r2_r if r2_r is not None else np.nan,
                          adj_r2_r if adj_r2_r is not None else np.nan,
                          len(y_refit), None, extra_lines=lines)
        except Exception as e:
            logger.warning(f"Failed stability section (Cook's D exclusion) reporting: {e}")

    logger.info("Multiple regression with Rainfall integration completed successfully.")

if __name__ == "__main__":
    main()