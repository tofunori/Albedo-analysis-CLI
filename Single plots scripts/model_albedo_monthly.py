#!/usr/bin/env python3
"""
Monthly Multiple Regression for Haig Glacier (paper-aligned)
- Monthly aggregation:
  * Albedo (mean), Temperature (mean), BC_AOD (mean), Precipitation (sum)
- MLR: Albedo ~ Temperature (+ Temperature^2 optional) + Precipitation + AOD (BC_AOD)
- Standardized coefficients, VIF, diagnostics, LMG, Partial R²
- Influence: Cook's D and refit without high-D
- Plots: pairs (monthly), partial regressions (monthly)
- Outputs: summary.txt + plots in variant-specific run directory
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Optional: statsmodels for robust HAC
try:
    import statsmodels.api as sm
    _HAS_SM = True
except Exception:
    _HAS_SM = False

from output_manager import OutputManager  # provided by your main project

# =============================================================================
# CONFIG
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("haig_monthly_regression")

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
    'quality_filters': {'min_glacier_fraction': 0.1, 'min_observations': 10},
    'output': {
        'analysis_name': 'haig_monthly_regression',
        'base_dir': 'outputs',
        'summary_template': {}
    },
    # Defaults (overridden by variants)
    'months_filter': [6, 7, 8, 9],  # JJAS
    'precip_column': 'total_precip_mm',
    'add_temp_quad': False,         # add Temperature^2 (centered) predictor
    'hac_lag': None,                # Newey-West lag; None disables HAC output

    # Variants to run
    'variants': [
        {'suffix': 'jjas_totalprecip', 'precip_column': 'total_precip_mm', 'months_filter': [6,7,8,9], 'add_temp_quad': False, 'hac_lag': None},
        {'suffix': 'jjas_snowfall',    'precip_column': 'snowfall_mm',    'months_filter': [6,7,8,9], 'add_temp_quad': False, 'hac_lag': None},
        {'suffix': 'jjas_rainfall',    'precip_column': 'rainfall_mm',    'months_filter': [6,7,8,9], 'add_temp_quad': False, 'hac_lag': None},
        # Examples with T^2 and/or HAC:
        {'suffix': 'jjas_rainfall_t2',  'precip_column': 'rainfall_mm',   'months_filter': [6,7,8,9], 'add_temp_quad': True,  'hac_lag': None},
        {'suffix': 'jjas_rainfall_hac2','precip_column': 'rainfall_mm',   'months_filter': [6,7,8,9], 'add_temp_quad': False, 'hac_lag': 2},
    ],
}

SUMMARY_TXT_NAME = "summary.txt"
PAIRS_PLOT = "monthly_pairs.png"
PARTIAL_TEMP_PLOT = "monthly_partial_temperature.png"
PARTIAL_BC_PLOT = "monthly_partial_bc_aod.png"
PARTIAL_PRECIP_PLOT = "monthly_partial_precip.png"
RESIDUALS_PLOT = "monthly_residuals_vs_fitted.png"
QQ_PLOT = "monthly_residuals_qq_plot.png"

# =============================================================================
# HELPERS
# =============================================================================

def enforce_albedo_bounds(df: pd.DataFrame, col: str = 'Albedo') -> pd.DataFrame:
    if col not in df.columns:
        return df
    return df[(df[col] >= 0) & (df[col] <= 1)].copy()

def to_month_start(date_series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(date_series)
    return pd.to_datetime(dt.dt.to_period('M').dt.to_timestamp())

def prep_ts(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df[['date', value_col]].dropna().copy()
    out['date'] = pd.to_datetime(out['date'])
    out['year'] = out['date'].dt.year
    out['month'] = out['date'].dt.month
    return out.sort_values('date').reset_index(drop=True)

def monthly_agg(df: pd.DataFrame, date_col: str, value_col: str, how: str) -> pd.DataFrame:
    d = df[[date_col, value_col]].dropna().copy()
    if d.empty:
        return pd.DataFrame(columns=['date', value_col, 'n_obs'])
    d['ym'] = pd.to_datetime(d[date_col]).dt.to_period('M')
    grp = d.groupby('ym')[value_col]
    agg = grp.mean() if how == 'mean' else grp.sum()
    out = agg.to_frame(value_col)
    out['n_obs'] = grp.size()
    out = out.reset_index()
    out['date'] = out['ym'].dt.to_timestamp()
    return out[['date', value_col, 'n_obs']].sort_values('date')

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
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if not date_cols:
                raise ValueError("No date column found in MERRA-2 file")
            df['date'] = pd.to_datetime(df[date_cols[0]])
        if 'temperature_c' in df.columns:
            temp_col = 'temperature_c'
        else:
            temp_cols = [c for c in df.columns if 'temp' in c.lower()]
            if not temp_cols:
                raise ValueError("No temperature column found in MERRA-2 file")
            temp_col = temp_cols[0]
        df['Temperature'] = pd.to_numeric(df[temp_col], errors='coerce')
        if 'bc_aod_regional' in df.columns:
            df['BC_AOD'] = pd.to_numeric(df['bc_aod_regional'], errors='coerce')
            out = df[['date', 'Temperature', 'BC_AOD']].copy()
        else:
            out = df[['date', 'Temperature']].copy()
        return out.dropna(subset=['Temperature']).drop_duplicates().sort_values('date').reset_index(drop=True)

    def _load_precip_data(self, file_path: str) -> pd.DataFrame:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.strip()
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            date_cols = [c for c in df.columns if 'date' in c.lower() or 'time' in c.lower()]
            if not date_cols:
                raise ValueError("No date column found in precip (MERRA-2) file")
            df['date'] = pd.to_datetime(df[date_cols[0]])

        prec_choice = self.config.get('precip_column', 'total_precip_mm')
        if prec_choice in df.columns:
            prec_col = prec_choice
        else:
            col_map = {c.lower(): c for c in df.columns}
            if 'total_precip_mm' in df.columns:
                prec_col = 'total_precip_mm'
            elif 'rainfall_mm' in df.columns:
                prec_col = 'rainfall_mm'
            elif 'rain_mm' in col_map:
                prec_col = col_map['rain_mm']
            else:
                raise ValueError("No precipitation column found (expected configured column like 'rainfall_mm').")

        out = pd.DataFrame({'date': df['date']})
        out['Precipitation'] = pd.to_numeric(df[prec_col], errors='coerce')
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
        """
        Select the best MODIS pixel(s) near the AWS based on glacier fraction and distance.
        If the necessary columns are missing, returns the input unchanged.
        """
        if modis_data.empty:
            return modis_data
        station = self.config['aws_stations'].get(glacier_id)
        if station is None:
            return modis_data
        if not {'latitude', 'longitude', 'pixel_id'}.issubset(modis_data.columns):
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
# STATS CORE
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

def hac_se_internal(y: np.ndarray, X: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Newey–West (HAC) SE with Bartlett kernel (internal, dependency-free).
    - Normalizes S by n
    - Centers residuals
    - Clips diagonal to avoid zero/negative numeric artefacts
    """
    n, k = X.shape
    beta, _, _, _, _, _, resid, _ = ols_with_stats(y, X)
    u = resid - resid.mean()
    u = u.reshape(-1, 1)
    XtX = X.T @ X
    XtX_inv = safe_inv(XtX)

    # S matrix
    Xu = X * u
    S = (Xu.T @ Xu) / n  # lag 0, normalized
    L = max(int(lag or 0), 0)
    for l in range(1, L + 1):
        if n - l <= 0:
            break
        w = 1.0 - l / (L + 1.0)  # Bartlett weight
        A = ((X[l:] * u[l:]).T @ X[:-l]) / n
        S += w * (A + A.T)

    cov_hac = XtX_inv @ S @ XtX_inv
    # Replace non-finite by 0; then clip diagonal
    cov_hac = np.where(np.isfinite(cov_hac), cov_hac, 0.0)
    diag = np.diag(cov_hac).copy()
    diag = np.clip(diag, 1e-12, np.inf)
    se_hac = np.sqrt(diag)
    with np.errstate(divide='ignore', invalid='ignore'):
        t_hac = beta / se_hac
    p_hac = 2 * (1 - stats.norm.cdf(np.abs(t_hac)))
    return se_hac, p_hac

def hac_se(y: np.ndarray, X: np.ndarray, lag: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Wrapper: use statsmodels if available, else internal HAC.
    Returns (se_hac, p_hac) for the same coefficient order as X.
    """
    if lag is None or lag <= 0:
        # fall back to OLS SE/p (not robust)
        _, se, tvals, pvals, _, _, _, _ = ols_with_stats(y, X)
        return se, pvals

    if _HAS_SM:
        try:
            model = sm.OLS(y, X)
            res = model.fit(cov_type='HAC', cov_kwds={'maxlags': int(lag), 'use_correction': True})
            beta = res.params
            se_hac = res.bse
            tvals = beta / se_hac
            p_hac = 2 * (1 - stats.norm.cdf(np.abs(tvals)))
            return np.asarray(se_hac), np.asarray(p_hac)
        except Exception as e:
            logger.warning(f"statsmodels HAC failed ({e}); using internal HAC.")
    return hac_se_internal(y, X, lag=lag)

def aic_bic(y: np.ndarray, y_hat: np.ndarray, k: int) -> Tuple[float, float]:
    n = len(y)
    resid = y - y_hat
    sse = float(resid.T @ resid)
    sigma2 = sse / n if n > 0 else np.nan
    ll = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1) if sigma2 > 0 else np.nan
    aic = 2 * k - 2 * ll if ll == ll else np.nan
    bic = np.log(n) * k - 2 * ll if ll == ll else np.nan
    return aic, bic

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

def _r2_from_design(y: np.ndarray, X: np.ndarray) -> float:
    _, _, _, _, r2, _, _, _ = ols_with_stats(y, X)
    return 0.0 if not np.isfinite(r2) else float(r2)

def _design_from_df(df: pd.DataFrame, cols: List[str]) -> np.ndarray:
    M = df[cols].values.astype(float)
    return np.column_stack([np.ones(len(df)), M])

def lmg_importance(y: np.ndarray, df_pred: pd.DataFrame, predictors: List[str]) -> Dict[str, Tuple[float, float]]:
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
    return {v: (contrib[v], (contrib[v] / total if total > 0 else np.nan)) for v in predictors}

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

def append_hac_report(run_dir: Path, title: str, names: List[str],
                      beta: np.ndarray, se_hac: np.ndarray, p_hac: np.ndarray,
                      lag: int):
    lines = []
    lines.append(f"\n{title} (HAC Newey–West, lag={lag})")
    lines.append("-" * max(50, len(title)))
    lines.append("\nCoefficients (same beta), HAC SE and p:")
    lines.append("  term            coef         SE(HAC)      p(HAC)")
    for name, b, s, p in zip(names, beta, se_hac, p_hac):
        lines.append(f"  {name:<12} {b:>+12.6f} {s:>12.6f} {p:>12.3g}")
    lines.append("")
    summary_path = run_dir / SUMMARY_TXT_NAME
    with open(summary_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(lines))
    logger.info(f"Appended HAC report to {summary_path}")

def plot_pairs_and_save(df: pd.DataFrame, run_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=160)
    axes = axes.ravel()
    pairs = [
        ('Temperature', 'Albedo', 'Temperature (°C)', 'Albedo'),
        ('BC_AOD', 'Albedo', 'BC AOD', 'Albedo'),
        ('Precipitation', 'Albedo', 'Precipitation (mm/month)', 'Albedo'),
        ('Temperature', 'BC_AOD', 'Temperature (°C)', 'BC AOD'),
    ]
    colors = ['#3498DB', '#E74C3C', '#27AE60', '#8E44AD']
    for ax, (xcol, ycol, xlabel, ylabel), color in zip(axes, pairs, colors):
        if xcol not in df.columns or ycol not in df.columns:
            ax.axis('off'); continue
        x = df[xcol].values; y = df[ycol].values
        ax.scatter(x, y, color=color, s=35, alpha=0.8)
        if len(x) > 1 and np.ptp(x) > 0:
            slope, intercept, r, p, _ = stats.linregress(x, y)
            xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
            yy = slope * xx + intercept
            ax.plot(xx, yy, color='#2C3E50', lw=2, label=f'R²={r**2:.3f}, p={p:.3f}')
            ax.legend(loc='best')
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
        if 'Albedo' in ycol:
            ax.set_ylim(0, 1)
    out = run_dir / PAIRS_PLOT
    fig.tight_layout()
    fig.savefig(out, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved monthly pairwise plot: {out}")

def partial_regression_plot_multi(df: pd.DataFrame, y_col: str, x_col: str, other_cols: List[str], out_path: Path):
    y = df[y_col].values.astype(float)
    X_other = df[other_cols].values.astype(float) if other_cols else np.empty((len(df), 0))
    Xo_const = np.column_stack([np.ones(len(df)), X_other]) if X_other.size else np.ones((len(df), 1))
    beta_y, _, _, _, _, _, _, _ = ols_with_stats(y, Xo_const)
    y_resid = y - (Xo_const @ beta_y)
    x = df[x_col].values.astype(float)
    beta_x, _, _, _, _, _, _, _ = ols_with_stats(x, Xo_const)
    x_resid = x - (Xo_const @ beta_x)

    fig, ax = plt.subplots(figsize=(7, 5), dpi=160)
    ax.scatter(x_resid, y_resid, color='#7D3C98', alpha=0.8, s=40)
    if len(x_resid) > 1 and np.ptp(x_resid) > 0:
        slope, intercept, r, p, _ = stats.linregress(x_resid, y_resid)
        xx = np.linspace(x_resid.min(), x_resid.max(), 100)
        yy = slope * xx + intercept
        ax.plot(xx, yy, color='#2C3E50', lw=2, label=f'partial: R²={r**2:.3f}, p={p:.3f}')
        ax.legend(loc='best')
    others_label = ", ".join(other_cols) if other_cols else "none"
    ax.set_xlabel(f"{x_col} residuals | {others_label}")
    ax.set_ylabel(f"{y_col} residuals | {others_label}")
    ax.set_title(f"Partial regression: {y_col} ~ {x_col} | {others_label}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logger.info(f"Saved partial regression plot: {out_path}")

def residual_diagnostics_plots(y: np.ndarray, y_hat: np.ndarray, run_dir: Path):
    resid = y - y_hat
    # Residuals vs fitted
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
    ax.scatter(y_hat, resid, color='#34495E', alpha=0.8, s=35)
    ax.axhline(0, color='red', lw=1)
    ax.set_xlabel("Fitted values"); ax.set_ylabel("Residuals")
    ax.set_title("Residuals vs Fitted (monthly)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / RESIDUALS_PLOT, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    # Q-Q
    fig, ax = plt.subplots(figsize=(6.5, 5), dpi=160)
    stats.probplot(resid, dist="norm", plot=ax)
    ax.set_title("Residuals Q-Q (monthly)")
    fig.tight_layout()
    fig.savefig(run_dir / QQ_PLOT, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def durbin_watson(residuals: np.ndarray) -> float:
    num = np.sum(np.diff(residuals) ** 2)
    den = np.sum(residuals ** 2)
    return float(num / den) if den > 0 else np.nan

# =============================================================================
# CORE RUN (single variant)
# =============================================================================

def run_once(config: Dict[str, Any], variant_suffix: Optional[str] = None) -> None:
    analysis_name = config['output']['analysis_name']
    if variant_suffix:
        analysis_name = f"{analysis_name}_{variant_suffix}"

    output_manager = OutputManager(analysis_name, config['output']['base_dir'])
    run_dir = find_latest_run_dir(Path(output_manager.base_dir), analysis_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Using run directory: {run_dir}")

    loader = DataLoader(config)
    selector = PixelSelector(config)

    logger.info("Loading datasets...")
    modis, aws, temp, precip = loader.load_haig_data_complete()
    modis_sel = selector.select_best_pixels(modis, 'haig')

    # Prepare daily time series
    ts_modis: Dict[str, pd.DataFrame] = {}
    if not modis_sel.empty:
        for method in [m for m in modis_sel['method'].unique() if m in config['methods']]:
            dfm = modis_sel[modis_sel['method'] == method][['date', 'Albedo']].dropna().copy()
            dfm = prep_ts(dfm, 'Albedo')
            ts_modis[f'modis_{method.lower()}'] = dfm

    ts_temp = prep_ts(temp, 'Temperature') if not temp.empty else pd.DataFrame()
    ts_bc = prep_ts(temp[['date', 'BC_AOD']].dropna(), 'BC_AOD') if ('BC_AOD' in temp.columns) else pd.DataFrame()
    ts_precip = prep_ts(precip, 'Precipitation') if ('Precipitation' in precip.columns) else pd.DataFrame()

    # Monthly aggregation
    months_filter = config.get('months_filter', None)

    def monthly_series(df: pd.DataFrame, val: str, how: str) -> pd.DataFrame:
        if df.empty: return pd.DataFrame(columns=['date', val, 'n_obs'])
        out = monthly_agg(df, 'date', val, how)
        if months_filter:
            out['month'] = out['date'].dt.month
            out = out[out['month'].isin(months_filter)].drop(columns=['month'])
        return out

    # Choose one MODIS method
    modis_key = next((k for k in ts_modis if k.startswith('modis_')), None)
    if not modis_key:
        raise RuntimeError("No MODIS daily series found.")

    m_alb = monthly_series(ts_modis[modis_key], 'Albedo', 'mean')
    m_temp = monthly_series(ts_temp, 'Temperature', 'mean') if not ts_temp.empty else pd.DataFrame()
    m_bc = monthly_series(ts_bc, 'BC_AOD', 'mean') if not ts_bc.empty else pd.DataFrame()
    m_prec = monthly_series(ts_precip, 'Precipitation', 'sum') if not ts_precip.empty else pd.DataFrame()

    # Merge monthly on 'date'
    dfm = m_alb[['date', 'Albedo']].merge(m_temp[['date', 'Temperature']], on='date', how='inner') \
                                   .merge(m_bc[['date', 'BC_AOD']], on='date', how='inner')
    if not m_prec.empty:
        dfm = dfm.merge(m_prec[['date', 'Precipitation']], on='date', how='left')

    # Optional: Temperature^2 centered
    predictors = ['Temperature', 'BC_AOD']
    if 'Precipitation' in dfm.columns:
        predictors.insert(1, 'Precipitation')
    if config.get('add_temp_quad', False):
        T = dfm['Temperature'].astype(float)
        T2 = (T - T.mean())**2
        dfm['Temperature2'] = T2
        if 'Precipitation' in predictors:
            predictors = ['Temperature', 'Temperature2', 'Precipitation', 'BC_AOD']
        else:
            predictors = ['Temperature', 'Temperature2', 'BC_AOD']

    dfm = dfm.dropna(subset=['Albedo', 'Temperature', 'BC_AOD']).sort_values('date').reset_index(drop=True)

    # Regressions
    y = dfm['Albedo'].values.astype(float)
    n = len(dfm)
    if n < 24:
        logger.warning("Fewer than 24 monthly observations; results may be unstable.")

    # Baseline without precip and without T^2
    base_preds = ['Temperature', 'BC_AOD']
    Xb_pred = dfm[base_preds].astype(float)
    Xb = np.column_stack([np.ones(n), Xb_pred.values])
    beta_b, se_b, t_b, p_b, r2_b, adj_b, resid_b, yhat_b = ols_with_stats(y, Xb)
    append_report(run_dir,
                  "Monthly baseline (no precip): Albedo ~ Temperature + AOD",
                  ['Intercept'] + base_preds, beta_b, se_b, t_b, p_b, r2_b, adj_b, n, compute_vif(Xb_pred))

    # Extended model
    X_pred = dfm[predictors].astype(float)
    X = np.column_stack([np.ones(n), X_pred.values])
    beta, se, tvals, pvals, r2, adj_r2, resid, y_hat = ols_with_stats(y, X)
    append_report(run_dir,
                  "Monthly regression: Albedo ~ " + " + ".join(predictors),
                  ['Intercept'] + predictors, beta, se, tvals, pvals, r2, adj_r2, n, compute_vif(X_pred))

    # HAC (optional)
    if config.get('hac_lag'):
        lag = int(config['hac_lag'])
        se_hac, p_hac = hac_se(y, X, lag)
        append_hac_report(run_dir,
                          "Monthly regression: Albedo ~ " + " + ".join(predictors),
                          ['Intercept'] + predictors, beta, se_hac, p_hac, lag)

    # Model comparison if precip present and no T^2 difference
    if set(predictors) != set(base_preds) and ('Precipitation' in predictors):
        sse_b = float(((y - yhat_b) ** 2).sum())
        sse_e = float(((y - y_hat) ** 2).sum())
        df_e = n - X.shape[1]
        num_df = (X.shape[1]-1) - (Xb.shape[1]-1)
        den_df = df_e
        F = ((sse_b - sse_e) / num_df) / (sse_e / den_df) if (num_df > 0 and den_df > 0 and sse_e > 0) else np.nan
        pF = 1 - stats.f.cdf(F, num_df, den_df) if np.isfinite(F) else np.nan
        aic_b, bic_b = aic_bic(y, yhat_b, Xb.shape[1])
        aic_e, bic_e = aic_bic(y, y_hat, X.shape[1])
        extra_lines = [
            "Model comparison (Extended vs Baseline):",
            f"  Δadj R² = {adj_r2 - adj_b:+.3f}",
            f"  ΔAIC = {aic_e - aic_b:+.2f} (lower is better)",
            f"  ΔBIC = {bic_e - bic_b:+.2f} (lower is better)",
            f"  Nested F-test: F = {F:.3f}, p = {pF:.3f}"
        ]
        append_report(run_dir, "Monthly: Extended vs Baseline", [], np.array([]), np.array([]), np.array([]), np.array([]),
                      r2, adj_r2, n, None, extra_lines=extra_lines)

    # Standardized coefficients (z on predictors only)
    Xz_pred = X_pred.copy()
    for c in Xz_pred.columns:
        Xz_pred[c] = standardize(Xz_pred[c])
    Xz = np.column_stack([np.ones(n), Xz_pred.values])
    b_z, se_z, t_z, p_z, r2_z, adj_r2_z, _, _ = ols_with_stats(y, Xz)
    z_names = ['Intercept'] + [f"{c}_z" for c in X_pred.columns]
    append_report(run_dir, "Monthly standardized coefficients (predictors z-scored)",
                  z_names, b_z, se_z, t_z, p_z, r2_z, adj_r2_z, n, None)

    # Influence check: Cook's D (monthly)
    D = cooks_distance(y, X)
    threshold = 4.0 / n
    idx_high = np.where(D > threshold)[0].tolist()
    years_months = dfm['date'].dt.strftime('%Y-%m').values
    flagged = [years_months[i] for i in idx_high] if idx_high else []
    extra = [f"Cook's distance: threshold = 4/n = {threshold:.3f}",
             f"High-influence indices (0-based) = {idx_high}",
             f"High-influence months = {flagged}"] if idx_high else \
            [f"Cook's distance: threshold = 4/n = {threshold:.3f}", "No high-influence points."]
    append_report(run_dir,
                  "Influence check (Cook's distance) – monthly additive model",
                  ['Intercept'] + predictors, beta, se, tvals, pvals, r2, adj_r2, n, None, extra_lines=extra)

    # Optional refit: exclude high Cook's D
    if idx_high:
        mask = np.ones(n, dtype=bool); mask[idx_high] = False
        y_ref = y[mask]
        Xp_ref = X_pred.iloc[mask].copy()
        X_ref = np.column_stack([np.ones(len(y_ref)), Xp_ref.values])
        b_r, se_r, t_r, p_r, r2_r, adj_r2_r, _, _ = ols_with_stats(y_ref, X_ref)
        append_report(run_dir,
                      "Monthly additive model refit (excluding high Cook's D)",
                      ['Intercept'] + list(Xp_ref.columns),
                      b_r, se_r, t_r, p_r, r2_r, adj_r2_r, len(y_ref), compute_vif(Xp_ref))

    # Residual diagnostics plots
    residual_diagnostics_plots(y, y_hat, run_dir)

    # Durbin–Watson statistic
    dw = durbin_watson(resid)
    append_report(run_dir,
                  "Residual autocorrelation (Durbin–Watson)",
                  [], np.array([]), np.array([]), np.array([]), np.array([]),
                  r2, adj_r2, n, None, extra_lines=[f"DW = {dw:.3f} (≈2 indique faible autocorrélation)"])

    # Plots: pairs and partials
    plot_pairs_and_save(dfm[['Albedo'] + predictors], run_dir)
    if 'Temperature' in predictors:
        partial_regression_plot_multi(dfm[['Albedo'] + predictors], 'Albedo', 'Temperature',
                                      [c for c in predictors if c not in ['Temperature']], run_dir / PARTIAL_TEMP_PLOT)
    if 'BC_AOD' in predictors:
        partial_regression_plot_multi(dfm[['Albedo'] + predictors], 'Albedo', 'BC_AOD',
                                      [c for c in predictors if c != 'BC_AOD'], run_dir / PARTIAL_BC_PLOT)
    if 'Precipitation' in predictors:
        partial_regression_plot_multi(dfm[['Albedo'] + predictors], 'Albedo', 'Precipitation',
                                      [c for c in predictors if c != 'Precipitation'], run_dir / PARTIAL_PRECIP_PLOT)

    # LMG and Partial R²
    try:
        lmg = lmg_importance(y, X_pred, predictors)
        lines = ["Relative importance (LMG, monthly):"]
        for var, (abs_r2, share) in sorted(lmg.items(), key=lambda kv: kv[1][0], reverse=True):
            lines.append(f"  {var:<13} ΔR² = {abs_r2:.3f} | share = {share:.1%}")
        append_report(run_dir, "LMG relative importance (monthly additive model)",
                      [], np.array([]), np.array([]), np.array([]), np.array([]),
                      r2, adj_r2, n, None, extra_lines=lines)
    except Exception as e:
        logger.warning(f"LMG computation failed: {e}")

    try:
        pr2_map = {}
        X_full = np.column_stack([np.ones(n), X_pred.values])
        for v in predictors:
            pr2_map[v] = partial_r2_for_var(y, X_full, predictors, v, X_pred)
        pr_lines = ["Partial R² (monthly):"]
        for v in predictors:
            pr_lines.append(f"  {v:<13} partial R² = {pr2_map.get(v, np.nan):.3f}")
        append_report(run_dir, "Partial R² by predictor (monthly additive model)",
                      [], np.array([]), np.array([]), np.array([]), np.array([]),
                      r2, adj_r2, n, None, extra_lines=pr_lines)
    except Exception as e:
        logger.warning(f"Partial R² computation failed: {e}")

    logger.info("Monthly regression (paper-aligned) completed successfully.")

# =============================================================================
# MAIN
# =============================================================================

def main():
    variants = CONFIG.get('variants', None)
    if not variants:
        run_once(CONFIG, variant_suffix=None)
    else:
        for v in variants:
            cfg = dict(CONFIG)
            cfg['months_filter'] = v.get('months_filter', CONFIG.get('months_filter'))
            cfg['precip_column'] = v.get('precip_column', CONFIG.get('precip_column', 'total_precip_mm'))
            cfg['add_temp_quad'] = v.get('add_temp_quad', CONFIG.get('add_temp_quad', False))
            cfg['hac_lag'] = v.get('hac_lag', CONFIG.get('hac_lag', None))
            run_once(cfg, variant_suffix=v.get('suffix'))

if __name__ == "__main__":
    main()