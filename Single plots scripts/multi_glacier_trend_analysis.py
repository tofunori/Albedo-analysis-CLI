#!/usr/bin/env python3
"""
Multi-Glacier Trend Analysis (configurable)
- Mann–Kendall with optional Yue–Pilon prewhitening
- Sen slope with CI
- Seasonal annual aggregation
- Robust timedelta handling in plotting
- Appends detailed stats into the existing results summary.txt
- Paper-style altitude analysis:
  * August trends by altitude bands relative to glacier median elevation:
    <Median, Median (±100 m), >Median
  * Two groups: Edge Effects (all QC pixels) vs Snow & Ice (glacier_fraction ≥ 0.50)
  * Outputs a Table-2-like block to summary.txt

Supports multiple glaciers: Haig, Athabasca, Coropuna
Change CURRENT_GLACIER variable to switch between glaciers
"""

# =============================================================================
# GLACIER SELECTION - Change this to switch glaciers
# =============================================================================
CURRENT_GLACIER = 'coropuna'  # Options: 'haig', 'athabasca', 'coropuna'

# =============================================================================
# GLACIER CONFIGURATIONS
# =============================================================================
GLACIER_CONFIGS = {
    'haig': {
        'name': 'Haig Glacier',
        'region': 'Canadian Rocky Mountains',
        'aws_coords': {'lat': 50.7186, 'lon': -115.3433, 'name': 'Haig AWS'},
        'data_paths': {
            'modis': "D:/Downloads/MODIS_Terra_Aqua_MultiProduct_2002-01-01_to_2025-01-01.csv",
            'merra2': "D:/Downloads/Haig_Glacier_Climate_JuneSept_Daily_MERRA2_Speciated_AOD - Haig_Glacier_Climate_JuneSept_Daily_MERRA2_Speciated_AOD.csv",
            'aws': "D:/Documents/Projects/Haig_analysis/data/csv/HaigAWS_daily_2002_2015_gapfilled.csv"
        },
        'output_prefix': 'haig',
        'description': f'Non-parametric trend analysis of Haig glacier albedo (2002-2024)'
    },
    'athabasca': {
        'name': 'Athabasca Glacier', 
        'region': 'Canadian Rocky Mountains',
        'aws_coords': {'lat': 52.2, 'lon': -117.2, 'name': 'Athabasca Center'},  # approximate coordinates
        'data_paths': {
            'modis': "D:/Downloads/Athabasca_MODIS_albedo_2002-01-01_to_2025-01-01.csv",
            'merra2': "D:/Downloads/Athabasca_JanDecem_Daily_MERRA2_AOD.csv",
            'aws': None  # No AWS data available
        },
        'output_prefix': 'athabasca',
        'description': f'Non-parametric trend analysis of Athabasca glacier albedo (2002-2024)'
    },
    'coropuna': {
        'name': 'Coropuna Glacier',
        'region': 'Peruvian Andes', 
        'aws_coords': {'lat': -15.5, 'lon': -72.7, 'name': 'Coropuna Center'},  # approximate coordinates
        'data_paths': {
            'modis': "D:/Downloads/Coropuna_MODIS_albedo_2002-2025.csv",
            'merra2': "D:/Downloads/Coropuna_Daily_MERRA2_AOD_2002_2025.csv",
            'aws': None  # No AWS data available
        },
        'output_prefix': 'coropuna',
        'description': f'Non-parametric trend analysis of Coropuna glacier albedo (2002-2024)'
    }
}

# Get current glacier configuration
CURRENT_CONFIG = GLACIER_CONFIGS[CURRENT_GLACIER]

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from output_manager import OutputManager  # your existing module

# =============================================================================
# CONFIG - Dynamic based on selected glacier
# =============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CONFIG = {
    'data_paths': {
        CURRENT_GLACIER: {
            'modis': CURRENT_CONFIG['data_paths']['modis'],
            'aws': CURRENT_CONFIG['data_paths']['aws'],
            'temperature': CURRENT_CONFIG['data_paths']['merra2']
        }
    },
    'aws_stations': {CURRENT_GLACIER: CURRENT_CONFIG['aws_coords']},
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
        'annual_agg': 'mean',
        'weighted_annual': False
    },
    'quality_filters': {'min_glacier_fraction': 0.1, 'min_observations': 10},
    'visualization': {'figsize': (12, 10), 'dpi': 300, 'style': 'seaborn-v0_8'},
    'output': {
        'analysis_name': f'{CURRENT_CONFIG["output_prefix"]}_trend_analysis',
        'base_dir': 'outputs',
        'trends_plot_filename': f'{CURRENT_CONFIG["output_prefix"]}_trends_analysis.png',
        'correlations_plot_filename': f'{CURRENT_CONFIG["output_prefix"]}_correlations_analysis.png',
        'summary_template': {
            'analysis_type': 'Sen Slope and Mann-Kendall Trend Analysis',
            'description': CURRENT_CONFIG['description']
        }
    },
    'sensitivity': {
        'run': True,
        'alt_months': [7, 8, 9],
        'alt_agg': 'median'
    },
    # NEW: paper-style altitude analysis config
    'altitude_paper': {
        'run': True,
        'band_half_width_m': 100,          # ±100 m for "Median" band
        'edge_min_glacier_fraction': 0.50,  # Snow & Ice (edge-free) threshold
        'month': 8                          # August
    }
}

SUMMARY_TXT_NAME = "summary.txt"

# =============================================================================
# HELPERS
# =============================================================================

def duration_years(start, end) -> float:
    return (pd.to_datetime(end) - pd.to_datetime(start)).days / 365.25

def get_value_col(df: pd.DataFrame) -> Optional[str]:
    for c in ('Albedo', 'Temperature', 'BC_AOD'):
        if c in df.columns:
            return c
    return None

def prep_ts(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    out = df[['date', value_col]].dropna().sort_values('date').copy()
    out['year'] = out['date'].dt.year
    out['month'] = out['date'].dt.month
    out['day_of_year'] = out['date'].dt.dayofyear
    return out

def log_series_info(name: str, df: pd.DataFrame, date_col: str = 'date'):
    if not df.empty:
        logger.info(f"{name}: {len(df)} observations from {df[date_col].min()} to {df[date_col].max()}")

def _agg_func(name: str):
    return np.mean if name == 'mean' else np.median

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
    func = _agg_func(agg)
    grp = d.groupby(d[date_col].dt.year)[value_col]
    out = grp.agg(func).to_frame(value_col)
    if return_counts:
        out['n_obs'] = grp.size()
    out = out.reset_index().rename(columns={date_col: 'year'})
    out['date'] = pd.to_datetime(out['year'], format='%Y')
    cols = ['date', value_col] + (['n_obs'] if return_counts else [])
    return out[cols]

def enforce_albedo_bounds(df: pd.DataFrame, col: str = 'Albedo') -> pd.DataFrame:
    if col not in df.columns:
        return df
    mask = (df[col] >= 0) & (df[col] <= 1)
    removed = (~mask).sum()
    if removed:
        logger.warning(f"Removed {removed} rows with out-of-bounds {col}")
    return df[mask].copy()

# =============================================================================
# DATA LOADER
# =============================================================================

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def load_glacier_data_complete(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading complete {CURRENT_CONFIG['name']} dataset...")
        paths = self.config['data_paths'][CURRENT_GLACIER]
        modis = self._load_modis_data(paths['modis'])
        
        # Handle AWS data (may be None for some glaciers)
        if paths['aws'] is not None:
            aws = self._load_aws_data(paths['aws'])
        else:
            logger.info("No AWS data available for this glacier")
            aws = pd.DataFrame(columns=['date', 'Albedo'])
        
        temp = self._load_temperature_data(paths['temperature'])
        logger.info(f"Loaded {len(modis):,} MODIS, {len(aws):,} AWS albedo, and {len(temp):,} temperature records for {CURRENT_CONFIG['name']}")
        return modis, aws, temp

    def _load_modis_data(self, file_path: str) -> pd.DataFrame:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"MODIS data file not found: {file_path}")
        logger.info(f"Loading MODIS data from: {file_path}")
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
        long_df = enforce_albedo_bounds(long_df, 'Albedo')
        return long_df

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
                if sub.empty:
                    continue
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
        if not Path(file_path).exists():
            raise FileNotFoundError(f"AWS data file not found: {file_path}")
        logger.info(f"Loading AWS data from: {file_path}")
        aws = pd.read_csv(file_path, sep=';', skiprows=6, decimal=',')
        aws.columns = aws.columns.str.strip()
        aws = aws.dropna(subset=['Year', 'Day'])
        aws['Year'] = aws['Year'].astype(int)
        aws['Day'] = aws['Day'].astype(int)
        aws['date'] = pd.to_datetime(aws['Year'].astype(str) + '-01-01') + pd.to_timedelta(aws['Day'] - 1, unit='D')
        albedo_cols = [c for c in aws.columns if 'albedo' in c.lower()]
        if not albedo_cols:
            raise ValueError(f"No albedo column found in {CURRENT_CONFIG['name']} AWS data")
        aws['Albedo'] = pd.to_numeric(aws[albedo_cols[0]], errors='coerce')
        aws = aws[['date', 'Albedo']].dropna()
        aws = enforce_albedo_bounds(aws, 'Albedo')
        aws = aws[aws['Albedo'] > 0].drop_duplicates().sort_values('date').reset_index(drop=True)
        return aws

    def _load_temperature_data(self, file_path: str) -> pd.DataFrame:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Temperature data file not found: {file_path}")
        logger.info(f"Loading temperature data from: {file_path}")
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

# =============================================================================
# PIXEL SELECTION
# =============================================================================

class PixelSelector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def select_best_pixels(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        logger.info(f"Applying pixel selection for {glacier_id}...")
        if modis_data.empty:
            logger.warning("Empty MODIS data. Skipping pixel selection.")
            return modis_data

        station = self.config['aws_stations'][glacier_id]
        if not {'latitude', 'longitude'}.issubset(modis_data.columns):
            logger.warning("Latitude/Longitude not found in MODIS data. Returning original data.")
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
            logger.warning("No quality pixels found; using all data")
            return modis_data

        quality['distance_to_aws'] = self._haversine_distance(
            quality['latitude'].values, quality['longitude'].values, station['lat'], station['lon']
        )
        best = quality.sort_values(['avg_glacier_fraction', 'distance_to_aws'], ascending=[False, True]).head(1)
        for _, p in best.iterrows():
            logger.info(f"  Pixel {p['pixel_id']}: glacier_fraction={p.get('avg_glacier_fraction', np.nan):.3f}, "
                        f"distance={p['distance_to_aws']:.2f} km, observations={p['n_observations']}")
        pix_ids = best['pixel_id'].tolist()
        out = modis_data[modis_data['pixel_id'].isin(pix_ids)].copy()
        logger.info(f"Filtered MODIS data from {len(modis_data)} to {len(out)} observations")
        return out

    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371.0
        lat1 = np.radians(lat1); lon1 = np.radians(lon1)
        lat2 = np.radians(lat2); lon2 = np.radians(lon2)
        dlat = lat2 - lat1; dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
        return 2*R*np.arcsin(np.sqrt(a))

# =============================================================================
# TREND ANALYSIS
# =============================================================================

class TrendAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha = config['trend_analysis']['alpha']
        self.use_pw = config['trend_analysis'].get('prewhitening', False)

    def _lag1_ac(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        if len(x) < 3:
            return 0.0
        x0, x1 = x[:-1], x[1:]
        x0 -= np.mean(x0); x1 -= np.mean(x1)
        denom = np.sqrt(np.sum(x0**2) * np.sum(x1**2))
        return 0.0 if denom == 0 else float(np.sum(x0 * x1) / denom)

    def _yue_pilon_prewhiten(self, x: np.ndarray) -> np.ndarray:
        n = len(x)
        if n < 4:
            return x.copy()
        t = np.arange(n, dtype=float)
        slopes = [(x[j] - x[i]) / (t[j] - t[i]) for i in range(n-1) for j in range(i+1, n)]
        s = np.median(slopes)
        r = x - (s * t)
        rho = self._lag1_ac(r)
        r_pw = r[1:] - rho * r[:-1]
        t2 = t[1:]
        x_pw = r_pw + s * (t2 - rho * t[:-1])
        return x_pw

    def mann_kendall_test(self, data: np.ndarray, dates: np.ndarray = None) -> Dict[str, float]:
        x = np.asarray(data, dtype=float)
        if len(x) < 3:
            return {'tau': np.nan, 'p_value': 1.0, 'z_score': 0.0, 'trend': 'no trend', 'significance': False}
        x_eff = self._yue_pilon_prewhiten(x) if self.use_pw else x.copy()
        n = len(x_eff)
        S = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                S += int(x_eff[j] > x_eff[i]) - int(x_eff[j] < x_eff[i])
        unique_vals, counts = np.unique(x_eff, return_counts=True)
        tie_corr = np.sum(counts * (counts - 1) * (2 * counts + 5))
        var_s = (n * (n - 1) * (2 * n + 5) - tie_corr) / 18.0
        z = 0.0 if var_s <= 0 else ((S - 1) if S > 0 else (S + 1) if S < 0 else 0.0) / np.sqrt(var_s)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        tau = S / (0.5 * n * (n - 1))
        trend = 'increasing' if (p <= self.alpha and S > 0) else 'decreasing' if (p <= self.alpha and S < 0) else 'no trend'
        return {'tau': tau, 'p_value': p, 'z_score': z, 'trend': trend, 'significance': p <= self.alpha,
                'S': S, 'var_s': var_s, 'n_effective': n, 'prewhitened': self.use_pw}

    def sen_slope_estimator(self, data: np.ndarray, dates: np.ndarray = None) -> Dict[str, float]:
        x = np.asarray(data, dtype=float)
        n = len(x)
        if n < 2:
            return {'slope': np.nan, 'intercept': np.nan, 'slope_per_year': np.nan, 'confidence_interval': (np.nan, np.nan)}
        if dates is None:
            t = np.arange(n, dtype=float)
        else:
            start = pd.to_datetime(dates[0])
            t = np.array([(pd.to_datetime(d) - start).days / 365.25 for d in dates], dtype=float)
        slopes = [(x[j] - x[i]) / (t[j] - t[i]) for i in range(n - 1) for j in range(i + 1, n) if t[j] != t[i]]
        if not slopes:
            return {'slope': np.nan, 'intercept': np.nan, 'slope_per_year': np.nan, 'confidence_interval': (np.nan, np.nan)}
        slopes = np.array(slopes)
        s = np.median(slopes)
        intercept = np.median(x - s * t)
        m = len(slopes)
        sorted_s = np.sort(slopes)
        c_alpha = stats.norm.ppf(1 - self.alpha / 2) * np.sqrt(n * (n - 1) * (2 * n + 5) / 18.0)
        if c_alpha < m:
            m1 = int(np.floor((m - c_alpha) / 2.0)); m2 = int(np.ceil((m + c_alpha) / 2.0))
            m2 = min(m2, m - 1); m1 = max(m1, 0)
            ci = (sorted_s[m1], sorted_s[m2])
        else:
            ci = (sorted_s[0], sorted_s[-1])
        return {'slope': s, 'intercept': intercept, 'slope_per_year': s, 'confidence_interval': ci, 'n_slopes': m}

    def analyze_time_series(self, data: pd.DataFrame, value_col: str, date_col: str = 'date') -> Dict[str, Any]:
        logger.info(f"Starting comprehensive trend analysis for {value_col}")
        df = data.dropna(subset=[value_col, date_col]).sort_values(date_col).copy()
        vals = df[value_col].values
        dates = pd.to_datetime(df[date_col]).values
        if len(vals) < self.config['trend_analysis']['min_years']:
            logger.warning(f"Insufficient data for trend analysis: {len(vals)} points")
            return {'error': 'Insufficient data for trend analysis'}
        results = {
            'data_info': {
                'n_observations': int(len(vals)),
                'start_date': str(pd.to_datetime(dates[0]).date()),
                'end_date': str(pd.to_datetime(dates[-1]).date()),
                'duration_years': duration_years(dates[0], dates[-1]),
                'annual_counts': df['n_obs'].tolist() if 'n_obs' in df.columns else None
            },
            'annual': {
                'mann_kendall': self.mann_kendall_test(vals, dates),
                'sen_slope': self.sen_slope_estimator(vals, dates)
            }
        }
        logger.info("Trend analysis completed successfully")
        return results

# =============================================================================
# NEW: Paper-style altitude helpers
# =============================================================================

def assign_altitude_band(df: pd.DataFrame, band_hw_m: int) -> Tuple[pd.DataFrame, float]:
    if 'elevation' not in df.columns or df['elevation'].isna().all():
        out = df.copy()
        out['band'] = 'Median'  # fallback
        return out, np.nan
    elev_med = float(df['elevation'].median())
    lo = elev_med - band_hw_m
    hi = elev_med + band_hw_m
    band = np.where(df['elevation'] > hi, '>Median',
            np.where(df['elevation'] < lo, '<Median', 'Median'))
    out = df.copy()
    out['band'] = band
    return out, elev_med

def qc_albedo_edges(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if 'Albedo' not in d.columns and 'albedo' in d.columns:
        d = d.rename(columns={'albedo': 'Albedo'})
    d = d[d['Albedo'].between(0.05, 0.99)]
    d['date'] = pd.to_datetime(d['date'])
    return d

def august_annual_series_by_band(modis: pd.DataFrame,
                                 month: int,
                                 band_hw_m: int,
                                 min_gf: float,
                                 edge_free: bool) -> Tuple[Dict[str, pd.DataFrame], float]:
    """
    Returns annual August albedo series for each band: '<Median', 'Median', '>Median'.
    edge_free=True => 'Snow & Ice' using glacier_fraction >= min_gf.
    """
    d = qc_albedo_edges(modis)
    if edge_free and 'glacier_fraction' in d.columns:
        d = d[d['glacier_fraction'] >= min_gf]
    d['month'] = d['date'].dt.month
    d = d[d['month'] == month]
    if d.empty:
        return {}, np.nan
    d, elev_med = assign_altitude_band(d, band_hw_m)
    out: Dict[str, pd.DataFrame] = {}
    for band in ['>Median', 'Median', '<Median']:
        sub = d[d['band'] == band]
        if sub.empty:
            continue
        ann = (sub.assign(year=sub['date'].dt.year)
                  .groupby('year', as_index=False)['Albedo'].mean())
        ann['date'] = pd.to_datetime(ann['year'], format='%Y')
        out[band] = ann[['date', 'Albedo']].sort_values('date').reset_index(drop=True)
    return out, elev_med

def sen_mk_for_series(df: pd.DataFrame, analyzer: 'TrendAnalyzer', value_col: str = 'Albedo') -> Dict[str, Any]:
    if df is None or df.empty or value_col not in df.columns:
        return {'sen': np.nan, 'mk_p': np.nan, 'n': 0}
    res = analyzer.analyze_time_series(df[['date', value_col]], value_col)
    if 'error' in res:
        return {'sen': np.nan, 'mk_p': np.nan, 'n': len(df)}
    sen = res['annual']['sen_slope']['slope_per_year']
    p = res['annual']['mann_kendall']['p_value']
    return {'sen': float(sen), 'mk_p': float(p), 'n': int(len(df))}

def format_altitude_analysis(elev_med: float,
                            edge_res: Dict[str, Dict[str, Any]],
                            pure_res: Dict[str, Dict[str, Any]],
                            month: int,
                            min_gf: float) -> str:
    """Format altitude analysis results as a string for inclusion in summary."""
    def fmt(v):
        return "NA" if v is None or not np.isfinite(v) else f"{v:+.3f}"
    
    lines = []
    lines.append("August Trends by Altitude Band (Paper-style Analysis)")
    lines.append("=" * 60)
    lines.append(f"Month: {month}, Snow & Ice threshold (glacier_fraction) ≥ {min_gf:.2f}")
    lines.append(f"Median elevation: {('NA' if not np.isfinite(elev_med) else f'{elev_med:.1f} m')}")
    lines.append("")
    lines.append("Edge Effects (all QC pixels):")
    for band in ['>Median','Median','<Median']:
        r = edge_res.get(band, {})
        lines.append(f"  {band:<8}  Sen slope = {fmt(r.get('sen'))}/yr  |  MK p = {fmt(r.get('mk_p'))}  |  n_years = {r.get('n', 0)}")
    lines.append("")
    lines.append("Snow & Ice (edge-free pixels):")
    for band in ['>Median','Median','<Median']:
        r = pure_res.get(band, {})
        lines.append(f"  {band:<8}  Sen slope = {fmt(r.get('sen'))}/yr  |  MK p = {fmt(r.get('mk_p'))}  |  n_years = {r.get('n', 0)}")
    lines.append("")
    
    return "\n".join(lines)

# =============================================================================
# DATA PROCESSOR
# =============================================================================

class DataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def prepare_time_series_data(self, modis: pd.DataFrame, aws: pd.DataFrame, temp: pd.DataFrame, glacier_id: str) -> Dict[str, pd.DataFrame]:
        logger.info(f"Preparing time series data for {glacier_id} trend analysis...")
        out: Dict[str, pd.DataFrame] = {}
        if not aws.empty:
            out['aws_albedo'] = prep_ts(aws, 'Albedo'); log_series_info('AWS albedo time series', out['aws_albedo'])
        if not temp.empty:
            if 'Temperature' in temp.columns:
                out['temperature'] = prep_ts(temp, 'Temperature'); log_series_info('Temperature time series', out['temperature'])
            if 'BC_AOD' in temp.columns:
                out['bc_aod'] = prep_ts(temp, 'BC_AOD'); log_series_info('BC AOD time series', out['bc_aod'])
        if modis is not None and not modis.empty:
            for method in [m for m in modis['method'].unique() if m in self.config['methods']]:
                df = modis[modis['method'] == method][['date', 'Albedo']].dropna().sort_values('date')
                df = prep_ts(df, 'Albedo')
                key = f'modis_{method.lower()}'
                out[key] = df
                log_series_info(f"MODIS {method} time series", df)
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
                logger.warning(f"{name} has no data after seasonal filter; skipping")
                continue
            key = f"{name}_annual"
            annual[key] = out
            logger.info(f"{name} annual series: {len(out)} years (median N={int(np.median(out['n_obs'])) if 'n_obs' in out.columns else 'NA'})")
        return annual

# =============================================================================
# VISUALIZATION
# =============================================================================

class TrendVisualizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def _get_sig(self, p: float) -> str:
        return '**' if p < 0.01 else '*' if p < 0.05 else ''

    # Robust timedelta-to-years conversion
    def _sen_line(self, x, y, slope, ci=None):
        xdt = pd.to_datetime(x)
        td = (xdt - xdt.min())
        t = (td / np.timedelta64(1, 'D')).astype(float) / 365.25
        t_line = np.array([t.min(), t.max()], dtype=float)
        y_med, t_med = np.median(y), np.median(t)
        intercept = y_med - slope * t_med
        y_line = slope * t_line + intercept
        ci_lines = None
        if ci and len(ci) == 2:
            s_lo, s_hi = ci
            y_lo = s_lo * t_line + (y_med - s_lo * t_med)
            y_hi = s_hi * t_line + (y_med - s_hi * t_med)
            ci_lines = (y_lo, y_hi)
        x_line = xdt.min() + pd.to_timedelta(t_line * 365.25, unit='D')
        return x_line, y_line, ci_lines

    def _slope_label(self, slope, y, unit):
        abs_lab = f"{slope:+.4f}/yr"
        if np.all(y > 0):
            m = float(np.mean(y))
            if m > 0:
                rel = 100.0 * slope / m
                return f"{abs_lab} ({rel:+.1f}%/yr)"
        return f"{abs_lab}"

    def _plot_trend(self, ax, data, res, color, label, ylabel, unit, ylim=None, show_data_legend=True):
        y = data.iloc[:, 1].values
        data_label = label if show_data_legend else None
        ax.scatter(data['date'], y, color=color, alpha=0.8, s=22, label=data_label, edgecolor='none', zorder=3)
        if 'sen_slope' in res:
            slope = res['sen_slope']['slope_per_year']
            ci = res['sen_slope']['confidence_interval']
            x_line, y_line, ci_lines = self._sen_line(data['date'], y, slope, ci)
            trend_label = 'Trend line' if show_data_legend else None
            ax.plot(x_line, y_line, color='#333333', linewidth=1.8, label=trend_label, zorder=2)
            if ci_lines is not None:
                ax.fill_between(x_line, ci_lines[0], ci_lines[1], color=color, alpha=0.15, zorder=1)
        # Create consolidated statistics box
        stats_text = ""
        if 'mann_kendall' in res:
            mk = res['mann_kendall']
            p, tau = mk['p_value'], mk['tau']
            tag = ' (PW)' if mk.get('prewhitened', False) else ''
            stats_text += f'τ={tau:.3f}{self._get_sig(p)}  p={p:.3f}{tag}\n'
        
        if 'sen_slope' in res:
            slope = res['sen_slope']['slope_per_year']
            ci = res['sen_slope']['confidence_interval']
            # Calculate percentage change for better context
            if np.all(y > 0):
                m = float(np.mean(y))
                if m > 0:
                    rel = 100.0 * slope / m
                    stats_text += f'Sen slope: {slope:+.4f}/yr ({rel:+.1f}%/yr)\n'
                else:
                    stats_text += f'Sen slope: {slope:+.4f}/yr\n'
            else:
                stats_text += f'Sen slope: {slope:+.4f}/yr\n'
            
            # Add confidence interval if available
            if ci and len(ci) == 2 and np.isfinite(ci).all():
                stats_text += f'95% CI: [{ci[0]:+.4f}, {ci[1]:+.4f}]'
        
        if stats_text:
            ax.text(0.02, 0.98, stats_text.strip(), transform=ax.transAxes,
                    va='top', ha='left', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, linewidth=0.5))
        if 'n_obs' in data.columns:
            ax.text(0.98, 0.02, f"N/yr median={int(np.median(data['n_obs']))}", ha='right', va='bottom', transform=ax.transAxes, fontsize=8)
        ax.set_xlabel('Year'); ax.set_ylabel(ylabel)
        # Only show legend if there are items to display
        if show_data_legend:
            ax.legend(loc='lower left', bbox_to_anchor=(0.01, 0.01), frameon=True, fancybox=True, framealpha=0.85, borderpad=0.3, handlelength=1.2, handletextpad=0.4, fontsize=9, ncol=1)
        ax.grid(True, alpha=0.3)
        if ylim: ax.set_ylim(*ylim)

    def create_trends_plot(self, trend_results: Dict[str, Any], time_series: Dict[str, pd.DataFrame], output_path: Optional[str] = None) -> plt.Figure:
        logger.info("Creating focused trends plot...")
        try: plt.style.use(self.config['visualization']['style'])
        except Exception: plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
        side = 4.5
        fig, axes = plt.subplots(1, 3, figsize=(side*3, side), dpi=max(300, int(self.config['visualization']['dpi'])))
        
        # Dynamic title with current glacier
        fig.suptitle(f'Mann–Kendall Trend Analysis: {CURRENT_CONFIG["name"]} (2002–2024)', fontsize=13, fontweight='bold', y=0.98)

        modis_key = next((k for k in time_series if k.startswith('modis_') and k.endswith('_annual') and k in trend_results), None)
        specs = []
        if modis_key:
            specs.append(dict(key=modis_key, color='#0072B2', label='MODIS albedo', ylabel='Albedo (unitless)', unit='units', ylim=(0, 1)))
        if 'temperature_annual' in time_series and 'temperature_annual' in trend_results:
            specs.append(dict(key='temperature_annual', color='#D55E00', label='Melt-season temperature', ylabel='Temperature (°C)', unit='°C'))
        if 'bc_aod_annual' in time_series and 'bc_aod_annual' in trend_results:
            specs.append(dict(key='bc_aod_annual', color='#009E73', label='Melt-season BC AOD', ylabel='BC AOD', unit='units'))

        for ax, spec in zip(axes, specs):
            data = time_series[spec['key']]
            res = trend_results[spec['key']]['annual']
            # Hide all legend items to clean up the plot - statistics are now in the info box
            show_data_legend = False
            self._plot_trend(ax, data, res, spec['color'], spec['label'], spec['ylabel'], spec['unit'], spec.get('ylim'), show_data_legend)
        for i in range(len(specs), 3):
            axes[i].set_visible(False)

        fig.text(0.99, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=8, ha='right')
        plt.tight_layout(rect=[0, 0.04, 1, 0.96], w_pad=1.2, h_pad=0.6)
        if output_path:
            fig.savefig(output_path, dpi=max(300, int(self.config['visualization']['dpi'])), bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Trends plot saved to: {output_path}")
        return fig

    def _corr_plot(self, ax, x, y, xlabel, ylabel, color, title):
        ax.scatter(x, y, color=color, alpha=0.8, s=22, edgecolor='none', zorder=3)
        if len(x) > 2:
            slope, intercept, r, p, _ = stats.linregress(x, y)
            lx = np.array([np.min(x), np.max(x)]); ly = slope * lx + intercept
            ax.plot(lx, ly, color='#333333', linewidth=1.8, zorder=2)
            
            # Consolidated statistics box with correlation and fit information
            sig_stars = "**" if p < 0.01 else "*" if p < 0.05 else ""
            stats_text = f'R={r:.3f}{sig_stars}  p={p:.3f}  n={len(x)}\nFit: R²={r**2:.3f}\nSlope={slope:.4f}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    va='top', ha='left', fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, linewidth=0.5))
        ax.set_title(title, fontsize=12, fontweight='bold'); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if 'albedo' in ylabel.lower(): ax.set_ylim(0, 1)

    def create_correlations_plot(self, time_series: Dict[str, pd.DataFrame], output_path: Optional[str] = None) -> plt.Figure:
        logger.info("Creating focused correlations plot...")
        try: plt.style.use(self.config['visualization']['style'])
        except Exception: plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({
            'font.family': 'DejaVu Sans',
            'axes.titlesize': 12,
            'axes.labelsize': 11,
            'legend.fontsize': 10,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10
        })
        side = 4.5
        fig, axes = plt.subplots(1, 3, figsize=(side*3, side), dpi=max(300, int(self.config['visualization']['dpi'])))
        
        # Dynamic title with current glacier
        fig.suptitle(f'Climate Variables Correlation Analysis: {CURRENT_CONFIG["name"]} (2002–2024)', fontsize=13, fontweight='bold', y=0.98)

        modis_key = next((k for k in time_series if k.startswith('modis_') and k.endswith('_annual')), None)
        if modis_key and 'temperature_annual' in time_series:
            merged = pd.merge(time_series[modis_key], time_series['temperature_annual'], on='date')
            if len(merged): self._corr_plot(axes[0], merged['Temperature'].values, merged['Albedo'].values, 'Temperature (°C)', 'Albedo (unitless)', '#0072B2', 'Temperature vs Albedo')
        else: axes[0].set_visible(False)
        if modis_key and 'bc_aod_annual' in time_series:
            merged = pd.merge(time_series[modis_key], time_series['bc_aod_annual'], on='date')
            if len(merged): self._corr_plot(axes[1], merged['BC_AOD'].values, merged['Albedo'].values, 'BC AOD', 'Albedo (unitless)', '#D55E00', 'BC AOD vs Albedo')
        else: axes[1].set_visible(False)
        if 'temperature_annual' in time_series and 'bc_aod_annual' in time_series:
            merged = pd.merge(time_series['temperature_annual'], time_series['bc_aod_annual'], on='date')
            if len(merged): self._corr_plot(axes[2], merged['Temperature'].values, merged['BC_AOD'].values, 'Temperature (°C)', 'BC AOD', '#009E73', 'Temperature vs BC AOD')
        else: axes[2].set_visible(False)

        fig.text(0.99, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', fontsize=8, ha='right')
        plt.tight_layout(rect=[0, 0.04, 1, 0.96], w_pad=1.2, h_pad=0.6)
        if output_path:
            fig.savefig(output_path, dpi=max(300, int(self.config['visualization']['dpi'])), bbox_inches='tight', facecolor='white', edgecolor='none')
            logger.info(f"Correlations plot saved to: {output_path}")
        return fig

# =============================================================================
# SUMMARY & README
# =============================================================================

def summarize_results(results: Dict[str, Any]) -> Dict[str, Any]:
    stats_map = {}
    for k, res in results.items():
        if 'error' in res:
            continue
        a = res.get('annual', {})
        mk, sen = a.get('mann_kendall', {}), a.get('sen_slope', {})
        stats_map[k] = {
            'n_observations': res.get('data_info', {}).get('n_observations', 0),
            'duration_years': res.get('data_info', {}).get('duration_years', 0),
            'mann_kendall_trend': mk.get('trend', 'no trend'),
            'statistical_significance': mk.get('significance', False),
            'p_value': mk.get('p_value', 1.0),
            'kendall_tau': mk.get('tau', 0.0),
            'prewhitened': mk.get('prewhitened', False),
            'sen_slope_per_year': sen.get('slope_per_year', 0.0),
            'slope_confidence_interval': sen.get('confidence_interval', (np.nan, np.nan))
        }
    return stats_map

# Function removed - all content now goes into comprehensive save_summary

def generate_comprehensive_summary(output_manager: OutputManager,
                                   trend_results: Dict[str, Any],
                                   time_series: Dict[str, pd.DataFrame],
                                   sensitivity_notes: List[str],
                                   altitude_analysis: str = ""):
    try:
        stats_map = summarize_results(trend_results)
        significant = sum(1 for s in stats_map.values() if s['statistical_significance'])

        # Build comprehensive summary with all information
        summary = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': CONFIG['output']['summary_template']['analysis_type'],
            'glacier': f'{CURRENT_CONFIG["name"]}, {CURRENT_CONFIG["region"]}',
            'configuration': {
                'trend_analysis_alpha': CONFIG['trend_analysis']['alpha'],
                'seasonal_analysis': CONFIG['trend_analysis']['seasonal_analysis'],
                'prewhitening': CONFIG['trend_analysis']['prewhitening'],
                'min_years': CONFIG['trend_analysis']['min_years'],
                'methods': CONFIG['methods'],
                'season_months': CONFIG['trend_analysis']['season_months'],
                'annual_agg': CONFIG['trend_analysis']['annual_agg'],
                'weighted_annual': CONFIG['trend_analysis']['weighted_annual'],
                'methodology': 'Mann–Kendall trend test (optional Yue–Pilon prewhitening) and Sen slope estimator',
                'statistical_significance_level': f"alpha = {CONFIG['trend_analysis']['alpha']} (95% confidence level)",
                'data_sources': 'AWS measurements and MODIS satellite observations',
                'quality_control': f"Minimum {CONFIG['trend_analysis']['min_years']} years of data required"
            },
            'data_info': {
                'total_time_series': len(stats_map),
                'significant_trends': significant,
                'analysis_period': '2002-2024 (based on available data)',
                'seasonal_definition': f"Melt season months: {CONFIG['trend_analysis']['season_months']}; aggregation: {CONFIG['trend_analysis']['annual_agg']}"
            },
            'detailed_statistics': stats_map,
            'sensitivity_notes': sensitivity_notes,
            'altitude_analysis': altitude_analysis
        }
        
        # Save comprehensive summary (replaces both JSON summary and README)
        output_manager.save_summary(summary)
        
        logger.info("Comprehensive summary generated with all statistical results, altitude analysis, and metadata")

    except Exception as e:
        logger.error(f"Error generating trend summary and README: {e}")

# =============================================================================
# MAIN
# =============================================================================

def run_sensitivity(processor: DataProcessor, ts: Dict[str, pd.DataFrame], analyzer: TrendAnalyzer) -> List[str]:
    notes = []
    if not CONFIG['sensitivity'].get('run', False):
        return notes
    alt_months = CONFIG['sensitivity'].get('alt_months', [7, 8, 9])
    alt_agg = CONFIG['sensitivity'].get('alt_agg', 'median')
    annual_jas = processor.create_annual_series(ts, alt_months, CONFIG['trend_analysis']['annual_agg'])
    annual_med = processor.create_annual_series(ts, CONFIG['trend_analysis']['season_months'], alt_agg)

    def _get_modis_key(d: Dict[str, pd.DataFrame]) -> Optional[str]:
        return next((k for k in d if k.startswith('modis_') and k.endswith('_annual')), None)

    for lbl, annual_set in [('JAS', annual_jas), (f"{alt_agg}", annual_med)]:
        mk = _get_modis_key(annual_set)
        if mk:
            res = analyzer.analyze_time_series(annual_set[mk], 'Albedo')
            if 'error' not in res:
                slope = res['annual']['sen_slope']['slope_per_year']
                p = res['annual']['mann_kendall']['p_value']
                notes.append(f"Sensitivity {lbl}: slope={slope:+.4f} units/yr, p={p:.3f}")

    return notes

def main():
    logger.info(f"Starting Sen Slope and Mann-Kendall Trend Analysis for {CURRENT_CONFIG['name']}")
    output_manager = OutputManager(CONFIG['output']['analysis_name'], CONFIG['output']['base_dir'])
    loader = DataLoader(CONFIG)
    selector = PixelSelector(CONFIG)
    processor = DataProcessor(CONFIG)
    analyzer = TrendAnalyzer(CONFIG)
    viz = TrendVisualizer(CONFIG)

    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {CURRENT_CONFIG['name'].upper()} - Trend Analysis")
        logger.info(f"{'='*60}")

        modis, aws, temp = loader.load_glacier_data_complete()
        modis_sel = selector.select_best_pixels(modis, CURRENT_GLACIER)
        ts = processor.prepare_time_series_data(modis_sel, aws, temp, CURRENT_GLACIER)

        months = CONFIG['trend_analysis']['season_months'] if CONFIG['trend_analysis']['seasonal_analysis'] else None
        annual = processor.create_annual_series(ts, months, CONFIG['trend_analysis']['annual_agg'])
        all_ts = {**ts, **annual}

        results: Dict[str, Any] = {}
        if 'aws_albedo_annual' in annual:
            logger.info("Analyzing AWS albedo annual trends...")
            results['aws_albedo_annual'] = analyzer.analyze_time_series(annual['aws_albedo_annual'], 'Albedo')
        if 'temperature_annual' in annual:
            logger.info("Analyzing melt season temperature trends...")
            results['temperature_annual'] = analyzer.analyze_time_series(annual['temperature_annual'], 'Temperature')
        if 'bc_aod_annual' in annual:
            logger.info("Analyzing melt season BC AOD trends...")
            results['bc_aod_annual'] = analyzer.analyze_time_series(annual['bc_aod_annual'], 'BC_AOD')

        modis_keys = [k for k in annual.keys() if 'modis' in k and 'annual' in k]
        logger.info(f"Available annual data keys: {list(annual.keys())}")
        logger.info(f"MODIS keys found: {modis_keys}")
        for mk in modis_keys:
            logger.info(f"Analyzing {mk} trends...")
            results[mk] = analyzer.analyze_time_series(annual[mk], 'Albedo')

        # ===== NEW: Altitude analysis "comme dans le papier" (Août par bandes) =====
        if CONFIG.get('altitude_paper', {}).get('run', False):
            logger.info("Running August altitude-band trends (paper-style)...")
            alt_cfg = CONFIG['altitude_paper']
            month = int(alt_cfg.get('month', 8))
            band_hw = int(alt_cfg.get('band_half_width_m', 100))
            min_gf = float(alt_cfg.get('edge_min_glacier_fraction', 0.50))

            # Edge Effects = all QC pixels; Snow & Ice = glacier_fraction ≥ min_gf
            edge_bands, elev_med = august_annual_series_by_band(modis_sel, month, band_hw, min_gf, edge_free=False)
            pure_bands, _ = august_annual_series_by_band(modis_sel, month, band_hw, min_gf, edge_free=True)

            edge_res, pure_res = {}, {}
            for band, df_band in edge_bands.items():
                edge_res[band] = sen_mk_for_series(df_band, analyzer, 'Albedo')
            for band, df_band in pure_bands.items():
                pure_res[band] = sen_mk_for_series(df_band, analyzer, 'Albedo')

            altitude_analysis_text = format_altitude_analysis(elev_med, edge_res, pure_res, month, min_gf)

        logger.info(f"\n{'='*60}")
        logger.info("Creating Enhanced Trend and Correlation Visualizations")
        logger.info(f"{'='*60}")
        trends_path = output_manager.get_plot_path(CONFIG['output']['trends_plot_filename'])
        _ = viz.create_trends_plot(results, annual, str(trends_path))
        output_manager.log_file_saved(trends_path, "plot")
        corr_path = output_manager.get_plot_path(CONFIG['output']['correlations_plot_filename'])
        _ = viz.create_correlations_plot(annual, str(corr_path))
        output_manager.log_file_saved(corr_path, "plot")

        sensitivity_notes = run_sensitivity(processor, ts, analyzer)
        
        # Get altitude analysis if it was performed
        altitude_text = locals().get('altitude_analysis_text', "")

        logger.info("Displaying plots...")
        plt.show()
        generate_comprehensive_summary(output_manager, results, all_ts, sensitivity_notes, altitude_text)

        logger.info(f"\nSUCCESS: {CURRENT_CONFIG['name']} trend analysis completed and saved")
        logger.info("Analysis period: 2002-2024")
        logger.info("Trend analysis methods: Mann-Kendall test (prewhitening={}) and Sen slope estimator".format(CONFIG['trend_analysis']['prewhitening']))

    except Exception as e:
        logger.error(f"Error in {CURRENT_CONFIG['name']} trend analysis: {e}")
        raise

if __name__ == "__main__":
    main()