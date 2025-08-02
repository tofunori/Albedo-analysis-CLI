#!/usr/bin/env python3
"""
Advanced Albedo Analysis with Multi-Scale Glaciological Methods
=============================================================

Comprehensive analysis extending multiple regression with:
- Multi-scale temporal analysis (Daily, JJA, Decadal)
- Non-linear temperature models (Hinge, PDD, Smoothing)
- Fresh snow indicators and interactions
- Advanced deposition proxies
- Benchmark validation against Nature paper (Δα/ΔT ≈ -0.04 per +1°C)

Author: Claude Code Analysis System
Date: 2025-08-02
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from scipy import stats
from scipy.stats import pearsonr, spearmanr
# Removed sklearn dependencies - using numpy/scipy only for consistency
import warnings
warnings.filterwarnings('ignore')

# Import the base analyzer
from multiple_regression_analysis import MultipleRegressionAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AdvancedAlbedoAnalyzer(MultipleRegressionAnalyzer):
    """
    Advanced albedo analysis with multi-scale temporal analysis, 
    non-linear temperature models, and glaciological process understanding.
    """
    
    def __init__(self, output_base_dir: str = "outputs"):
        """Initialize the advanced albedo analyzer."""
        super().__init__(output_base_dir)
        
        # Create additional output directories for advanced analysis
        (self.output_dir / "multi_scale_analysis").mkdir(exist_ok=True)
        (self.output_dir / "nonlinear_models").mkdir(exist_ok=True)
        (self.output_dir / "fresh_snow_analysis").mkdir(exist_ok=True)
        (self.output_dir / "benchmark_validation").mkdir(exist_ok=True)
        
        # Benchmark values from literature
        self.BENCHMARK_DALPHA_DT = -0.04  # Nature paper benchmark
        self.FRESH_SNOW_THRESHOLD = 1.0   # mm for significant snowfall
        
        logger.info(f"Initialized Advanced Albedo Analyzer with benchmark Δα/ΔT = {self.BENCHMARK_DALPHA_DT} per +1°C")
    
    def create_multi_scale_temporal_datasets(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create datasets at different temporal scales:
        - Daily: Original data
        - JJA Annual: June-July-August means by year
        - Decadal: 2002-2011 vs 2012-2024 comparison
        """
        logger.info("Creating multi-scale temporal datasets...")
        
        datasets = {}
        
        # Daily scale (original data)
        datasets['daily'] = data.copy()
        
        # JJA Annual scale
        logger.info("Creating JJA annual aggregation...")
        data_jja = data[data['date'].dt.month.isin([6, 7, 8])].copy()
        data_jja['year'] = data_jja['date'].dt.year
        
        jja_annual = data_jja.groupby('year').agg({
            'bc_aod_regional': 'mean',
            'temperature_c': 'mean',
            'total_precip_mm': 'sum',
            'snowfall_mm': 'sum',
            'rainfall_mm': 'sum',
            'albedo_mean': 'mean'
        }).reset_index()
        
        # Add date for compatibility
        jja_annual['date'] = pd.to_datetime(jja_annual['year'].astype(str) + '-07-15')
        jja_annual['month'] = 7
        jja_annual['day_of_year'] = 196  # Mid-July
        
        datasets['jja_annual'] = jja_annual
        logger.info(f"Created JJA annual dataset with {len(jja_annual)} years")
        
        # Decadal comparison
        logger.info("Creating decadal comparison datasets...")
        decade1 = data_jja[data_jja['year'] <= 2011].copy()
        decade2 = data_jja[data_jja['year'] >= 2012].copy()
        
        decade1_mean = decade1.groupby('year').agg({
            'bc_aod_regional': 'mean',
            'temperature_c': 'mean',
            'total_precip_mm': 'sum',
            'snowfall_mm': 'sum',
            'rainfall_mm': 'sum',
            'albedo_mean': 'mean'
        }).reset_index()
        decade1_mean['decade'] = '2002-2011'
        
        decade2_mean = decade2.groupby('year').agg({
            'bc_aod_regional': 'mean',
            'temperature_c': 'mean',
            'total_precip_mm': 'sum',  
            'snowfall_mm': 'sum',
            'rainfall_mm': 'sum',
            'albedo_mean': 'mean'
        }).reset_index()
        decade2_mean['decade'] = '2012-2024'
        
        datasets['decade1'] = decade1_mean
        datasets['decade2'] = decade2_mean
        
        logger.info(f"Decade 1 (2002-2011): {len(decade1_mean)} years")
        logger.info(f"Decade 2 (2012-2024): {len(decade2_mean)} years")
        
        return datasets
    
    def calculate_positive_degree_days(self, data: pd.DataFrame, windows: List[int] = [3, 5, 7]) -> pd.DataFrame:
        """
        Calculate Positive Degree Days (PDD) over various time windows.
        PDD = cumulative temperature > 0°C over specified days.
        """
        logger.info(f"Calculating Positive Degree Days for windows: {windows}")
        
        data_pdd = data.copy()
        data_pdd = data_pdd.sort_values('date').reset_index(drop=True)
        
        for window in windows:
            pdd_values = []
            
            for i in range(len(data_pdd)):
                start_idx = max(0, i - window + 1)
                temp_window = data_pdd.iloc[start_idx:i+1]['temperature_c']
                pdd = np.sum(np.maximum(temp_window, 0))
                pdd_values.append(pdd)
            
            data_pdd[f'pdd_{window}day'] = pdd_values
        
        logger.info(f"Added PDD columns for {len(windows)} time windows")
        return data_pdd
    
    def calculate_days_since_snowfall(self, data: pd.DataFrame, lookback_days: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        Calculate days since last significant snowfall for fresh snow indicators.
        """
        logger.info(f"Calculating days since snowfall for lookback periods: {lookback_days}")
        
        data_snow = data.copy()
        data_snow = data_snow.sort_values('date').reset_index(drop=True)
        
        for lookback in lookback_days:
            days_since_snow = []
            
            for i in range(len(data_snow)):
                # Look back up to lookback days
                days_since = lookback + 1  # Default: more than lookback days
                
                for j in range(1, min(lookback + 1, i + 1)):
                    if data_snow.iloc[i-j]['snowfall_mm'] >= self.FRESH_SNOW_THRESHOLD:
                        days_since = j
                        break
                
                days_since_snow.append(days_since)
            
            data_snow[f'days_since_snow_{lookback}d'] = days_since_snow
            
            # Create binary fresh snow indicator
            data_snow[f'fresh_snow_{lookback}d'] = (np.array(days_since_snow) <= lookback).astype(int)
        
        logger.info(f"Added fresh snow indicators for {len(lookback_days)} lookback periods")
        return data_snow
    
    def calculate_temperature_smoothing(self, data: pd.DataFrame, windows: List[int] = [3, 5, 7]) -> pd.DataFrame:
        """
        Calculate rolling mean temperatures to reduce weather noise.
        """
        logger.info(f"Calculating temperature smoothing for windows: {windows}")
        
        data_smooth = data.copy()
        data_smooth = data_smooth.sort_values('date').reset_index(drop=True)
        
        for window in windows:
            data_smooth[f'temp_smooth_{window}d'] = data_smooth['temperature_c'].rolling(
                window=window, center=True, min_periods=1
            ).mean()
        
        logger.info(f"Added smoothed temperature columns for {len(windows)} time windows")
        return data_smooth
    
    def calculate_seasonal_alpha_min(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate minimum albedo per season as proxy for impurity accumulation.
        """
        logger.info("Calculating seasonal minimum albedo...")
        
        data_season = data.copy()
        data_season['year'] = data_season['date'].dt.year
        data_season['season'] = 'summer'  # June-September data
        
        # Calculate minimum albedo per year
        yearly_alpha_min = data_season.groupby('year')['albedo_mean'].min().reset_index()
        yearly_alpha_min.rename(columns={'albedo_mean': 'alpha_min_seasonal'}, inplace=True)
        
        # Merge back to main dataset
        data_season = data_season.merge(yearly_alpha_min, on='year', how='left')
        
        logger.info(f"Added seasonal minimum albedo for {len(yearly_alpha_min)} years")
        return data_season
    
    def fit_hinge_temperature_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Fit hinge model with separate effects for T > 0°C and T ≤ 0°C.
        Model: Albedo = β₀ + β₁·T_pos + β₂·T_neg + other_predictors
        """
        logger.info("Fitting hinge temperature model...")
        
        clean_data = data.dropna(subset=['bc_aod_regional', 'temperature_c', 'snowfall_mm', 'rainfall_mm', 'albedo_mean'])
        n = len(clean_data)
        
        # Create temperature components
        temp_positive = np.maximum(clean_data['temperature_c'], 0)
        temp_negative = np.minimum(clean_data['temperature_c'], 0)
        
        # Design matrix
        X = np.column_stack([
            np.ones(n),  # Intercept
            clean_data['bc_aod_regional'],
            temp_positive,   # Positive temperature effect
            temp_negative,   # Negative temperature effect  
            clean_data['snowfall_mm'],
            clean_data['rainfall_mm']
        ])
        y = clean_data['albedo_mean'].values
        
        # Fit model
        try:
            beta, residuals_lstsq, rank, s = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ beta
            
            # Calculate statistics
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            k = X.shape[1] - 1
            adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
            
            # Calculate Delta_alpha/Delta_T for positive temperatures
            temp_coefficient_positive = beta[2]
            
            results = {
                'data': clean_data,
                'n_observations': n,
                'r_squared': r_squared,
                'adjusted_r_squared': adjusted_r_squared,
                'coefficients': {
                    'intercept': beta[0],
                    'bc_aod': beta[1],
                    'temp_positive': beta[2],
                    'temp_negative': beta[3],
                    'snowfall': beta[4],
                    'rainfall': beta[5]
                },
                'predictions': y_pred,
                'residuals': y - y_pred,
                'temp_positive': temp_positive,
                'temp_negative': temp_negative,
                'dalpha_dt_positive': temp_coefficient_positive
            }
            
            logger.info(f"Hinge model: R² = {r_squared:.4f}, Δα/ΔT(+) = {temp_coefficient_positive:.6f}")
            return results
            
        except Exception as e:
            logger.error(f"Error in hinge model fitting: {e}")
            return {}
    
    def fit_pdd_models(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Fit models using Positive Degree Days with different time windows.
        """
        logger.info("Fitting Positive Degree Day models...")
        
        # Add PDD calculations
        data_pdd = self.calculate_positive_degree_days(data)
        
        pdd_results = {}
        pdd_columns = [col for col in data_pdd.columns if col.startswith('pdd_')]
        
        for pdd_col in pdd_columns:
            logger.info(f"Fitting model with {pdd_col}...")
            
            clean_data = data_pdd.dropna(subset=['bc_aod_regional', pdd_col, 'snowfall_mm', 'rainfall_mm', 'albedo_mean'])
            n = len(clean_data)
            
            # Design matrix with PDD instead of raw temperature
            X = np.column_stack([
                np.ones(n),
                clean_data['bc_aod_regional'],
                clean_data[pdd_col],
                clean_data['snowfall_mm'],
                clean_data['rainfall_mm']
            ])
            y = clean_data['albedo_mean'].values
            
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ beta
                
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                k = X.shape[1] - 1
                adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - k - 1)
                
                pdd_results[pdd_col] = {
                    'data': clean_data,
                    'r_squared': r_squared,
                    'adjusted_r_squared': adjusted_r_squared,
                    'pdd_coefficient': beta[2],
                    'predictions': y_pred,
                    'residuals': y - y_pred
                }
                
                logger.info(f"{pdd_col}: R² = {r_squared:.4f}, coef = {beta[2]:.6f}")
                
            except Exception as e:
                logger.warning(f"Error fitting {pdd_col} model: {e}")
                continue
        
        return pdd_results
    
    def analyze_fresh_snow_interactions(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze temperature × fresh snow interactions.
        Hypothesis: Temperature effects stronger without recent snow.
        """
        logger.info("Analyzing fresh snow interactions...")
        
        # Add fresh snow indicators
        data_snow = self.calculate_days_since_snowfall(data)
        
        interaction_results = {}
        fresh_snow_columns = [col for col in data_snow.columns if col.startswith('fresh_snow_')]
        
        for fresh_col in fresh_snow_columns:
            logger.info(f"Analyzing interaction with {fresh_col}...")
            
            clean_data = data_snow.dropna(subset=['bc_aod_regional', 'temperature_c', 'snowfall_mm', 
                                                 'rainfall_mm', 'albedo_mean', fresh_col])
            n = len(clean_data)
            
            # Create interaction term
            temp_fresh_interaction = clean_data['temperature_c'] * clean_data[fresh_col]
            
            # Design matrix with interaction
            X = np.column_stack([
                np.ones(n),
                clean_data['bc_aod_regional'],
                clean_data['temperature_c'],
                clean_data[fresh_col],
                temp_fresh_interaction,
                clean_data['snowfall_mm'],
                clean_data['rainfall_mm']
            ])
            y = clean_data['albedo_mean'].values
            
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ beta
                
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Separate effects: with vs without fresh snow
                temp_effect_no_fresh = beta[2]  # Temperature main effect (when fresh_snow = 0)
                temp_effect_with_fresh = beta[2] + beta[4]  # Temperature + interaction (when fresh_snow = 1)
                
                interaction_results[fresh_col] = {
                    'data': clean_data,
                    'r_squared': r_squared,
                    'temp_effect_no_fresh': temp_effect_no_fresh,
                    'temp_effect_with_fresh': temp_effect_with_fresh,
                    'interaction_coefficient': beta[4],
                    'predictions': y_pred,
                    'fresh_snow_indicator': clean_data[fresh_col]
                }
                
                logger.info(f"{fresh_col}: R² = {r_squared:.4f}, T|no_snow = {temp_effect_no_fresh:.6f}, T|fresh_snow = {temp_effect_with_fresh:.6f}")
                
            except Exception as e:
                logger.warning(f"Error in {fresh_col} interaction model: {e}")
                continue
        
        return interaction_results
    
    def compare_scales_and_validate_benchmark(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Compare model performance across scales and validate against benchmark.
        """
        logger.info("Comparing scales and validating against benchmark...")
        
        scale_results = {}
        
        for scale_name, data in datasets.items():
            logger.info(f"Analyzing {scale_name} scale...")
            
            if len(data) < 10:  # Skip if insufficient data
                logger.warning(f"Insufficient data for {scale_name} scale ({len(data)} observations)")
                continue
            
            # Fit standard model
            try:
                results = self.fit_multiple_regression(data)
                
                # Extract temperature coefficient (Δα/ΔT)
                dalpha_dt = results['coefficients']['temperature']
                
                # Calculate benchmark deviation
                benchmark_deviation = dalpha_dt - self.BENCHMARK_DALPHA_DT
                benchmark_ratio = dalpha_dt / self.BENCHMARK_DALPHA_DT if self.BENCHMARK_DALPHA_DT != 0 else np.nan
                
                scale_results[scale_name] = {
                    'n_observations': results['n_observations'],
                    'r_squared': results['multiple_r_squared'],
                    'adjusted_r_squared': results['adjusted_r_squared'],
                    'dalpha_dt': dalpha_dt,
                    'benchmark_deviation': benchmark_deviation,
                    'benchmark_ratio': benchmark_ratio,
                    'data': results['data'],
                    'predictions': results['predictions']
                }
                
                logger.info(f"{scale_name}: n={results['n_observations']}, R²={results['multiple_r_squared']:.4f}, Δα/ΔT={dalpha_dt:.6f}")
                
            except Exception as e:
                logger.warning(f"Error analyzing {scale_name} scale: {e}")
                continue
        
        return scale_results
    
    def create_multi_scale_comparison_plot(self, scale_results: Dict[str, Any]) -> None:
        """
        Create visualization comparing R² and Δα/ΔT across temporal scales.
        """
        logger.info("Creating multi-scale comparison plot...")
        
        if not scale_results:
            logger.warning("No scale results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multi-Scale Temporal Analysis Comparison\nHaig Glacier Albedo Analysis (2002-2024)', 
                    fontsize=16, fontweight='bold')
        
        scales = list(scale_results.keys())
        r_squared_values = [scale_results[scale]['r_squared'] for scale in scales]
        adj_r_squared_values = [scale_results[scale]['adjusted_r_squared'] for scale in scales]
        dalpha_dt_values = [scale_results[scale]['dalpha_dt'] for scale in scales]
        n_obs_values = [scale_results[scale]['n_observations'] for scale in scales]
        
        # Plot 1: R² comparison
        ax1 = axes[0, 0]
        bars1 = ax1.bar(scales, r_squared_values, alpha=0.7, color='skyblue', edgecolor='black')
        bars2 = ax1.bar(scales, adj_r_squared_values, alpha=0.7, color='lightcoral', edgecolor='black')
        ax1.set_ylabel('R²', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance Across Scales', fontsize=14, fontweight='bold')
        ax1.legend(['R²', 'Adjusted R²'])
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, r_squared_values):
            ax1.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: Δα/ΔT comparison with benchmark
        ax2 = axes[0, 1]
        colors = ['darkgreen' if abs(val - self.BENCHMARK_DALPHA_DT) < abs(self.BENCHMARK_DALPHA_DT * 0.5) else 'darkred' 
                 for val in dalpha_dt_values]
        bars = ax2.bar(scales, dalpha_dt_values, alpha=0.7, color=colors, edgecolor='black')
        ax2.axhline(y=self.BENCHMARK_DALPHA_DT, color='red', linestyle='--', linewidth=2, 
                   label=f'Benchmark = {self.BENCHMARK_DALPHA_DT:.3f}')
        ax2.set_ylabel('Δα/ΔT (per +1°C)', fontsize=12, fontweight='bold')
        ax2.set_title('Temperature Sensitivity vs Benchmark', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, dalpha_dt_values):
            ax2.text(bar.get_x() + bar.get_width()/2, val - 0.002, f'{val:.4f}', 
                    ha='center', va='top', fontweight='bold', color='white')
        
        # Plot 3: Sample size comparison
        ax3 = axes[1, 0]
        ax3.bar(scales, n_obs_values, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.set_ylabel('Number of Observations', fontsize=12, fontweight='bold')
        ax3.set_title('Sample Size by Scale', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels
        for i, (scale, n) in enumerate(zip(scales, n_obs_values)):
            ax3.text(i, n + max(n_obs_values) * 0.02, str(n), ha='center', va='bottom', fontweight='bold')
        
        # Plot 4: Benchmark ratio (how close to benchmark)
        ax4 = axes[1, 1]
        benchmark_ratios = [scale_results[scale]['benchmark_ratio'] for scale in scales]
        colors4 = ['darkgreen' if 0.5 <= ratio <= 1.5 else 'orange' if 0.2 <= ratio <= 2.0 else 'darkred' 
                  for ratio in benchmark_ratios]
        bars4 = ax4.bar(scales, benchmark_ratios, alpha=0.7, color=colors4, edgecolor='black')
        ax4.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Perfect Match (ratio = 1.0)')
        ax4.set_ylabel('Δα/ΔT Ratio to Benchmark', fontsize=12, fontweight='bold')
        ax4.set_title('Benchmark Alignment', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, ratio in zip(bars4, benchmark_ratios):
            ax4.text(bar.get_x() + bar.get_width()/2, ratio + 0.05, f'{ratio:.2f}', 
                    ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "multi_scale_analysis" / "scale_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Multi-scale comparison plot saved")
    
    def create_hinge_model_visualization(self, hinge_results: Dict[str, Any]) -> None:
        """
        Create visualization of hinge model showing different temperature effects above/below 0°C.
        """
        logger.info("Creating hinge model visualization...")
        
        if not hinge_results:
            logger.warning("No hinge results to plot")
            return
        
        data = hinge_results['data']
        temp_pos = hinge_results['temp_positive']
        temp_neg = hinge_results['temp_negative']
        coefs = hinge_results['coefficients']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Hinge Temperature Model: Separate Effects Above/Below 0°C\nHaig Glacier Albedo Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Temperature vs Albedo with hinge visualization
        ax1 = axes[0, 0]
        
        # Separate data by temperature regime
        cold_mask = data['temperature_c'] <= 0
        warm_mask = data['temperature_c'] > 0
        
        # Plot data points
        ax1.scatter(data[cold_mask]['temperature_c'], data[cold_mask]['albedo_mean'], 
                   c='blue', alpha=0.6, label=f'T ≤ 0°C (n={np.sum(cold_mask)})', s=50)
        ax1.scatter(data[warm_mask]['temperature_c'], data[warm_mask]['albedo_mean'], 
                   c='red', alpha=0.6, label=f'T > 0°C (n={np.sum(warm_mask)})', s=50)
        
        # Create hinge function for visualization
        temp_range = np.linspace(data['temperature_c'].min(), data['temperature_c'].max(), 200)
        temp_pos_range = np.maximum(temp_range, 0)
        temp_neg_range = np.minimum(temp_range, 0)
        
        # Predicted albedo using average values for other predictors
        bc_mean = data['bc_aod_regional'].mean()
        snow_mean = data['snowfall_mm'].mean()
        rain_mean = data['rainfall_mm'].mean()
        
        albedo_pred = (coefs['intercept'] + 
                      coefs['bc_aod'] * bc_mean +
                      coefs['temp_positive'] * temp_pos_range +
                      coefs['temp_negative'] * temp_neg_range +
                      coefs['snowfall'] * snow_mean +
                      coefs['rainfall'] * rain_mean)
        
        ax1.plot(temp_range, albedo_pred, 'black', linewidth=3, alpha=0.8, label='Hinge Model')
        ax1.axvline(x=0, color='gray', linestyle='--', alpha=0.5, label='0°C Threshold')
        
        ax1.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Albedo', fontsize=12, fontweight='bold')
        ax1.set_title('Hinge Model: Temperature-Albedo Relationship', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Coefficient comparison
        ax2 = axes[0, 1]
        coef_names = ['T > 0°C', 'T ≤ 0°C']
        coef_values = [coefs['temp_positive'], coefs['temp_negative']]
        colors = ['red', 'blue']
        
        bars = ax2.bar(coef_names, coef_values, alpha=0.7, color=colors, edgecolor='black')
        ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax2.set_ylabel('Temperature Coefficient', fontsize=12, fontweight='bold')
        ax2.set_title('Temperature Effects by Regime', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels and benchmark comparison
        for bar, val, regime in zip(bars, coef_values, coef_names):
            ax2.text(bar.get_x() + bar.get_width()/2, val + (0.001 if val >= 0 else -0.001), 
                    f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold')
        
        # Add benchmark line for positive temperatures
        ax2.axhline(y=self.BENCHMARK_DALPHA_DT, color='red', linestyle='--', linewidth=2, alpha=0.7, 
                   label=f'Benchmark = {self.BENCHMARK_DALPHA_DT:.3f}')
        ax2.legend()
        
        # Plot 3: Residuals analysis by temperature regime
        ax3 = axes[1, 0]
        residuals = hinge_results['residuals']
        
        ax3.scatter(data[cold_mask]['temperature_c'], residuals[cold_mask], 
                   c='blue', alpha=0.6, label='T ≤ 0°C', s=40)
        ax3.scatter(data[warm_mask]['temperature_c'], residuals[warm_mask], 
                   c='red', alpha=0.6, label='T > 0°C', s=40)
        ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.8)
        ax3.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        ax3.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Residuals', fontsize=12, fontweight='bold')
        ax3.set_title('Residuals by Temperature Regime', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Model performance metrics
        ax4 = axes[1, 1]
        metrics = ['R²', 'Δα/ΔT(+)', 'Δα/ΔT(−)']
        values = [hinge_results['r_squared'], coefs['temp_positive'], coefs['temp_negative']]
        
        # Normalize for visualization (different scales)
        normalized_values = [hinge_results['r_squared'], 
                           coefs['temp_positive'] / self.BENCHMARK_DALPHA_DT,
                           coefs['temp_negative'] / self.BENCHMARK_DALPHA_DT]
        
        bars = ax4.bar(metrics, normalized_values, alpha=0.7, 
                      color=['green', 'red', 'blue'], edgecolor='black')
        ax4.set_ylabel('Normalized Values', fontsize=12, fontweight='bold')
        ax4.set_title('Hinge Model Performance', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add actual values as text
        for bar, actual, normalized in zip(bars, values, normalized_values):
            ax4.text(bar.get_x() + bar.get_width()/2, normalized + 0.05, 
                    f'{actual:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "nonlinear_models" / "hinge_model_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Hinge model visualization saved")
    
    def create_pdd_analysis_plots(self, pdd_results: Dict[str, Dict[str, Any]]) -> None:
        """
        Create visualization of Positive Degree Day analysis with different time windows.
        """
        logger.info("Creating PDD analysis plots...")
        
        if not pdd_results:
            logger.warning("No PDD results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Positive Degree Days (PDD) Analysis\nHaig Glacier Albedo Response to Cumulative Heat', 
                    fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        pdd_windows = list(pdd_results.keys())
        r_squared_values = [pdd_results[window]['r_squared'] for window in pdd_windows]
        pdd_coefficients = [pdd_results[window]['pdd_coefficient'] for window in pdd_windows]
        window_numbers = [int(window.split('_')[1].replace('day', '')) for window in pdd_windows]
        
        # Plot 1: R² vs PDD window size
        ax1 = axes[0, 0]
        ax1.plot(window_numbers, r_squared_values, 'o-', linewidth=2, markersize=8, color='darkblue')
        ax1.set_xlabel('PDD Window Size (days)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('R²', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance vs PDD Window', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(window_numbers, r_squared_values):
            ax1.text(x, y + 0.005, f'{y:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 2: PDD coefficients vs window size
        ax2 = axes[0, 1]
        ax2.plot(window_numbers, pdd_coefficients, 's-', linewidth=2, markersize=8, color='darkred')
        ax2.set_xlabel('PDD Window Size (days)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('PDD Coefficient', fontsize=12, fontweight='bold')
        ax2.set_title('Temperature Sensitivity vs PDD Window', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(window_numbers, pdd_coefficients):
            ax2.text(x, y + max(pdd_coefficients) * 0.02, f'{y:.5f}', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: PDD vs Albedo scatter (best performing window)
        best_window = max(pdd_results.keys(), key=lambda x: pdd_results[x]['r_squared'])
        best_data = pdd_results[best_window]['data']
        best_predictions = pdd_results[best_window]['predictions']
        
        ax3 = axes[1, 0]
        pdd_col = best_window
        scatter = ax3.scatter(best_data[pdd_col], best_data['albedo_mean'], 
                             c=best_data['temperature_c'], s=50, alpha=0.7, 
                             cmap='coolwarm', edgecolors='black', linewidth=0.3)
        
        # Add regression line
        z = np.polyfit(best_data[pdd_col], best_data['albedo_mean'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(best_data[pdd_col].min(), best_data[pdd_col].max(), 100)
        ax3.plot(x_line, p(x_line), 'darkblue', linewidth=2, alpha=0.8)
        
        window_days = int(best_window.split('_')[1].replace('day', ''))
        ax3.set_xlabel(f'{window_days}-Day PDD (°C·days)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Albedo', fontsize=12, fontweight='bold')
        ax3.set_title(f'Best PDD Model: {window_days}-Day Window\nR² = {pdd_results[best_window]["r_squared"]:.4f}', 
                     fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax3, shrink=0.8)
        cbar.set_label('Temperature (°C)', fontsize=10)
        
        # Plot 4: Model comparison (PDD vs standard temperature)
        ax4 = axes[1, 1]
        
        # Compare with baseline temperature model
        baseline_r2 = 0.142  # From your current results
        pdd_r2_values = list(r_squared_values)
        
        models = ['Baseline\n(Raw Temp)'] + [f'PDD\n({w}d)' for w in window_numbers]
        r2_comparison = [baseline_r2] + pdd_r2_values
        
        colors = ['gray'] + ['darkgreen' if r2 > baseline_r2 else 'darkred' for r2 in pdd_r2_values]
        bars = ax4.bar(models, r2_comparison, alpha=0.7, color=colors, edgecolor='black')
        
        ax4.set_ylabel('R²', fontsize=12, fontweight='bold')
        ax4.set_title('PDD Models vs Baseline', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        ax4.tick_params(axis='x', rotation=45)
        
        # Add value labels
        for bar, val in zip(bars, r2_comparison):
            ax4.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "nonlinear_models" / "pdd_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("PDD analysis plots saved")
    
    def create_fresh_snow_impact_plots(self, interaction_results: Dict[str, Any]) -> None:
        """
        Create visualization of fresh snow impacts on temperature-albedo relationships.
        """
        logger.info("Creating fresh snow impact plots...")
        
        if not interaction_results:
            logger.warning("No fresh snow interaction results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Fresh Snow Impact on Temperature-Albedo Relationship\nHaig Glacier Process Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Get the best interaction model (highest R²)
        best_model = max(interaction_results.keys(), key=lambda x: interaction_results[x]['r_squared'])
        best_data = interaction_results[best_model]['data']
        best_fresh_indicator = interaction_results[best_model]['fresh_snow_indicator']
        
        # Plot 1: Temperature effects with vs without fresh snow
        ax1 = axes[0, 0]
        lookback_days = [int(key.split('_')[-1].replace('d', '')) for key in interaction_results.keys()]
        temp_effects_no_fresh = [interaction_results[key]['temp_effect_no_fresh'] for key in interaction_results.keys()]
        temp_effects_with_fresh = [interaction_results[key]['temp_effect_with_fresh'] for key in interaction_results.keys()]
        
        x = np.arange(len(lookback_days))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, temp_effects_no_fresh, width, alpha=0.7, 
                       color='orange', edgecolor='black', label='No Fresh Snow')
        bars2 = ax1.bar(x + width/2, temp_effects_with_fresh, width, alpha=0.7, 
                       color='lightblue', edgecolor='black', label='With Fresh Snow')
        
        ax1.axhline(y=self.BENCHMARK_DALPHA_DT, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Benchmark = {self.BENCHMARK_DALPHA_DT:.3f}')
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        
        ax1.set_xlabel('Fresh Snow Lookback Period (days)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Temperature Effect (Δα/ΔT)', fontsize=12, fontweight='bold')
        ax1.set_title('Temperature Sensitivity: Fresh Snow vs No Fresh Snow', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(lookback_days)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels
        for bars, values in [(bars1, temp_effects_no_fresh), (bars2, temp_effects_with_fresh)]:
            for bar, val in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, val + 0.0005, f'{val:.4f}', 
                        ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 2: R² improvement with interactions
        ax2 = axes[0, 1]
        r_squared_values = [interaction_results[key]['r_squared'] for key in interaction_results.keys()]
        baseline_r2 = 0.142  # Your baseline R²
        
        bars = ax2.bar(lookback_days, r_squared_values, alpha=0.7, color='lightgreen', edgecolor='black')
        ax2.axhline(y=baseline_r2, color='red', linestyle='--', linewidth=2, alpha=0.7,
                   label=f'Baseline R² = {baseline_r2:.3f}')
        
        ax2.set_xlabel('Fresh Snow Lookback Period (days)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('R² with Interaction', fontsize=12, fontweight='bold')
        ax2.set_title('Model Performance with Fresh Snow Interactions', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, r_squared_values):
            improvement = val - baseline_r2
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005, 
                    f'{val:.3f}\n(+{improvement:.3f})', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 3: Scatter plot showing the interaction effect (best model)
        ax3 = axes[1, 0]
        
        # Separate data by fresh snow status
        fresh_mask = best_fresh_indicator == 1
        no_fresh_mask = best_fresh_indicator == 0
        
        ax3.scatter(best_data[no_fresh_mask]['temperature_c'], best_data[no_fresh_mask]['albedo_mean'], 
                   c='orange', alpha=0.7, s=50, label=f'No Fresh Snow (n={np.sum(no_fresh_mask)})', edgecolors='black', linewidth=0.3)
        ax3.scatter(best_data[fresh_mask]['temperature_c'], best_data[fresh_mask]['albedo_mean'], 
                   c='lightblue', alpha=0.7, s=50, label=f'Fresh Snow (n={np.sum(fresh_mask)})', edgecolors='black', linewidth=0.3)
        
        # Add regression lines for each group
        if np.sum(no_fresh_mask) > 1:
            z1 = np.polyfit(best_data[no_fresh_mask]['temperature_c'], best_data[no_fresh_mask]['albedo_mean'], 1)
            p1 = np.poly1d(z1)
            temp_range1 = np.linspace(best_data[no_fresh_mask]['temperature_c'].min(), 
                                     best_data[no_fresh_mask]['temperature_c'].max(), 100)
            ax3.plot(temp_range1, p1(temp_range1), 'darkorange', linewidth=2, alpha=0.8)
        
        if np.sum(fresh_mask) > 1:
            z2 = np.polyfit(best_data[fresh_mask]['temperature_c'], best_data[fresh_mask]['albedo_mean'], 1)
            p2 = np.poly1d(z2)
            temp_range2 = np.linspace(best_data[fresh_mask]['temperature_c'].min(), 
                                     best_data[fresh_mask]['temperature_c'].max(), 100)
            ax3.plot(temp_range2, p2(temp_range2), 'darkblue', linewidth=2, alpha=0.8)
        
        lookback = int(best_model.split('_')[-1].replace('d', ''))
        ax3.set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Albedo', fontsize=12, fontweight='bold')
        ax3.set_title(f'Temperature-Albedo by Fresh Snow Status\n({lookback}-day lookback)', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Interaction strength analysis
        ax4 = axes[1, 1]
        interaction_coeffs = [interaction_results[key]['interaction_coefficient'] for key in interaction_results.keys()]
        
        colors = ['darkgreen' if coef < 0 else 'darkred' for coef in interaction_coeffs]
        bars = ax4.bar(lookback_days, interaction_coeffs, alpha=0.7, color=colors, edgecolor='black')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.8)
        
        ax4.set_xlabel('Fresh Snow Lookback Period (days)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Interaction Coefficient', fontsize=12, fontweight='bold')
        ax4.set_title('Temperature × Fresh Snow Interaction Strength', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels and interpretation
        for bar, val in zip(bars, interaction_coeffs):
            ax4.text(bar.get_x() + bar.get_width()/2, val + (0.0005 if val >= 0 else -0.0005), 
                    f'{val:.4f}', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold', fontsize=9)
        
        # Add interpretation text
        ax4.text(0.02, 0.98, 'Negative: Temperature effect\nweaker with fresh snow\n\nPositive: Temperature effect\nstronger with fresh snow', 
                transform=ax4.transAxes, fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "fresh_snow_analysis" / "fresh_snow_impacts.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Fresh snow impact plots saved")
    
    def create_benchmark_validation_summary(self, scale_results: Dict[str, Any], 
                                          hinge_results: Dict[str, Any],
                                          pdd_results: Dict[str, Dict[str, Any]],
                                          interaction_results: Dict[str, Any]) -> None:
        """
        Create comprehensive benchmark validation summary comparing all approaches.
        """
        logger.info("Creating benchmark validation summary...")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Benchmark Validation Summary: Δα/ΔT ≈ -0.04 per +1°C\nComparison of Advanced Glaciological Methods', 
                    fontsize=16, fontweight='bold')
        
        # Collect all Δα/ΔT estimates
        dalpha_dt_estimates = []
        method_names = []
        r_squared_values = []
        colors = []
        
        # Scale results
        for scale, results in scale_results.items():
            dalpha_dt_estimates.append(results['dalpha_dt'])
            method_names.append(f'{scale.replace("_", " ").title()}')
            r_squared_values.append(results['r_squared'])
            # Color by closeness to benchmark
            ratio = abs(results['dalpha_dt'] / self.BENCHMARK_DALPHA_DT)
            if 0.5 <= ratio <= 1.5:
                colors.append('darkgreen')
            elif 0.2 <= ratio <= 2.0:
                colors.append('orange')
            else:
                colors.append('darkred')
        
        # Hinge model (positive temperature effect)
        if hinge_results:
            dalpha_dt_estimates.append(hinge_results['dalpha_dt_positive'])
            method_names.append('Hinge Model (T>0°C)')
            r_squared_values.append(hinge_results['r_squared'])
            ratio = abs(hinge_results['dalpha_dt_positive'] / self.BENCHMARK_DALPHA_DT)
            if 0.5 <= ratio <= 1.5:
                colors.append('darkgreen')
            elif 0.2 <= ratio <= 2.0:
                colors.append('orange')
            else:
                colors.append('darkred')
        
        # Best PDD model
        if pdd_results:
            best_pdd = max(pdd_results.keys(), key=lambda x: pdd_results[x]['r_squared'])
            # Convert PDD coefficient to temperature equivalent (approximate)
            pdd_coef = pdd_results[best_pdd]['pdd_coefficient']
            window_days = int(best_pdd.split('_')[1].replace('day', ''))
            # Rough approximation: PDD coefficient × typical temperature ≈ daily temperature effect
            temp_equivalent = pdd_coef * 5  # Assume 5°C average positive temperature
            
            dalpha_dt_estimates.append(temp_equivalent)
            method_names.append(f'PDD {window_days}d')
            r_squared_values.append(pdd_results[best_pdd]['r_squared'])
            ratio = abs(temp_equivalent / self.BENCHMARK_DALPHA_DT)
            if 0.5 <= ratio <= 1.5:
                colors.append('darkgreen')
            elif 0.2 <= ratio <= 2.0:
                colors.append('orange')
            else:
                colors.append('darkred')
        
        # Best interaction model (no fresh snow effect)
        if interaction_results:
            best_interaction = max(interaction_results.keys(), key=lambda x: interaction_results[x]['r_squared'])
            dalpha_dt_estimates.append(interaction_results[best_interaction]['temp_effect_no_fresh'])
            method_names.append('Interaction (No Fresh Snow)')
            r_squared_values.append(interaction_results[best_interaction]['r_squared'])
            ratio = abs(interaction_results[best_interaction]['temp_effect_no_fresh'] / self.BENCHMARK_DALPHA_DT)
            if 0.5 <= ratio <= 1.5:
                colors.append('darkgreen')
            elif 0.2 <= ratio <= 2.0:
                colors.append('orange')
            else:
                colors.append('darkred')
        
        # Plot 1: Δα/ΔT estimates vs benchmark
        ax1 = axes[0, 0]
        y_pos = np.arange(len(method_names))
        bars = ax1.barh(y_pos, dalpha_dt_estimates, alpha=0.7, color=colors, edgecolor='black')
        ax1.axvline(x=self.BENCHMARK_DALPHA_DT, color='red', linestyle='--', linewidth=3, alpha=0.8,
                   label=f'Benchmark = {self.BENCHMARK_DALPHA_DT:.3f}')
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(method_names)
        ax1.set_xlabel('Δα/ΔT (per +1°C)', fontsize=12, fontweight='bold')
        ax1.set_title('Temperature Sensitivity: Methods vs Benchmark', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar, val in zip(bars, dalpha_dt_estimates):
            ax1.text(val - 0.002 if val < 0 else val + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', ha='right' if val < 0 else 'left', va='center', fontweight='bold', fontsize=9)
        
        # Plot 2: R² performance comparison
        ax2 = axes[0, 1]
        bars2 = ax2.bar(range(len(method_names)), r_squared_values, alpha=0.7, color=colors, edgecolor='black')
        ax2.set_xticks(range(len(method_names)))
        ax2.set_xticklabels(method_names, rotation=45, ha='right')
        ax2.set_ylabel('R²', fontsize=12, fontweight='bold')
        ax2.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars2, r_squared_values):
            ax2.text(bar.get_x() + bar.get_width()/2, val + 0.005, f'{val:.3f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 3: Benchmark ratio analysis
        ax3 = axes[1, 0]
        benchmark_ratios = [val / self.BENCHMARK_DALPHA_DT for val in dalpha_dt_estimates]
        bars3 = ax3.bar(range(len(method_names)), benchmark_ratios, alpha=0.7, color=colors, edgecolor='black')
        ax3.axhline(y=1.0, color='red', linestyle='--', linewidth=3, alpha=0.8, label='Perfect Match (1.0)')
        ax3.axhline(y=0.5, color='orange', linestyle=':', linewidth=2, alpha=0.6, label='Acceptable Range')
        ax3.axhline(y=1.5, color='orange', linestyle=':', linewidth=2, alpha=0.6)
        
        ax3.set_xticks(range(len(method_names)))
        ax3.set_xticklabels(method_names, rotation=45, ha='right')
        ax3.set_ylabel('Ratio to Benchmark', fontsize=12, fontweight='bold')
        ax3.set_title('Benchmark Alignment Analysis', fontsize=14, fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, ratio in zip(bars3, benchmark_ratios):
            ax3.text(bar.get_x() + bar.get_width()/2, ratio + 0.05, f'{ratio:.2f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Plot 4: Method ranking summary
        ax4 = axes[1, 1]
        
        # Create ranking score: weighted combination of R² improvement and benchmark alignment
        baseline_r2 = 0.142
        r2_improvements = [(r2 - baseline_r2) / baseline_r2 for r2 in r_squared_values]
        benchmark_scores = [1 / (1 + abs(ratio - 1)) for ratio in benchmark_ratios]  # Higher is better
        
        # Combined score (equal weighting)
        combined_scores = [(r2_imp + bench_score) / 2 for r2_imp, bench_score in zip(r2_improvements, benchmark_scores)]
        
        # Sort by combined score
        sorted_indices = sorted(range(len(combined_scores)), key=lambda i: combined_scores[i], reverse=True)
        sorted_methods = [method_names[i] for i in sorted_indices]
        sorted_scores = [combined_scores[i] for i in sorted_indices]
        sorted_colors = [colors[i] for i in sorted_indices]
        
        bars4 = ax4.bar(range(len(sorted_methods)), sorted_scores, alpha=0.7, color=sorted_colors, edgecolor='black')
        ax4.set_xticks(range(len(sorted_methods)))
        ax4.set_xticklabels(sorted_methods, rotation=45, ha='right')
        ax4.set_ylabel('Combined Performance Score', fontsize=12, fontweight='bold')
        ax4.set_title('Method Ranking: R² + Benchmark Alignment', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add ranking numbers
        for i, (bar, score) in enumerate(zip(bars4, sorted_scores)):
            ax4.text(bar.get_x() + bar.get_width()/2, score + 0.02, f'#{i+1}\n{score:.2f}', 
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # Add color legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkgreen', alpha=0.7, label='Excellent (0.5-1.5× benchmark)'),
            Patch(facecolor='orange', alpha=0.7, label='Good (0.2-2.0× benchmark)'),
            Patch(facecolor='darkred', alpha=0.7, label='Poor (>2.0× or <0.2× benchmark)')
        ]
        ax4.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "benchmark_validation" / "validation_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Benchmark validation summary saved")
    
    def run_advanced_analysis(self) -> None:
        """
        Run the complete advanced albedo analysis with all sophisticated methods.
        """
        logger.info("Starting advanced albedo analysis...")
        
        # Load base data
        data = self.load_data()
        
        # Create multi-scale datasets
        datasets = self.create_multi_scale_temporal_datasets(data)
        
        # Compare scales and validate against benchmark
        scale_results = self.compare_scales_and_validate_benchmark(datasets)
        
        # Fit non-linear temperature models
        logger.info("Fitting non-linear temperature models...")
        hinge_results = self.fit_hinge_temperature_model(data)
        pdd_results = self.fit_pdd_models(data)
        
        # Analyze fresh snow interactions
        logger.info("Analyzing fresh snow interactions...")
        interaction_results = self.analyze_fresh_snow_interactions(data)
        
        # Create advanced visualizations
        logger.info("Creating advanced visualizations...")
        self.create_multi_scale_comparison_plot(scale_results)
        
        if hinge_results:
            self.create_hinge_model_visualization(hinge_results)
        
        if pdd_results:
            self.create_pdd_analysis_plots(pdd_results)
        
        if interaction_results:
            self.create_fresh_snow_impact_plots(interaction_results)
        
        # Create comprehensive benchmark validation
        self.create_benchmark_validation_summary(scale_results, hinge_results, pdd_results, interaction_results)
        
        # Run baseline analysis for comparison
        logger.info("Running baseline multiple regression analysis...")
        baseline_results = self.fit_multiple_regression(data)
        baseline_diagnostics = self.calculate_diagnostics(baseline_results)
        
        # Create essential baseline plots
        self.create_observed_vs_predicted_plot(baseline_results)
        self.create_predictor_importance_plot(baseline_results)
        
        # Save advanced results
        self.save_advanced_results(scale_results, hinge_results, pdd_results, interaction_results, baseline_results)
        
        # Print comprehensive summary
        self.print_advanced_summary(scale_results, hinge_results, pdd_results, interaction_results, baseline_results)
        
        logger.info(f"Advanced analysis complete! Results saved to: {self.output_dir}")
    
    def save_advanced_results(self, scale_results: Dict[str, Any], 
                            hinge_results: Dict[str, Any],
                            pdd_results: Dict[str, Dict[str, Any]],
                            interaction_results: Dict[str, Any],
                            baseline_results: Dict[str, Any]) -> None:
        """
        Save all advanced analysis results to CSV files.
        """
        logger.info("Saving advanced analysis results...")
        
        # Multi-scale comparison results
        scale_summary_data = []
        for scale, results in scale_results.items():
            scale_summary_data.append({
                'scale': scale,
                'n_observations': results['n_observations'],
                'r_squared': results['r_squared'],
                'adjusted_r_squared': results['adjusted_r_squared'],
                'dalpha_dt': results['dalpha_dt'],
                'benchmark_deviation': results['benchmark_deviation'],
                'benchmark_ratio': results['benchmark_ratio']
            })
        
        if scale_summary_data:
            scale_df = pd.DataFrame(scale_summary_data)
            scale_df.to_csv(self.output_dir / "multi_scale_analysis" / "scale_comparison_results.csv", index=False)
        
        # Hinge model results
        if hinge_results:
            hinge_summary = {
                'model': 'hinge_temperature',
                'n_observations': hinge_results['n_observations'],
                'r_squared': hinge_results['r_squared'],
                'adjusted_r_squared': hinge_results['adjusted_r_squared'],
                'temp_positive_coef': hinge_results['coefficients']['temp_positive'],
                'temp_negative_coef': hinge_results['coefficients']['temp_negative'],
                'dalpha_dt_positive': hinge_results['dalpha_dt_positive']
            }
            hinge_df = pd.DataFrame([hinge_summary])
            hinge_df.to_csv(self.output_dir / "nonlinear_models" / "hinge_model_results.csv", index=False)
        
        # PDD model results
        if pdd_results:
            pdd_summary_data = []
            for pdd_window, results in pdd_results.items():
                window_days = int(pdd_window.split('_')[1].replace('day', ''))
                pdd_summary_data.append({
                    'pdd_window': pdd_window,
                    'window_days': window_days,
                    'r_squared': results['r_squared'],
                    'adjusted_r_squared': results['adjusted_r_squared'],
                    'pdd_coefficient': results['pdd_coefficient']
                })
            
            pdd_df = pd.DataFrame(pdd_summary_data)
            pdd_df.to_csv(self.output_dir / "nonlinear_models" / "pdd_model_results.csv", index=False)
        
        # Fresh snow interaction results
        if interaction_results:
            interaction_summary_data = []
            for fresh_snow_model, results in interaction_results.items():
                lookback_days = int(fresh_snow_model.split('_')[-1].replace('d', ''))
                interaction_summary_data.append({
                    'fresh_snow_model': fresh_snow_model,
                    'lookback_days': lookback_days,
                    'r_squared': results['r_squared'],
                    'temp_effect_no_fresh': results['temp_effect_no_fresh'],
                    'temp_effect_with_fresh': results['temp_effect_with_fresh'],
                    'interaction_coefficient': results['interaction_coefficient']
                })
            
            interaction_df = pd.DataFrame(interaction_summary_data)
            interaction_df.to_csv(self.output_dir / "fresh_snow_analysis" / "interaction_results.csv", index=False)
        
        # Benchmark validation summary
        benchmark_summary = {
            'benchmark_dalpha_dt': self.BENCHMARK_DALPHA_DT,
            'baseline_dalpha_dt': baseline_results['coefficients']['temperature'],
            'baseline_r_squared': baseline_results['multiple_r_squared'],
            'baseline_benchmark_ratio': baseline_results['coefficients']['temperature'] / self.BENCHMARK_DALPHA_DT,
            'fresh_snow_threshold_mm': self.FRESH_SNOW_THRESHOLD
        }
        
        # Add best results from each method
        if scale_results:
            best_scale = max(scale_results.keys(), key=lambda x: scale_results[x]['r_squared'])
            benchmark_summary.update({
                'best_scale_method': best_scale,
                'best_scale_r_squared': scale_results[best_scale]['r_squared'],
                'best_scale_dalpha_dt': scale_results[best_scale]['dalpha_dt']
            })
        
        if hinge_results:
            benchmark_summary.update({
                'hinge_r_squared': hinge_results['r_squared'],
                'hinge_dalpha_dt_positive': hinge_results['dalpha_dt_positive']
            })
        
        if pdd_results:
            best_pdd = max(pdd_results.keys(), key=lambda x: pdd_results[x]['r_squared'])
            benchmark_summary.update({
                'best_pdd_method': best_pdd,
                'best_pdd_r_squared': pdd_results[best_pdd]['r_squared'],
                'best_pdd_coefficient': pdd_results[best_pdd]['pdd_coefficient']
            })
        
        if interaction_results:
            best_interaction = max(interaction_results.keys(), key=lambda x: interaction_results[x]['r_squared'])
            benchmark_summary.update({
                'best_interaction_method': best_interaction,
                'best_interaction_r_squared': interaction_results[best_interaction]['r_squared'],
                'best_interaction_temp_no_fresh': interaction_results[best_interaction]['temp_effect_no_fresh']
            })
        
        benchmark_df = pd.DataFrame([benchmark_summary])
        benchmark_df.to_csv(self.output_dir / "benchmark_validation" / "benchmark_summary.csv", index=False)
        
        logger.info("Advanced analysis results saved successfully")
    
    def print_advanced_summary(self, scale_results: Dict[str, Any], 
                             hinge_results: Dict[str, Any],
                             pdd_results: Dict[str, Dict[str, Any]],
                             interaction_results: Dict[str, Any],
                             baseline_results: Dict[str, Any]) -> None:
        """
        Print comprehensive advanced analysis summary.
        """
        print(f"\n{'='*100}")
        print(f"ADVANCED ALBEDO ANALYSIS RESULTS - GLACIOLOGICAL METHODS")
        print(f"{'='*100}")
        
        # Baseline comparison
        baseline_dalpha_dt = baseline_results['coefficients']['temperature']
        baseline_r2 = baseline_results['multiple_r_squared']
        print(f"\nBaseline Results (Daily Scale):")
        print(f"{'R²:':<30} {baseline_r2:.4f}")
        print(f"{'Delta-alpha/Delta-T:':<30} {baseline_dalpha_dt:.6f} per +1°C")
        print(f"{'Benchmark deviation:':<30} {baseline_dalpha_dt - self.BENCHMARK_DALPHA_DT:.6f}")
        print(f"{'Benchmark ratio:':<30} {baseline_dalpha_dt / self.BENCHMARK_DALPHA_DT:.3f}")
        
        # Multi-scale analysis
        if scale_results:
            print(f"\n{'Multi-Scale Temporal Analysis:':<30}")
            print(f"{'Scale':<15} {'n':<8} {'R²':<8} {'Δα/ΔT':<12} {'Benchmark Ratio':<15}")
            print(f"{'-'*60}")
            for scale, results in scale_results.items():
                print(f"{scale:<15} {results['n_observations']:<8} {results['r_squared']:<8.4f} "
                      f"{results['dalpha_dt']:<12.6f} {results['benchmark_ratio']:<15.3f}")
        
        # Non-linear temperature models
        print(f"\n{'Non-Linear Temperature Models:':<30}")
        
        if hinge_results:
            print(f"Hinge Model (T>0°C vs T≤0°C):")
            print(f"{'R²:':<30} {hinge_results['r_squared']:.4f}")
            print(f"{'Δα/ΔT (T>0°C):':<30} {hinge_results['coefficients']['temp_positive']:.6f}")
            print(f"{'Δα/ΔT (T≤0°C):':<30} {hinge_results['coefficients']['temp_negative']:.6f}")
            print(f"{'Benchmark ratio (T>0°C):':<30} {hinge_results['coefficients']['temp_positive'] / self.BENCHMARK_DALPHA_DT:.3f}")
        
        if pdd_results:
            print(f"\nPositive Degree Days Models:")
            print(f"{'Window':<12} {'R²':<8} {'PDD Coefficient':<15}")
            print(f"{'-'*35}")
            for window, results in pdd_results.items():
                window_days = int(window.split('_')[1].replace('day', ''))
                print(f"{window_days}d{'':<8} {results['r_squared']:<8.4f} {results['pdd_coefficient']:<15.6f}")
        
        # Fresh snow interactions
        if interaction_results:
            print(f"\n{'Fresh Snow Interaction Analysis:':<30}")
            print(f"{'Lookback':<10} {'R²':<8} {'T|No Snow':<12} {'T|Fresh Snow':<15} {'Interaction':<12}")
            print(f"{'-'*65}")
            for model, results in interaction_results.items():
                lookback = int(model.split('_')[-1].replace('d', ''))
                print(f"{lookback}d{'':<7} {results['r_squared']:<8.4f} "
                      f"{results['temp_effect_no_fresh']:<12.6f} {results['temp_effect_with_fresh']:<15.6f} "
                      f"{results['interaction_coefficient']:<12.6f}")
        
        # Best performing methods
        print(f"\n{'BEST PERFORMING METHODS vs BENCHMARK:':<50}")
        print(f"Benchmark: Δα/ΔT = {self.BENCHMARK_DALPHA_DT:.3f} per +1°C")
        print(f"{'-'*80}")
        
        best_methods = []
        
        # Find best from each category
        if scale_results:
            best_scale = max(scale_results.keys(), key=lambda x: scale_results[x]['r_squared'])
            best_methods.append((f'{best_scale.replace("_", " ").title()}', 
                               scale_results[best_scale]['r_squared'],
                               scale_results[best_scale]['dalpha_dt']))
        
        if hinge_results:
            best_methods.append(('Hinge Model (T>0°C)', 
                               hinge_results['r_squared'],
                               hinge_results['coefficients']['temp_positive']))
        
        if pdd_results:
            best_pdd = max(pdd_results.keys(), key=lambda x: pdd_results[x]['r_squared'])
            window_days = int(best_pdd.split('_')[1].replace('day', ''))
            # Approximate temperature equivalent
            temp_equiv = pdd_results[best_pdd]['pdd_coefficient'] * 5
            best_methods.append((f'PDD {window_days}d', 
                               pdd_results[best_pdd]['r_squared'],
                               temp_equiv))
        
        if interaction_results:
            best_interact = max(interaction_results.keys(), key=lambda x: interaction_results[x]['r_squared'])
            lookback = int(best_interact.split('_')[-1].replace('d', ''))
            best_methods.append((f'Interaction {lookback}d (No Fresh)', 
                               interaction_results[best_interact]['r_squared'],
                               interaction_results[best_interact]['temp_effect_no_fresh']))
        
        # Sort by R² and print
        best_methods.sort(key=lambda x: x[1], reverse=True)
        print(f"{'Method':<25} {'R²':<8} {'Δα/ΔT':<12} {'Benchmark Ratio':<15} {'Status':<10}")
        print(f"{'-'*80}")
        
        for method, r2, dalpha_dt in best_methods:
            ratio = dalpha_dt / self.BENCHMARK_DALPHA_DT
            if 0.5 <= ratio <= 1.5:
                status = "EXCELLENT"
            elif 0.2 <= ratio <= 2.0:
                status = "GOOD"
            else:
                status = "POOR"
            
            print(f"{method:<25} {r2:<8.4f} {dalpha_dt:<12.6f} {ratio:<15.3f} {status:<10}")
        
        # Key insights
        print(f"\n{'KEY INSIGHTS:':<30}")
        
        # R² improvements
        best_r2 = max([method[1] for method in best_methods]) if best_methods else baseline_r2
        r2_improvement = ((best_r2 - baseline_r2) / baseline_r2) * 100
        print(f"• Best R² improvement: {r2_improvement:.1f}% over baseline")
        
        # Benchmark alignment
        benchmark_aligned = [method for method in best_methods 
                           if 0.5 <= abs(method[2] / self.BENCHMARK_DALPHA_DT) <= 1.5]
        print(f"• Methods aligned with benchmark: {len(benchmark_aligned)}/{len(best_methods)}")
        
        # Scale effects
        if scale_results and 'jja_annual' in scale_results:
            jja_ratio = scale_results['jja_annual']['dalpha_dt'] / baseline_dalpha_dt
            print(f"• JJA annual vs daily scale effect: {jja_ratio:.2f}× stronger")
        
        # Process understanding
        if hinge_results:
            pos_temp_effect = hinge_results['coefficients']['temp_positive']
            neg_temp_effect = hinge_results['coefficients']['temp_negative']
            print(f"• Hinge model: T>0°C effect {abs(pos_temp_effect/neg_temp_effect):.1f}× stronger than T≤0°C")
        
        if interaction_results:
            best_interact = max(interaction_results.keys(), key=lambda x: interaction_results[x]['r_squared'])
            no_fresh_effect = interaction_results[best_interact]['temp_effect_no_fresh']
            fresh_effect = interaction_results[best_interact]['temp_effect_with_fresh']
            if abs(no_fresh_effect) > abs(fresh_effect):
                print(f"• Fresh snow reduces temperature sensitivity by {(1 - abs(fresh_effect/no_fresh_effect))*100:.0f}%")
            else:
                print(f"• Fresh snow increases temperature sensitivity by {(abs(fresh_effect/no_fresh_effect) - 1)*100:.0f}%")
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*100}")


def main():
    """Main function to run the advanced analysis."""
    analyzer = AdvancedAlbedoAnalyzer()
    analyzer.run_advanced_analysis()


if __name__ == "__main__":
    main()