#!/usr/bin/env python3
"""
Multiple Regression Analysis for BC AOD, Temperature, Precipitation vs Albedo

Specialized script for comprehensive multiple regression analysis of:
- BC AOD (Black Carbon Aerosol Optical Depth)
- Temperature
- Total Precipitation  
- Snowfall
vs Albedo

Author: Analysis System
Date: 2025-08-02
"""

# ============================================================================
# IMPORTS
# ============================================================================

import logging
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import warnings

from scipy import stats
from scipy.stats import pearsonr, spearmanr

from typing import Any, Dict, List, Optional, Tuple

# ============================================================================
# LOGGING SETUP
# ============================================================================

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'data_paths': {
        'merra2': r"D:\Downloads\Haig_Glacier_Climate_JuneSept_Daily_MERRA2_Speciated_AOD - Haig_Glacier_Climate_JuneSept_Daily_MERRA2_Speciated_AOD.csv",
        'modis': r"D:\Downloads\MODIS_Terra_Aqua_MultiProduct_2002-01-01_to_2025-01-01.csv"
    },
    'temporal_filters': {
        'years': {'min': 2002, 'max': 2024},
        'months': [6, 7, 8, 9]
    },
    'output_config': {
        'base_dir': "outputs",
        'prefix': "multiple_regression"
    },
    'visualization': {
        'figsize': (15, 10),
        'dpi': 300,
        'style': 'seaborn-v0_8'
    }
}

# ============================================================================
# DATA LOADING MODULE
# ============================================================================

class MultipleRegressionAnalyzer:
    """Comprehensive multiple regression analysis for albedo prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the multiple regression analyzer."""
        self.config = config
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(config['output_config']['base_dir']) / f"{config['output_config']['prefix']}_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        logger.info(f"Initialized Multiple Regression Analyzer. Output: {self.output_dir}")
    
    def load_data(self) -> pd.DataFrame:
        """Load and merge MERRA2 and MODIS data."""
        logger.info("Loading and merging datasets...")
        
        # Load MERRA2 data
        logger.info("Loading MERRA2 data...")
        merra2_df = pd.read_csv(self.config['data_paths']['merra2'])
        merra2_df['date'] = pd.to_datetime(merra2_df['date'])
        logger.info(f"Loaded {len(merra2_df)} MERRA2 records")
        
        # Load MODIS data
        logger.info("Loading MODIS data...")
        modis_df = pd.read_csv(self.config['data_paths']['modis'])
        modis_df['date'] = pd.to_datetime(modis_df['date'])
        logger.info(f"Loaded {len(modis_df)} MODIS records")
        
        # Filter MODIS data using config temporal filters
        months = self.config['temporal_filters']['months']
        year_min = self.config['temporal_filters']['years']['min']
        year_max = self.config['temporal_filters']['years']['max']
        logger.info(f"Filtering MODIS data to months {months} for years {year_min}-{year_max}...")
        modis_df = modis_df[
            (modis_df['date'].dt.month.isin(months)) &
            (modis_df['date'].dt.year >= year_min) &
            (modis_df['date'].dt.year <= year_max)
        ]
        logger.info(f"Filtered to {len(modis_df)} MODIS records")
        
        # Calculate daily mean albedo
        logger.info("Calculating daily mean albedo...")
        daily_albedo = modis_df.groupby('date').agg({
            'albedo': 'mean'
        }).reset_index()
        daily_albedo.rename(columns={'albedo': 'albedo_mean'}, inplace=True)
        logger.info(f"Created {len(daily_albedo)} daily mean albedo records")
        
        # Merge datasets
        logger.info("Merging datasets...")
        merged_df = pd.merge(
            merra2_df[['date', 'bc_aod_regional', 'temperature_c', 'total_precip_mm', 'snowfall_mm', 'rainfall_mm', 'year', 'month', 'day_of_year']],
            daily_albedo,
            on='date',
            how='inner'
        )
        
        logger.info(f"Found {len(merged_df)} complete paired observations")
        return merged_df
    
    # ============================================================================
    # STATISTICAL ANALYSIS METHODS
    # ============================================================================
    
    def fit_multiple_regression(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Fit 4-predictor multiple regression model and calculate comprehensive statistics."""
        logger.info("Fitting multiple regression model...")
        
        # Clean data (remove any remaining NaN values)
        clean_data = data.dropna(subset=['bc_aod_regional', 'temperature_c', 'snowfall_mm', 'rainfall_mm', 'albedo_mean'])
        n = len(clean_data)
        
        # Prepare design matrix: Albedo ~ BC AOD + Temperature + Snowfall + Rainfall
        X = np.column_stack([
            np.ones(n),  # Intercept
            clean_data['bc_aod_regional'], 
            clean_data['temperature_c'],
            clean_data['snowfall_mm'],
            clean_data['rainfall_mm']
        ])
        y = clean_data['albedo_mean'].values
        
        # Fit multiple regression using robust least squares
        try:
            # Check for zero variance in predictors
            predictor_vars = np.var(X[:, 1:], axis=0)
            if np.any(predictor_vars < 1e-10):
                logger.warning("Some predictors have near-zero variance")
            
            # Use least squares for numerical stability
            beta, residuals_lstsq, rank, s = np.linalg.lstsq(X, y, rcond=None)
            y_pred = X @ beta
            
            # Calculate model statistics
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            multiple_r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate adjusted R-squared
            k = X.shape[1] - 1  # Number of predictors (excluding intercept)
            adjusted_r_squared = 1 - (1 - multiple_r_squared) * (n - 1) / (n - k - 1)
            
            # Calculate F-statistic
            mse_reg = (ss_tot - ss_res) / k
            mse_res = ss_res / (n - k - 1)
            f_statistic = mse_reg / mse_res
            f_p_value = 1 - stats.f.cdf(f_statistic, k, n - k - 1)
            
            # Extract coefficients
            intercept = beta[0]
            bc_aod_coef = beta[1]
            temp_coef = beta[2]
            snowfall_coef = beta[3]
            rainfall_coef = beta[4]
            
            # Calculate standardized coefficients (beta coefficients)
            bc_aod_std = clean_data['bc_aod_regional'].std()
            temp_std = clean_data['temperature_c'].std()
            snowfall_std = clean_data['snowfall_mm'].std()
            rainfall_std = clean_data['rainfall_mm'].std()
            albedo_std = clean_data['albedo_mean'].std()
            
            bc_aod_beta = bc_aod_coef * (bc_aod_std / albedo_std)
            temp_beta = temp_coef * (temp_std / albedo_std)
            snowfall_beta = snowfall_coef * (snowfall_std / albedo_std)
            rainfall_beta = rainfall_coef * (rainfall_std / albedo_std)
            
            # Calculate residuals
            residuals = y - y_pred
            
            # Calculate standard errors using robust method
            mse = ss_res / (n - k - 1)
            try:
                cov_matrix = mse * np.linalg.pinv(X.T @ X)  # Use pseudo-inverse for stability
                std_errors = np.sqrt(np.diag(cov_matrix))
            except:
                std_errors = np.full(5, np.nan)
            
            # Calculate t-statistics and p-values
            t_stats = beta / std_errors
            t_p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n - k - 1))
            
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            logger.warning("Matrix is singular, using fallback values")
            multiple_r_squared = np.nan
            adjusted_r_squared = np.nan
            f_statistic = np.nan
            f_p_value = np.nan
            intercept = np.nan
            bc_aod_coef = np.nan
            temp_coef = np.nan
            snowfall_coef = np.nan
            rainfall_coef = np.nan
            bc_aod_beta = np.nan
            temp_beta = np.nan
            snowfall_beta = np.nan
            rainfall_beta = np.nan
            y_pred = np.full_like(y, np.mean(y))
            residuals = y - y_pred
            std_errors = np.full(5, np.nan)
            t_stats = np.full(5, np.nan)
            t_p_values = np.full(5, np.nan)
        
        # Calculate individual variable correlations with albedo
        bc_albedo_r, bc_albedo_p = pearsonr(clean_data['bc_aod_regional'], clean_data['albedo_mean'])
        temp_albedo_r, temp_albedo_p = pearsonr(clean_data['temperature_c'], clean_data['albedo_mean'])
        snow_albedo_r, snow_albedo_p = pearsonr(clean_data['snowfall_mm'], clean_data['albedo_mean'])
        rain_albedo_r, rain_albedo_p = pearsonr(clean_data['rainfall_mm'], clean_data['albedo_mean'])
        
        # Package results
        results = {
            'data': clean_data,
            'n_observations': n,
            'multiple_r_squared': multiple_r_squared,
            'adjusted_r_squared': adjusted_r_squared,
            'f_statistic': f_statistic,
            'f_p_value': f_p_value,
            'coefficients': {
                'intercept': intercept,
                'bc_aod': bc_aod_coef,
                'temperature': temp_coef,
                'snowfall': snowfall_coef,
                'rainfall': rainfall_coef
            },
            'standardized_coefficients': {
                'bc_aod': bc_aod_beta,
                'temperature': temp_beta,
                'snowfall': snowfall_beta,
                'rainfall': rainfall_beta
            },
            'standard_errors': std_errors,
            't_statistics': t_stats,
            'p_values': t_p_values,
            'predictions': y_pred,
            'residuals': residuals,
            'individual_correlations': {
                'bc_aod': {'r': bc_albedo_r, 'p': bc_albedo_p},
                'temperature': {'r': temp_albedo_r, 'p': temp_albedo_p},
                'snowfall': {'r': snow_albedo_r, 'p': snow_albedo_p},
                'rainfall': {'r': rain_albedo_r, 'p': rain_albedo_p}
            }
        }
        
        logger.info(f"Multiple regression fitted: R² = {multiple_r_squared:.4f}, Adj R² = {adjusted_r_squared:.4f}")
        logger.info(f"Model: Albedo ~ BC AOD + Temperature + Snowfall + Rainfall")
        return results
    
    def calculate_diagnostics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate regression diagnostics."""
        logger.info("Calculating regression diagnostics...")
        
        data = results['data']
        y = data['albedo_mean'].values
        y_pred = results['predictions']
        residuals = results['residuals']
        n = results['n_observations']
        
        # Calculate leverage values (hat matrix diagonal)
        X = np.column_stack([
            np.ones(n),
            data['bc_aod_regional'], 
            data['temperature_c'],
            data['snowfall_mm'],
            data['rainfall_mm']
        ])
        
        try:
            # Use pseudo-inverse for numerical stability
            H = X @ np.linalg.pinv(X.T @ X) @ X.T
            leverage = np.diag(H)
            
            # Calculate Cook's distance
            mse = np.sum(residuals**2) / (n - 5)  # 5 parameters (including intercept)
            cooks_d = (residuals**2 / (5 * mse)) * (leverage / (1 - leverage)**2)
            
            # Calculate studentized residuals
            residual_std = np.sqrt(mse * (1 - leverage))
            studentized_residuals = residuals / residual_std
            
        except np.linalg.LinAlgError:
            leverage = np.full(n, np.nan)
            cooks_d = np.full(n, np.nan)
            studentized_residuals = np.full(n, np.nan)
        
        # Test for normality of residuals (Shapiro-Wilk)
        if n <= 5000:  # Shapiro-Wilk has sample size limit
            shapiro_stat, shapiro_p = stats.shapiro(residuals)
        else:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        # Test for homoscedasticity (Breusch-Pagan test approximation)
        # Regress squared residuals on predicted values
        try:
            bp_slope, bp_intercept, bp_r, bp_p, bp_se = stats.linregress(y_pred, residuals**2)
            homoscedasticity_p = bp_p
        except:
            homoscedasticity_p = np.nan
        
        # Calculate VIF (Variance Inflation Factors) for multicollinearity
        vif_values = {}
        predictors = ['bc_aod_regional', 'temperature_c', 'snowfall_mm', 'rainfall_mm']
        
        for i, predictor in enumerate(predictors):
            try:
                # Regress predictor on other predictors
                other_predictors = [p for j, p in enumerate(predictors) if j != i]
                X_others = data[other_predictors].values
                X_others = np.column_stack([np.ones(len(X_others)), X_others])
                y_predictor = data[predictor].values
                
                beta_vif, _, _, _ = np.linalg.lstsq(X_others, y_predictor, rcond=None)
                y_pred_vif = X_others @ beta_vif
                
                ss_res_vif = np.sum((y_predictor - y_pred_vif) ** 2)
                ss_tot_vif = np.sum((y_predictor - np.mean(y_predictor)) ** 2)
                r_squared_vif = 1 - (ss_res_vif / ss_tot_vif)
                
                vif = 1 / (1 - r_squared_vif) if r_squared_vif < 0.999 else np.inf
                vif_values[predictor] = vif
                
            except:
                vif_values[predictor] = np.nan
        
        diagnostics = {
            'leverage': leverage,
            'cooks_distance': cooks_d,
            'studentized_residuals': studentized_residuals,
            'shapiro_wilk_stat': shapiro_stat,
            'shapiro_wilk_p': shapiro_p,
            'homoscedasticity_p': homoscedasticity_p,
            'vif_values': vif_values,
            'outliers_high_leverage': np.where(leverage > 2 * 5 / n)[0],  # 2*p/n threshold
            'outliers_high_cooks': np.where(cooks_d > 4 / n)[0],  # 4/n threshold
            'outliers_high_residuals': np.where(np.abs(studentized_residuals) > 3)[0]  # |t| > 3
        }
        
        logger.info("Regression diagnostics calculated")
        return diagnostics
    
    # ============================================================================
    # VISUALIZATION METHODS
    # ============================================================================
    
    def create_observed_vs_predicted_plot(self, results: Dict[str, Any]) -> None:
        """Create observed vs predicted scatter plot with performance metrics."""
        logger.info("Creating observed vs predicted plot...")
        
        data = results['data']
        y_obs = data['albedo_mean'].values
        y_pred = results['predictions']
        
        # Calculate performance metrics
        rmse = np.sqrt(np.mean((y_obs - y_pred) ** 2))
        mae = np.mean(np.abs(y_obs - y_pred))
        r_squared = results['multiple_r_squared']
        n = len(y_obs)
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create scatter plot colored by temperature
        scatter = ax.scatter(y_obs, y_pred, c=data['temperature_c'], s=60, alpha=0.7, 
                            cmap='coolwarm', edgecolors='black', linewidth=0.5)
        
        # Add 1:1 reference line
        min_val = min(y_obs.min(), y_pred.min())
        max_val = max(y_obs.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'darkred', linestyle='--', 
               linewidth=2, alpha=0.8, label='Perfect 1:1 Line')
        
        # Add best fit line
        z = np.polyfit(y_obs, y_pred, 1)
        p = np.poly1d(z)
        ax.plot(y_obs, p(y_obs), 'darkblue', linewidth=2, alpha=0.8, 
               label=f'Best Fit (slope={z[0]:.3f})')
        
        # Customize plot
        ax.set_xlabel('Observed Albedo (MODIS)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted Albedo (Model)', fontsize=14, fontweight='bold')
        ax.set_title('Model Performance: Observed vs Predicted Albedo\nHaig Glacier - MODIS Terra+Aqua (June-September 2002-2024)', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Air Temperature (°C)', fontsize=12, fontweight='bold')
        
        # Add performance metrics text box
        textbox_content = (
            f"Performance Metrics:\n"
            f"R² = {r_squared:.4f}\n"
            f"RMSE = {rmse:.4f}\n"
            f"MAE = {mae:.4f}\n"
            f"n = {n} observations"
        )
        
        ax.text(0.05, 0.95, textbox_content, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "observed_vs_predicted.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Observed vs predicted plot saved")
    
    def create_residuals_vs_fitted_plot(self, results: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
        """Create residuals vs fitted values plot for diagnostic purposes."""
        logger.info("Creating residuals vs fitted plot...")
        
        y_pred = results['predictions']
        residuals = results['residuals']
        data = results['data']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create scatter plot
        scatter = ax.scatter(y_pred, residuals, c=data['temperature_c'], s=50, alpha=0.7, 
                            cmap='coolwarm', edgecolors='black', linewidth=0.3)
        
        # Add reference line at y=0
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Zero Residual Line')
        
        # Add LOWESS smooth line for trend detection
        sorted_indices = np.argsort(y_pred)
        sorted_pred = y_pred[sorted_indices]
        sorted_resid = residuals[sorted_indices]
        
        # Simple moving average as LOWESS substitute
        window_size = max(5, len(y_pred) // 10)
        if len(y_pred) > window_size:
            smooth_pred = []
            smooth_resid = []
            for i in range(window_size//2, len(sorted_pred) - window_size//2):
                smooth_pred.append(sorted_pred[i])
                smooth_resid.append(np.mean(sorted_resid[i-window_size//2:i+window_size//2+1]))
            ax.plot(smooth_pred, smooth_resid, 'blue', linewidth=2, alpha=0.8, label='Trend Line')
        
        # Calculate homoscedasticity test
        homo_p = diagnostics.get('homoscedasticity_p', np.nan)
        
        # Customize plot
        ax.set_xlabel('Fitted Values (Predicted Albedo)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Residuals', fontsize=14, fontweight='bold')
        ax.set_title('Residuals vs Fitted Values\nDiagnostic for Homoscedasticity and Model Assumptions', 
                    fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)
        
        # Add colorbar
        cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('Air Temperature (°C)', fontsize=12, fontweight='bold')
        
        # Add diagnostic text box
        textbox_content = (
            f"Diagnostic Results:\n"
            f"Homoscedasticity p = {homo_p:.4f}\n"
            f"Residual Std = {np.std(residuals):.4f}\n"
            f"Mean Residual = {np.mean(residuals):.6f}"
        )
        
        ax.text(0.05, 0.95, textbox_content, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "residuals_vs_fitted.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Residuals vs fitted plot saved")
    
    def create_qq_plot(self, results: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
        """Create Q-Q plot for normality assessment of residuals."""
        logger.info("Creating Q-Q plot...")
        
        residuals = results['residuals']
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # Create Q-Q plot
        stats.probplot(residuals, dist="norm", plot=ax)
        
        # Customize the plot
        ax.set_title('Q-Q Plot: Normality Assessment of Residuals\nHaig Glacier - MODIS Terra+Aqua (June-September 2002-2024)', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Theoretical Quantiles (Standard Normal)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Sample Quantiles (Residuals)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add normality test results
        shapiro_stat = diagnostics.get('shapiro_wilk_stat', np.nan)
        shapiro_p = diagnostics.get('shapiro_wilk_p', np.nan)
        
        textbox_content = (
            f"Normality Tests:\n"
            f"Shapiro-Wilk Statistic = {shapiro_stat:.4f}\n"
            f"Shapiro-Wilk p-value = {shapiro_p:.4f}\n"
            f"\n"
            f"Interpretation:\n"
            f"p > 0.05: Residuals are normal\n"
            f"p ≤ 0.05: Residuals deviate from normal"
        )
        
        ax.text(0.05, 0.95, textbox_content, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "qq_plot_normality.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Q-Q plot saved")
    
    def create_partial_regression_plots(self, results: Dict[str, Any]) -> None:
        """Create 4-panel partial regression plots (added-variable plots)."""
        logger.info("Creating partial regression plots...")
        
        data = results['data']
        predictors = ['bc_aod_regional', 'temperature_c', 'snowfall_mm', 'rainfall_mm']
        predictor_labels = ['BC AOD (Regional)', 'Temperature (°C)', 'Snowfall (mm)', 'Rainfall (mm)']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        fig.suptitle('Partial Regression Plots (Added-Variable Plots)\nHaig Glacier - MODIS Terra+Aqua (June-September 2002-2024)', 
                    fontsize=16, fontweight='bold')
        
        y = data['albedo_mean'].values
        n = len(y)
        
        for i, (predictor, label) in enumerate(zip(predictors, predictor_labels)):
            ax = axes[i]
            
            try:
                # Get other predictors
                other_predictors = [p for p in predictors if p != predictor]
                
                # Regress y on other predictors
                X_others = np.column_stack([np.ones(n)] + [data[p].values for p in other_predictors])
                beta_y, _, _, _ = np.linalg.lstsq(X_others, y, rcond=None)
                y_residual = y - X_others @ beta_y
                
                # Regress target predictor on other predictors
                x_target = data[predictor].values
                beta_x, _, _, _ = np.linalg.lstsq(X_others, x_target, rcond=None)
                x_residual = x_target - X_others @ beta_x
                
                # Create partial regression plot
                scatter = ax.scatter(x_residual, y_residual, c=data['temperature_c'], 
                                   s=50, alpha=0.7, cmap='coolwarm', 
                                   edgecolors='black', linewidth=0.3)
                
                # Add regression line
                if len(x_residual) > 1 and np.std(x_residual) > 1e-10:
                    z = np.polyfit(x_residual, y_residual, 1)
                    p = np.poly1d(z)
                    x_line = np.linspace(x_residual.min(), x_residual.max(), 100)
                    ax.plot(x_line, p(x_line), 'red', linewidth=2, alpha=0.8)
                    
                    # Calculate partial correlation
                    r_partial = np.corrcoef(x_residual, y_residual)[0, 1]
                    ax.set_title(f'{label}\nPartial r = {r_partial:.3f}', fontsize=12, fontweight='bold')
                else:
                    ax.set_title(f'{label}\nInsufficient variation', fontsize=12, fontweight='bold')
                
                ax.set_xlabel(f'{label} | Others', fontsize=11)
                ax.set_ylabel('Albedo | Others', fontsize=11)
                ax.grid(True, alpha=0.3)
                
            except Exception as e:
                ax.text(0.5, 0.5, f'Error in partial regression\nfor {label}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{label}\nError in calculation', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "partial_regression_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Partial regression plots saved")
    
    def create_marginal_effects_plots(self, results: Dict[str, Any]) -> None:
        """Create marginal effects plots showing sensitivity to each predictor."""
        logger.info("Creating marginal effects plots...")
        
        data = results['data'] 
        coef = results['coefficients']
        predictors = ['bc_aod_regional', 'temperature_c', 'snowfall_mm', 'rainfall_mm']
        predictor_labels = ['BC AOD (Regional)', 'Temperature (°C)', 'Snowfall (mm)', 'Rainfall (mm)']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        fig.suptitle('Marginal Effects: Sensitivity Analysis\nHaig Glacier - MODIS Terra+Aqua (June-September 2002-2024)', 
                    fontsize=16, fontweight='bold')
        
        for i, (predictor, label) in enumerate(zip(predictors, predictor_labels)):
            ax = axes[i]
            
            # Create range for the focal predictor
            x_focal = data[predictor].values
            x_range = np.linspace(x_focal.min(), x_focal.max(), 100)
            
            # Set other predictors to their means
            other_means = {}
            for p in predictors:
                if p != predictor:
                    other_means[p] = data[p].mean()
            
            # Calculate marginal effect
            y_marginal = np.full_like(x_range, coef['intercept'])
            
            # Add focal predictor effect
            if predictor == 'bc_aod_regional':
                y_marginal += coef['bc_aod'] * x_range
            elif predictor == 'temperature_c':
                y_marginal += coef['temperature'] * x_range
            elif predictor == 'snowfall_mm':
                y_marginal += coef['snowfall'] * x_range
            elif predictor == 'rainfall_mm':
                y_marginal += coef['rainfall'] * x_range
            
            # Add other predictors at their means
            for p in predictors:
                if p != predictor:
                    if p == 'bc_aod_regional':
                        y_marginal += coef['bc_aod'] * other_means[p]
                    elif p == 'temperature_c':
                        y_marginal += coef['temperature'] * other_means[p]
                    elif p == 'snowfall_mm':
                        y_marginal += coef['snowfall'] * other_means[p]
                    elif p == 'rainfall_mm':
                        y_marginal += coef['rainfall'] * other_means[p]
            
            # Plot marginal effect
            ax.plot(x_range, y_marginal, 'darkblue', linewidth=3, alpha=0.8, label='Marginal Effect')
            
            # Add data points
            scatter = ax.scatter(x_focal, data['albedo_mean'], c=data['temperature_c'], 
                               s=30, alpha=0.6, cmap='coolwarm', 
                               edgecolors='black', linewidth=0.3, label='Observed Data')
            
            # Calculate coefficient for this predictor
            if predictor == 'bc_aod_regional':
                coef_val = coef['bc_aod']
            elif predictor == 'temperature_c':
                coef_val = coef['temperature']
            elif predictor == 'snowfall_mm':
                coef_val = coef['snowfall']
            elif predictor == 'rainfall_mm':
                coef_val = coef['rainfall']
            
            ax.set_xlabel(label, fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted Albedo', fontsize=12, fontweight='bold')
            ax.set_title(f'{label}\nMarginal Effect = {coef_val:.6f}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "marginal_effects_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Marginal effects plots saved")
    
    def create_predictor_importance_plot(self, results: Dict[str, Any]) -> None:
        """Create predictor importance plot using standardized coefficients."""
        logger.info("Creating predictor importance plot...")
        
        std_coef = results['standardized_coefficients']
        p_values = results['p_values'][1:]  # Exclude intercept
        
        # Prepare data
        predictors = ['BC AOD', 'Temperature', 'Snowfall', 'Rainfall']
        coefficients = [std_coef['bc_aod'], std_coef['temperature'], 
                       std_coef['snowfall'], std_coef['rainfall']]
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Predictor Importance Analysis\nHaig Glacier - MODIS Terra+Aqua (June-September 2002-2024)', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Standardized coefficients (absolute values for importance)
        abs_coef = np.abs(coefficients)
        colors = ['red' if p <= 0.05 else 'orange' if p <= 0.1 else 'gray' 
                 for p in p_values]
        
        bars1 = ax1.barh(predictors, abs_coef, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('|Standardized Coefficient (β)|', fontsize=12, fontweight='bold')
        ax1.set_title('Predictor Importance\n(Absolute Standardized Coefficients)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, coef, p_val) in enumerate(zip(bars1, coefficients, p_values)):
            width = bar.get_width()
            ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{coef:.3f}\n(p={p_val:.3f})', 
                    ha='left', va='center', fontweight='bold', fontsize=10)
        
        # Add legend for significance
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='red', alpha=0.7, label='p ≤ 0.05 (Significant)'),
                          Patch(facecolor='orange', alpha=0.7, label='0.05 < p ≤ 0.10 (Marginal)'),
                          Patch(facecolor='gray', alpha=0.7, label='p > 0.10 (Not Significant)')]
        ax1.legend(handles=legend_elements, loc='lower right')
        
        # Plot 2: Effect direction (signed coefficients)
        colors2 = ['darkgreen' if c > 0 else 'darkred' for c in coefficients]
        bars2 = ax2.barh(predictors, coefficients, color=colors2, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Standardized Coefficient (β)', fontsize=12, fontweight='bold')
        ax2.set_title('Effect Direction\n(Signed Standardized Coefficients)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.axvline(x=0, color='black', linestyle='-', alpha=0.5)
        
        # Add value labels
        for bar, coef in zip(bars2, coefficients):
            width = bar.get_width()
            label_x = width + 0.01 if width >= 0 else width - 0.01
            ha = 'left' if width >= 0 else 'right'
            ax2.text(label_x, bar.get_y() + bar.get_height()/2, 
                    f'{coef:.3f}', ha=ha, va='center', fontweight='bold', fontsize=10)
        
        # Add legend for direction
        legend_elements2 = [Patch(facecolor='darkgreen', alpha=0.7, label='Positive Effect'),
                           Patch(facecolor='darkred', alpha=0.7, label='Negative Effect')]
        ax2.legend(handles=legend_elements2, loc='lower right')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "predictor_importance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Predictor importance plot saved")
    
    def validate_improvements(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Compare old model (with total_precip) vs new model (with rainfall) performance."""
        logger.info("Validating model improvements...")
        
        # Clean data
        clean_data = data.dropna(subset=['bc_aod_regional', 'temperature_c', 'total_precip_mm', 
                                        'snowfall_mm', 'rainfall_mm', 'albedo_mean'])
        n = len(clean_data)
        y = clean_data['albedo_mean'].values
        
        # Old model: BC AOD + Temperature + Total Precip + Snowfall
        X_old = np.column_stack([
            np.ones(n),
            clean_data['bc_aod_regional'], 
            clean_data['temperature_c'],
            clean_data['total_precip_mm'],
            clean_data['snowfall_mm']
        ])
        
        # New model: BC AOD + Temperature + Snowfall + Rainfall
        X_new = np.column_stack([
            np.ones(n),
            clean_data['bc_aod_regional'], 
            clean_data['temperature_c'],
            clean_data['snowfall_mm'],
            clean_data['rainfall_mm']
        ])
        
        results_comparison = {}
        
        for model_name, X in [('Old Model (Total Precip)', X_old), ('New Model (Rainfall)', X_new)]:
            try:
                beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                y_pred = X @ beta
                
                # Calculate metrics
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                adj_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - X.shape[1])
                rmse = np.sqrt(np.mean((y - y_pred) ** 2))
                mae = np.mean(np.abs(y - y_pred))
                
                # AIC and BIC
                k = X.shape[1]
                mse = ss_res / (n - k)
                aic = n * np.log(mse) + 2 * k
                bic = n * np.log(mse) + k * np.log(n)
                
                results_comparison[model_name] = {
                    'r_squared': r_squared,
                    'adj_r_squared': adj_r_squared,
                    'rmse': rmse,
                    'mae': mae,
                    'aic': aic,
                    'bic': bic,
                    'predictions': y_pred
                }
                
            except Exception as e:
                logger.warning(f"Error fitting {model_name}: {e}")
                results_comparison[model_name] = {
                    'r_squared': np.nan, 'adj_r_squared': np.nan,
                    'rmse': np.nan, 'mae': np.nan, 'aic': np.nan, 'bic': np.nan,
                    'predictions': np.full_like(y, np.mean(y))
                }
        
        # Create comparison plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Model Comparison: Old vs New Predictor Structure\nHaig Glacier - MODIS Terra+Aqua (June-September 2002-2024)', 
                    fontsize=16, fontweight='bold')
        
        for i, (model_name, results) in enumerate(results_comparison.items()):
            ax = axes[i]
            y_pred = results['predictions']
            
            # Scatter plot
            scatter = ax.scatter(y, y_pred, c=clean_data['temperature_c'], s=50, alpha=0.7, 
                               cmap='coolwarm', edgecolors='black', linewidth=0.3)
            
            # 1:1 line
            min_val = min(y.min(), y_pred.min())
            max_val = max(y.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'red', linestyle='--', 
                   linewidth=2, alpha=0.8, label='1:1 Line')
            
            ax.set_xlabel('Observed Albedo', fontsize=12, fontweight='bold')
            ax.set_ylabel('Predicted Albedo', fontsize=12, fontweight='bold')
            ax.set_title(f'{model_name}\nR² = {results["r_squared"]:.4f}, RMSE = {results["rmse"]:.4f}', 
                        fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Add metrics text
            textbox_content = (
                f"Adj R² = {results['adj_r_squared']:.4f}\n"
                f"MAE = {results['mae']:.4f}\n"
                f"AIC = {results['aic']:.1f}\n"
                f"BIC = {results['bic']:.1f}"
            )
            ax.text(0.05, 0.95, textbox_content, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "model_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Model validation completed")
        return results_comparison
    
    def run_stability_test(self, data: pd.DataFrame, n_bootstrap: int = 100) -> Dict[str, Any]:
        """Run bootstrap stability test for model coefficients."""
        logger.info(f"Running bootstrap stability test with {n_bootstrap} iterations...")
        
        clean_data = data.dropna(subset=['bc_aod_regional', 'temperature_c', 
                                        'snowfall_mm', 'rainfall_mm', 'albedo_mean'])
        n = len(clean_data)
        
        # Storage for bootstrap results
        bootstrap_results = {
            'r_squared': [],
            'coefficients': {'intercept': [], 'bc_aod': [], 'temperature': [], 'snowfall': [], 'rainfall': []}
        }
        
        np.random.seed(42)  # For reproducibility
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            boot_data = clean_data.iloc[indices]
            
            try:
                # Fit model on bootstrap sample
                X_boot = np.column_stack([
                    np.ones(n),
                    boot_data['bc_aod_regional'], 
                    boot_data['temperature_c'],
                    boot_data['snowfall_mm'],
                    boot_data['rainfall_mm']
                ])
                y_boot = boot_data['albedo_mean'].values
                
                beta, _, _, _ = np.linalg.lstsq(X_boot, y_boot, rcond=None)
                y_pred = X_boot @ beta
                
                # Calculate R-squared
                ss_res = np.sum((y_boot - y_pred) ** 2)
                ss_tot = np.sum((y_boot - np.mean(y_boot)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                
                # Store results
                bootstrap_results['r_squared'].append(r_squared)
                bootstrap_results['coefficients']['intercept'].append(beta[0])
                bootstrap_results['coefficients']['bc_aod'].append(beta[1])
                bootstrap_results['coefficients']['temperature'].append(beta[2])
                bootstrap_results['coefficients']['snowfall'].append(beta[3])
                bootstrap_results['coefficients']['rainfall'].append(beta[4])
                
            except Exception:
                # Skip failed bootstrap iterations
                continue
        
        # Calculate stability statistics
        stability_stats = {}
        
        # R-squared stability
        r_sq_array = np.array(bootstrap_results['r_squared'])
        stability_stats['r_squared'] = {
            'mean': np.mean(r_sq_array),
            'std': np.std(r_sq_array),
            'ci_lower': np.percentile(r_sq_array, 2.5),
            'ci_upper': np.percentile(r_sq_array, 97.5)
        }
        
        # Coefficient stability
        stability_stats['coefficients'] = {}
        for coef_name, coef_values in bootstrap_results['coefficients'].items():
            coef_array = np.array(coef_values)
            stability_stats['coefficients'][coef_name] = {
                'mean': np.mean(coef_array),
                'std': np.std(coef_array),
                'ci_lower': np.percentile(coef_array, 2.5),
                'ci_upper': np.percentile(coef_array, 97.5)
            }
        
        # Create stability plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Bootstrap Stability Analysis (n={len(r_sq_array)} successful iterations)\nHaig Glacier - MODIS Terra+Aqua (June-September 2002-2024)', 
                    fontsize=16, fontweight='bold')
        
        # Plot R-squared distribution
        ax = axes[0, 0]
        ax.hist(r_sq_array, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(stability_stats['r_squared']['mean'], color='red', linestyle='--', linewidth=2, label=f"Mean = {stability_stats['r_squared']['mean']:.4f}")
        ax.set_xlabel('R²', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('R² Distribution', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot coefficient distributions
        coef_names = ['bc_aod', 'temperature', 'snowfall', 'rainfall', 'intercept']
        coef_labels = ['BC AOD', 'Temperature', 'Snowfall', 'Rainfall', 'Intercept']
        
        for i, (coef_name, coef_label) in enumerate(zip(coef_names, coef_labels)):
            if i < 4:
                ax = axes[0, i+1] if i < 2 else axes[1, i-2]
            else:
                ax = axes[1, 2]
            
            coef_values = bootstrap_results['coefficients'][coef_name]
            ax.hist(coef_values, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
            mean_val = stability_stats['coefficients'][coef_name]['mean']
            ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean = {mean_val:.6f}')
            ax.set_xlabel(f'{coef_label} Coefficient', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{coef_label} Distribution', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "stability_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Bootstrap stability test completed")
        return stability_stats
    
    def create_multiple_regression_plot(self, results: Dict[str, Any]) -> None:
        """Create the main multiple regression visualization (converted to observed vs predicted style)."""
        logger.info("Creating multiple regression visualization...")
        
        # This now delegates to the observed vs predicted plot for consistency
        self.create_observed_vs_predicted_plot(results)
        # Keep the logger message for compatibility
        logger.info("Multiple regression visualization saved")
    
    def create_diagnostic_plots(self, results: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
        """Create comprehensive regression diagnostic plots."""
        logger.info("Creating regression diagnostic plots...")
        
        data = results['data']
        residuals = results['residuals']
        y_pred = results['predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Multiple Regression Diagnostic Plots\nHaig Glacier - MODIS Terra+Aqua (June-September 2002-2024)', 
                    fontsize=16, fontweight='bold')
        
        # 1. Residuals vs Fitted Values
        ax1 = axes[0, 0]
        ax1.scatter(y_pred, residuals, alpha=0.6, edgecolors='black', linewidth=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax1.set_xlabel('Fitted Values', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.set_title('Residuals vs Fitted Values', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add LOWESS smooth line
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
            smoothed = lowess(residuals, y_pred, frac=0.3)
            ax1.plot(smoothed[:, 0], smoothed[:, 1], 'blue', linewidth=2, alpha=0.8)
        except ImportError:
            pass
        
        # 2. Q-Q Plot for normality
        ax2 = axes[0, 1]
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Test)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add Shapiro-Wilk test result
        if not np.isnan(diagnostics['shapiro_wilk_p']):
            ax2.text(0.05, 0.95, f"Shapiro-Wilk p = {diagnostics['shapiro_wilk_p']:.4f}", 
                    transform=ax2.transAxes, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 3. Scale-Location Plot (sqrt of standardized residuals vs fitted)
        ax3 = axes[1, 0]
        sqrt_std_resid = np.sqrt(np.abs(diagnostics['studentized_residuals']))
        ax3.scatter(y_pred, sqrt_std_resid, alpha=0.6, edgecolors='black', linewidth=0.3)
        ax3.set_xlabel('Fitted Values', fontsize=12)
        ax3.set_ylabel('√|Standardized Residuals|', fontsize=12)
        ax3.set_title('Scale-Location Plot', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Cook's Distance
        ax4 = axes[1, 1]
        n = len(data)
        ax4.stem(range(n), diagnostics['cooks_distance'], basefmt=" ")
        ax4.axhline(y=4/n, color='red', linestyle='--', alpha=0.8, label=f'Threshold = {4/n:.4f}')
        ax4.set_xlabel('Observation Index', fontsize=12)
        ax4.set_ylabel("Cook's Distance", fontsize=12)
        ax4.set_title("Cook's Distance", fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "regression_diagnostics.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Regression diagnostic plots saved")
    
    def create_correlation_matrix(self, results: Dict[str, Any]) -> None:
        """Create correlation matrix of all variables."""
        logger.info("Creating correlation matrix...")
        
        data = results['data']
        
        # Select variables for correlation matrix
        variables = ['bc_aod_regional', 'temperature_c', 'total_precip_mm', 'snowfall_mm', 'rainfall_mm', 'albedo_mean']
        var_labels = ['BC AOD', 'Temperature', 'Total Precip', 'Snowfall', 'Rainfall', 'Albedo']
        
        corr_data = data[variables]
        correlation_matrix = corr_data.corr()
        
        # Create heatmap
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                   square=True, xticklabels=var_labels, yticklabels=var_labels,
                   cbar_kws={"shrink": .8}, fmt='.3f', ax=ax)
        
        ax.set_title('Variable Correlation Matrix\nHaig Glacier - MODIS Terra+Aqua (June-September 2002-2024)', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "correlation_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Correlation matrix saved")
    
    def create_comprehensive_scatter_grid(self, results: Dict[str, Any]) -> None:
        """Create comprehensive 6-panel scatter plot grid showing all variable relationships."""
        logger.info("Creating comprehensive 6-panel scatter plot grid...")
        
        data = results['data']
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Comprehensive Variable Relationships: BC AOD, Temperature, Snowfall, Rainfall, and Albedo\nHaig Glacier - MODIS Terra+Aqua (June-September 2002-2024)', 
                    fontsize=16, fontweight='bold')
        
        # Panel 1: BC AOD vs Albedo (colored by temperature)
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(data['bc_aod_regional'], data['albedo_mean'], 
                              c=data['temperature_c'], s=50, alpha=0.7, 
                              cmap='coolwarm', edgecolors='black', linewidth=0.3)
        z1 = np.polyfit(data['bc_aod_regional'], data['albedo_mean'], 1)
        p1 = np.poly1d(z1)
        x_reg1 = np.linspace(data['bc_aod_regional'].min(), data['bc_aod_regional'].max(), 100)
        ax1.plot(x_reg1, p1(x_reg1), "darkblue", linestyle='--', alpha=0.8, linewidth=2)
        ax1.set_xlabel('BC AOD (Regional)', fontsize=11)
        ax1.set_ylabel('MODIS Albedo', fontsize=11)
        bc_r = results['individual_correlations']['bc_aod']['r']
        bc_p = results['individual_correlations']['bc_aod']['p']
        ax1.set_title(f'BC AOD vs Albedo\nr = {bc_r:.3f}, p = {bc_p:.4f}', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Temperature vs Albedo (colored by temperature)
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(data['temperature_c'], data['albedo_mean'], 
                              c=data['temperature_c'], s=50, alpha=0.7, 
                              cmap='coolwarm', edgecolors='black', linewidth=0.3)
        z2 = np.polyfit(data['temperature_c'], data['albedo_mean'], 1)
        p2 = np.poly1d(z2)
        x_reg2 = np.linspace(data['temperature_c'].min(), data['temperature_c'].max(), 100)
        ax2.plot(x_reg2, p2(x_reg2), "darkblue", linestyle='--', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Air Temperature (°C)', fontsize=11)
        ax2.set_ylabel('MODIS Albedo', fontsize=11)
        temp_r = results['individual_correlations']['temperature']['r']
        temp_p = results['individual_correlations']['temperature']['p']
        ax2.set_title(f'Temperature vs Albedo\nr = {temp_r:.3f}, p = {temp_p:.4f}', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Snowfall vs Albedo (colored by temperature) - moved from panel 4
        ax3 = axes[0, 2]
        scatter3 = ax3.scatter(data['snowfall_mm'], data['albedo_mean'], 
                              c=data['temperature_c'], s=50, alpha=0.7, 
                              cmap='coolwarm', edgecolors='black', linewidth=0.3)
        z3 = np.polyfit(data['snowfall_mm'], data['albedo_mean'], 1)
        p3 = np.poly1d(z3)
        x_reg3 = np.linspace(data['snowfall_mm'].min(), data['snowfall_mm'].max(), 100)
        ax3.plot(x_reg3, p3(x_reg3), "darkblue", linestyle='--', alpha=0.8, linewidth=2)
        ax3.set_xlabel('Snowfall (mm)', fontsize=11)
        ax3.set_ylabel('MODIS Albedo', fontsize=11)
        snow_r = results['individual_correlations']['snowfall']['r']
        snow_p = results['individual_correlations']['snowfall']['p']
        ax3.set_title(f'Snowfall vs Albedo\nr = {snow_r:.3f}, p = {snow_p:.4f}', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Rainfall vs Albedo (colored by temperature)
        ax4 = axes[1, 0]
        scatter4 = ax4.scatter(data['rainfall_mm'], data['albedo_mean'], 
                              c=data['temperature_c'], s=50, alpha=0.7, 
                              cmap='coolwarm', edgecolors='black', linewidth=0.3)
        z4 = np.polyfit(data['rainfall_mm'], data['albedo_mean'], 1)
        p4 = np.poly1d(z4)
        x_reg4 = np.linspace(data['rainfall_mm'].min(), data['rainfall_mm'].max(), 100)
        ax4.plot(x_reg4, p4(x_reg4), "darkblue", linestyle='--', alpha=0.8, linewidth=2)
        ax4.set_xlabel('Rainfall (mm)', fontsize=11)
        ax4.set_ylabel('MODIS Albedo', fontsize=11)
        rain_r = results['individual_correlations']['rainfall']['r']
        rain_p = results['individual_correlations']['rainfall']['p']
        ax4.set_title(f'Rainfall vs Albedo\nr = {rain_r:.3f}, p = {rain_p:.4f}', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Panel 5: Total Precipitation vs Albedo (colored by temperature) - for comparison
        ax5 = axes[1, 1]
        scatter5 = ax5.scatter(data['total_precip_mm'], data['albedo_mean'], 
                              c=data['temperature_c'], s=50, alpha=0.7, 
                              cmap='coolwarm', edgecolors='black', linewidth=0.3)
        z5 = np.polyfit(data['total_precip_mm'], data['albedo_mean'], 1)
        p5 = np.poly1d(z5)
        x_reg5 = np.linspace(data['total_precip_mm'].min(), data['total_precip_mm'].max(), 100)
        ax5.plot(x_reg5, p5(x_reg5), "darkblue", linestyle='--', alpha=0.8, linewidth=2)
        ax5.set_xlabel('Total Precipitation (mm)', fontsize=11)
        ax5.set_ylabel('MODIS Albedo', fontsize=11)
        # Calculate total precipitation correlation for comparison
        total_precip_r, total_precip_p = pearsonr(data['total_precip_mm'], data['albedo_mean'])
        ax5.set_title(f'Total Precip vs Albedo\nr = {total_precip_r:.3f}, p = {total_precip_p:.4f}', fontsize=12)
        ax5.grid(True, alpha=0.3)
        
        # Panel 6: Multiple Regression Model Performance (Predicted vs Observed)
        ax6 = axes[1, 2]
        scatter6 = ax6.scatter(data['albedo_mean'], results['predictions'], 
                              c=data['temperature_c'], s=50, alpha=0.7, 
                              cmap='coolwarm', edgecolors='black', linewidth=0.3)
        # Add 1:1 reference line
        min_val = min(data['albedo_mean'].min(), results['predictions'].min())
        max_val = max(data['albedo_mean'].max(), results['predictions'].max())
        ax6.plot([min_val, max_val], [min_val, max_val], 'darkred', linestyle='--', alpha=0.8, linewidth=2, label='1:1 Line')
        ax6.set_xlabel('Observed Albedo', fontsize=11)
        ax6.set_ylabel('Predicted Albedo', fontsize=11)
        multiple_r = np.sqrt(results['multiple_r_squared'])
        ax6.set_title(f'Multiple Regression Model\nR² = {results["multiple_r_squared"]:.3f}, Adj R² = {results["adjusted_r_squared"]:.3f}', fontsize=12)
        ax6.legend(fontsize=10)
        ax6.grid(True, alpha=0.3)
        
        # Position tight_layout first, then adjust for colorbar space
        plt.tight_layout()
        plt.subplots_adjust(right=0.85)  # Make room for colorbar
        
        # Add colorbar for temperature (shared across all plots) - positioned at far right
        cbar = fig.colorbar(scatter1, ax=axes.ravel().tolist(), shrink=0.7, pad=0.08, aspect=25)
        cbar.set_label('Air Temperature (°C)', fontsize=12)
        plt.savefig(self.output_dir / "plots" / "comprehensive_scatter_grid.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Comprehensive scatter grid saved")
    
    def save_results(self, results: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
        """Save all results to CSV files."""
        logger.info("Saving results...")
        
        # Main results summary
        summary_data = {
            'n_observations': [results['n_observations']],
            'multiple_r_squared': [results['multiple_r_squared']],
            'adjusted_r_squared': [results['adjusted_r_squared']],
            'f_statistic': [results['f_statistic']],
            'f_p_value': [results['f_p_value']],
            'intercept': [results['coefficients']['intercept']],
            'bc_aod_coef': [results['coefficients']['bc_aod']],
            'temperature_coef': [results['coefficients']['temperature']],
            'snowfall_coef': [results['coefficients']['snowfall']],
            'rainfall_coef': [results['coefficients']['rainfall']],
            'bc_aod_beta': [results['standardized_coefficients']['bc_aod']],
            'temperature_beta': [results['standardized_coefficients']['temperature']],
            'snowfall_beta': [results['standardized_coefficients']['snowfall']],
            'rainfall_beta': [results['standardized_coefficients']['rainfall']],
            'shapiro_wilk_p': [diagnostics['shapiro_wilk_p']],
            'homoscedasticity_p': [diagnostics['homoscedasticity_p']]
        }
        
        # Add VIF values
        for var, vif in diagnostics['vif_values'].items():
            summary_data[f'vif_{var}'] = [vif]
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / "results" / "multiple_regression_summary.csv", index=False)
        
        # Individual coefficients with statistics
        coef_data = {
            'variable': ['intercept', 'bc_aod', 'temperature', 'snowfall', 'rainfall'],
            'coefficient': [
                results['coefficients']['intercept'],
                results['coefficients']['bc_aod'],
                results['coefficients']['temperature'],
                results['coefficients']['snowfall'],
                results['coefficients']['rainfall']
            ],
            'standard_error': results['standard_errors'],
            't_statistic': results['t_statistics'],
            'p_value': results['p_values']
        }
        
        coef_df = pd.DataFrame(coef_data)
        coef_df.to_csv(self.output_dir / "results" / "regression_coefficients.csv", index=False)
        
        # Diagnostic data
        data = results['data'].copy()
        data['predicted_albedo'] = results['predictions']
        data['residuals'] = results['residuals']
        data['leverage'] = diagnostics['leverage']
        data['cooks_distance'] = diagnostics['cooks_distance']
        data['studentized_residuals'] = diagnostics['studentized_residuals']
        
        data.to_csv(self.output_dir / "results" / "regression_data_diagnostics.csv", index=False)
        
        # Individual correlations
        corr_data = []
        for var, stats in results['individual_correlations'].items():
            corr_data.append({
                'variable': var,
                'correlation_r': stats['r'],
                'correlation_p': stats['p'],
                'r_squared': stats['r']**2
            })
        
        corr_df = pd.DataFrame(corr_data)
        corr_df.to_csv(self.output_dir / "results" / "individual_correlations.csv", index=False)
        
        logger.info("All results saved successfully")
    
    def run_complete_analysis(self) -> None:
        """Run the complete multiple regression analysis."""
        logger.info("Starting complete multiple regression analysis...")
        
        # Load data
        data = self.load_data()
        
        # Fit model
        results = self.fit_multiple_regression(data)
        
        # Calculate diagnostics
        diagnostics = self.calculate_diagnostics(results)
        
        # Create visualizations (prioritized order)
        # Essential (Priority 1)
        self.create_observed_vs_predicted_plot(results)
        self.create_residuals_vs_fitted_plot(results, diagnostics)
        self.create_qq_plot(results, diagnostics)
        
        # Advanced (Priority 2)
        self.create_partial_regression_plots(results)
        self.create_marginal_effects_plots(results)
        self.create_predictor_importance_plot(results)
        
        # Contextual (Priority 3)
        self.create_comprehensive_scatter_grid(results)
        self.create_diagnostic_plots(results, diagnostics)
        self.create_correlation_matrix(results)
        
        # Validation and testing
        validation_results = self.validate_improvements(data)
        stability_results = self.run_stability_test(data)
        
        # Save results
        self.save_results(results, diagnostics)
        
        # Print summary
        self.print_summary(results, diagnostics)
        
        logger.info(f"Analysis complete! Results saved to: {self.output_dir}")
    
    def print_summary(self, results: Dict[str, Any], diagnostics: Dict[str, Any]) -> None:
        """Print comprehensive analysis summary."""
        print(f"\n{'='*80}")
        print(f"MULTIPLE REGRESSION ANALYSIS RESULTS")
        print(f"{'='*80}")
        print(f"Sample size: {results['n_observations']} observations")
        print(f"\nModel: Albedo ~ BC AOD + Temperature + Snowfall + Rainfall")
        
        print(f"\n{'Model Statistics:':<25}")
        print(f"{'Multiple R²:':<25} {results['multiple_r_squared']:.4f}")
        print(f"{'Adjusted R²:':<25} {results['adjusted_r_squared']:.4f}")
        print(f"{'F-statistic:':<25} {results['f_statistic']:.4f}")
        print(f"{'F p-value:':<25} {results['f_p_value']:.6f}")
        
        print(f"\n{'Coefficients:':<25}")
        coef = results['coefficients']
        p_vals = results['p_values']
        print(f"{'Intercept:':<25} {coef['intercept']:>8.4f} (p = {p_vals[0]:.4f})")
        print(f"{'BC AOD:':<25} {coef['bc_aod']:>8.4f} (p = {p_vals[1]:.4f})")
        print(f"{'Temperature:':<25} {coef['temperature']:>8.4f} (p = {p_vals[2]:.4f})")
        print(f"{'Snowfall:':<25} {coef['snowfall']:>8.4f} (p = {p_vals[3]:.4f})")
        print(f"{'Rainfall:':<25} {coef['rainfall']:>8.4f} (p = {p_vals[4]:.4f})")
        
        print(f"\n{'Standardized Coefficients (Beta):':<25}")
        std_coef = results['standardized_coefficients']
        print(f"{'BC AOD:':<25} {std_coef['bc_aod']:>8.4f}")
        print(f"{'Temperature:':<25} {std_coef['temperature']:>8.4f}")
        print(f"{'Snowfall:':<25} {std_coef['snowfall']:>8.4f}")
        print(f"{'Rainfall:':<25} {std_coef['rainfall']:>8.4f}")
        
        print(f"\n{'Individual Correlations with Albedo:':<25}")
        ind_corr = results['individual_correlations']
        for var, stats in ind_corr.items():
            print(f"{var.replace('_', ' ').title()+':':<25} r = {stats['r']:>7.4f} (p = {stats['p']:.4f})")
        
        print(f"\n{'Regression Diagnostics:':<25}")
        print(f"{'Normality (Shapiro-Wilk p):':<25} {diagnostics['shapiro_wilk_p']:.4f}")
        print(f"{'Homoscedasticity p:':<25} {diagnostics['homoscedasticity_p']:.4f}")
        
        print(f"\n{'Variance Inflation Factors:':<25}")
        for var, vif in diagnostics['vif_values'].items():
            print(f"{var.replace('_', ' ').title()+':':<25} {vif:>8.2f}")
        
        print(f"\n{'Potential Outliers:':<25}")
        print(f"{'High leverage:':<25} {len(diagnostics['outliers_high_leverage'])}")
        print(f"{'High Cook distance:':<25} {len(diagnostics['outliers_high_cooks'])}")
        print(f"{'High residuals:':<25} {len(diagnostics['outliers_high_residuals'])}")
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"{'='*80}")


# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def main():
    """Main function to run the analysis."""
    logger.info("Starting Multiple Regression Analysis")
    
    analyzer = MultipleRegressionAnalyzer(CONFIG)
    analyzer.run_complete_analysis()
    
    logger.info("Analysis completed successfully!")


if __name__ == "__main__":
    main()