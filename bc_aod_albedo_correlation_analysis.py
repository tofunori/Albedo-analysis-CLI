#!/usr/bin/env python3
"""
Black Carbon AOD vs MODIS Albedo Correlation Analysis

This script analyzes the relationship between black carbon aerosol optical depth (bc_aod_regional)
from MERRA2 data and surface albedo from MODIS MOD09GA data for Haig Glacier.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import logging
from pathlib import Path
from typing import Tuple, Dict, Any

# Import existing framework modules
from visualization.plots.correlation_plots import CorrelationPlotter
from visualization.plots.time_series_plots import TimeSeriesPlotter
from visualization.plots.base import BasePlotter

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BCAlbedoCorrelationAnalyzer:
    """Analyzer for Black Carbon AOD vs Albedo correlation analysis."""
    
    def __init__(self, merra2_path: str, modis_path: str, output_dir: str = None):
        """Initialize the analyzer with data paths."""
        self.merra2_path = merra2_path
        self.modis_path = modis_path
        
        # Create timestamped output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_dir = Path(f"outputs/bc_aod_albedo_correlation_{timestamp}")
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        (self.output_dir / "results").mkdir(exist_ok=True)
        
        # Initialize plotters with basic config
        config = {'output_dir': str(self.output_dir / "plots")}
        self.correlation_plotter = CorrelationPlotter(config)
        self.timeseries_plotter = TimeSeriesPlotter(config)
        
        logger.info(f"Initialized BC-Albedo correlation analyzer. Output: {self.output_dir}")
    
    def load_and_process_data(self) -> pd.DataFrame:
        """Load both datasets and create merged dataset with complete paired observations."""
        logger.info("Loading MERRA2 data...")
        
        # Load MERRA2 data
        merra2_df = pd.read_csv(self.merra2_path)
        merra2_df['date'] = pd.to_datetime(merra2_df['date'])
        logger.info(f"Loaded {len(merra2_df)} MERRA2 records")
        
        # Load MODIS data
        logger.info("Loading MODIS data...")
        modis_df = pd.read_csv(self.modis_path)
        modis_df['date'] = pd.to_datetime(modis_df['date'])
        logger.info(f"Loaded {len(modis_df)} MODIS records")
        
        # Filter MODIS data to match MERRA2 temporal range (June-September, 2022-2024)
        logger.info("Filtering MODIS data to June-September 2022-2024...")
        modis_filtered = modis_df[
            (modis_df['date'].dt.year.isin([2022, 2023, 2024])) &
            (modis_df['date'].dt.month.isin([6, 7, 8, 9]))
        ].copy()
        logger.info(f"Filtered to {len(modis_filtered)} MODIS records")
        
        # Calculate daily mean albedo from multiple pixels
        logger.info("Calculating daily mean albedo...")
        daily_albedo = modis_filtered.groupby('date').agg({
            'albedo': ['mean', 'std', 'count'],
            'glacier_fraction': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_albedo.columns = ['date', 'albedo_mean', 'albedo_std', 'pixel_count', 'glacier_fraction_mean']
        logger.info(f"Created {len(daily_albedo)} daily mean albedo records")
        
        # Merge datasets on date
        logger.info("Merging datasets...")
        merged_df = pd.merge(
            merra2_df[['date', 'bc_aod_regional', 'temperature_c', 'total_precip_mm', 'snowfall_mm', 'rainfall_mm', 'year', 'month', 'day_of_year']],
            daily_albedo,
            on='date',
            how='inner'
        )
        
        # Keep only complete paired observations (including all precipitation variables)
        complete_pairs = merged_df.dropna(subset=['bc_aod_regional', 'temperature_c', 'total_precip_mm', 'snowfall_mm', 'rainfall_mm', 'albedo_mean'])
        logger.info(f"Found {len(complete_pairs)} complete paired observations with BC AOD, temperature, precipitation, and albedo data")
        
        # Add seasonal information
        complete_pairs['season'] = complete_pairs['month'].map({
            6: 'June', 7: 'July', 8: 'August', 9: 'September'
        })
        
        return complete_pairs
    
    def filter_outliers(self, data: pd.DataFrame, threshold_sd: float = 2.5) -> pd.DataFrame:
        """Filter outliers from albedo data using standard deviation threshold."""
        logger.info(f"Filtering albedo outliers using {threshold_sd} standard deviation threshold...")
        
        # Calculate mean and standard deviation for albedo
        albedo_mean = data['albedo_mean'].mean()
        albedo_std = data['albedo_mean'].std()
        
        # Define outlier bounds
        lower_bound = albedo_mean - threshold_sd * albedo_std
        upper_bound = albedo_mean + threshold_sd * albedo_std
        
        # Filter data
        original_count = len(data)
        filtered_data = data[
            (data['albedo_mean'] >= lower_bound) & 
            (data['albedo_mean'] <= upper_bound)
        ].copy()
        
        outliers_removed = original_count - len(filtered_data)
        outlier_percentage = (outliers_removed / original_count) * 100
        
        logger.info(f"Removed {outliers_removed} outliers ({outlier_percentage:.1f}%) from {original_count} observations")
        logger.info(f"Albedo bounds: {lower_bound:.3f} to {upper_bound:.3f}")
        logger.info(f"Remaining observations: {len(filtered_data)}")
        
        return filtered_data
    
    def calculate_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive correlation statistics for all five variables."""
        logger.info("Calculating comprehensive correlation statistics...")
        
        # Remove any remaining NaN values for all variables
        clean_data = data[['bc_aod_regional', 'temperature_c', 'total_precip_mm', 'snowfall_mm', 'rainfall_mm', 'albedo_mean']].dropna()
        
        # Calculate pairwise correlations with albedo
        # BC AOD vs Albedo
        bc_albedo_pearson_r, bc_albedo_pearson_p = pearsonr(clean_data['bc_aod_regional'], clean_data['albedo_mean'])
        bc_albedo_spearman_r, bc_albedo_spearman_p = spearmanr(clean_data['bc_aod_regional'], clean_data['albedo_mean'])
        
        # Temperature vs Albedo
        temp_albedo_pearson_r, temp_albedo_pearson_p = pearsonr(clean_data['temperature_c'], clean_data['albedo_mean'])
        temp_albedo_spearman_r, temp_albedo_spearman_p = spearmanr(clean_data['temperature_c'], clean_data['albedo_mean'])
        
        # Total Precipitation vs Albedo
        total_precip_albedo_pearson_r, total_precip_albedo_pearson_p = pearsonr(clean_data['total_precip_mm'], clean_data['albedo_mean'])
        total_precip_albedo_spearman_r, total_precip_albedo_spearman_p = spearmanr(clean_data['total_precip_mm'], clean_data['albedo_mean'])
        
        # Snowfall vs Albedo
        snow_albedo_pearson_r, snow_albedo_pearson_p = pearsonr(clean_data['snowfall_mm'], clean_data['albedo_mean'])
        snow_albedo_spearman_r, snow_albedo_spearman_p = spearmanr(clean_data['snowfall_mm'], clean_data['albedo_mean'])
        
        # Rainfall vs Albedo
        rain_albedo_pearson_r, rain_albedo_pearson_p = pearsonr(clean_data['rainfall_mm'], clean_data['albedo_mean'])
        rain_albedo_spearman_r, rain_albedo_spearman_p = spearmanr(clean_data['rainfall_mm'], clean_data['albedo_mean'])
        
        # Additional inter-variable correlations
        # BC AOD vs Temperature
        bc_temp_pearson_r, bc_temp_pearson_p = pearsonr(clean_data['bc_aod_regional'], clean_data['temperature_c'])
        bc_temp_spearman_r, bc_temp_spearman_p = spearmanr(clean_data['bc_aod_regional'], clean_data['temperature_c'])
        
        # Calculate R-squared values
        bc_albedo_r_squared = bc_albedo_pearson_r ** 2
        temp_albedo_r_squared = temp_albedo_pearson_r ** 2
        total_precip_albedo_r_squared = total_precip_albedo_pearson_r ** 2
        snow_albedo_r_squared = snow_albedo_pearson_r ** 2
        rain_albedo_r_squared = rain_albedo_pearson_r ** 2
        bc_temp_r_squared = bc_temp_pearson_r ** 2
        
        # Calculate confidence intervals (95%)
        n = len(clean_data)
        bc_albedo_ci = self._correlation_confidence_interval(bc_albedo_pearson_r, n)
        temp_albedo_ci = self._correlation_confidence_interval(temp_albedo_pearson_r, n)
        total_precip_albedo_ci = self._correlation_confidence_interval(total_precip_albedo_pearson_r, n)
        snow_albedo_ci = self._correlation_confidence_interval(snow_albedo_pearson_r, n)
        rain_albedo_ci = self._correlation_confidence_interval(rain_albedo_pearson_r, n)
        bc_temp_ci = self._correlation_confidence_interval(bc_temp_pearson_r, n)
        
        # Calculate multiple regression: Albedo ~ BC AOD + Temperature + Total Precip + Snowfall using numpy
        # Prepare design matrix (add intercept column)
        X = np.column_stack([
            np.ones(n), 
            clean_data['bc_aod_regional'], 
            clean_data['temperature_c'],
            clean_data['total_precip_mm'],
            clean_data['snowfall_mm']
        ])
        y = clean_data['albedo_mean'].values
        
        # Fit multiple regression using normal equation: beta = (X'X)^(-1)X'y
        try:
            beta = np.linalg.solve(X.T @ X, X.T @ y)
            y_pred = X @ beta
            
            # Calculate R-squared
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            multiple_r_squared = 1 - (ss_res / ss_tot)
            
            # Calculate adjusted R-squared
            k = X.shape[1] - 1  # Number of predictors (excluding intercept)
            adjusted_r_squared = 1 - (1 - multiple_r_squared) * (n - 1) / (n - k - 1)
            
            # Calculate F-statistic for multiple regression
            mse_reg = (ss_tot - ss_res) / k
            mse_res = ss_res / (n - k - 1)
            f_statistic = mse_reg / mse_res
            
            intercept = beta[0]
            bc_aod_coef = beta[1]
            temp_coef = beta[2]
            total_precip_coef = beta[3]
            snowfall_coef = beta[4]
            
            # Calculate standardized coefficients (beta coefficients)
            # Standardize variables: z = (x - mean) / std
            bc_aod_std = clean_data['bc_aod_regional'].std()
            temp_std = clean_data['temperature_c'].std()
            total_precip_std = clean_data['total_precip_mm'].std()
            snowfall_std = clean_data['snowfall_mm'].std()
            albedo_std = clean_data['albedo_mean'].std()
            
            # Standardized coefficients (beta) = unstandardized_coef * (std_x / std_y)
            bc_aod_beta = bc_aod_coef * (bc_aod_std / albedo_std)
            temp_beta = temp_coef * (temp_std / albedo_std)
            total_precip_beta = total_precip_coef * (total_precip_std / albedo_std)
            snowfall_beta = snowfall_coef * (snowfall_std / albedo_std)
            
        except np.linalg.LinAlgError:
            # Fallback if matrix is singular
            multiple_r_squared = np.nan
            adjusted_r_squared = np.nan
            f_statistic = np.nan
            intercept = np.nan
            bc_aod_coef = np.nan
            temp_coef = np.nan
            total_precip_coef = np.nan
            snowfall_coef = np.nan
            bc_aod_beta = np.nan
            temp_beta = np.nan
            total_precip_beta = np.nan
            snowfall_beta = np.nan
            y_pred = np.full_like(y, np.mean(y))
        
        stats_dict = {
            'sample_size': n,
            
            # BC AOD vs Albedo (primary relationship)
            'bc_albedo_pearson_r': bc_albedo_pearson_r,
            'bc_albedo_pearson_p': bc_albedo_pearson_p,
            'bc_albedo_ci_lower': bc_albedo_ci[0],
            'bc_albedo_ci_upper': bc_albedo_ci[1],
            'bc_albedo_spearman_r': bc_albedo_spearman_r,
            'bc_albedo_spearman_p': bc_albedo_spearman_p,
            'bc_albedo_r_squared': bc_albedo_r_squared,
            
            # Temperature vs Albedo
            'temp_albedo_pearson_r': temp_albedo_pearson_r,
            'temp_albedo_pearson_p': temp_albedo_pearson_p,
            'temp_albedo_ci_lower': temp_albedo_ci[0],
            'temp_albedo_ci_upper': temp_albedo_ci[1],
            'temp_albedo_spearman_r': temp_albedo_spearman_r,
            'temp_albedo_spearman_p': temp_albedo_spearman_p,
            'temp_albedo_r_squared': temp_albedo_r_squared,
            
            # Total Precipitation vs Albedo
            'total_precip_albedo_pearson_r': total_precip_albedo_pearson_r,
            'total_precip_albedo_pearson_p': total_precip_albedo_pearson_p,
            'total_precip_albedo_ci_lower': total_precip_albedo_ci[0],
            'total_precip_albedo_ci_upper': total_precip_albedo_ci[1],
            'total_precip_albedo_spearman_r': total_precip_albedo_spearman_r,
            'total_precip_albedo_spearman_p': total_precip_albedo_spearman_p,
            'total_precip_albedo_r_squared': total_precip_albedo_r_squared,
            
            # Snowfall vs Albedo
            'snow_albedo_pearson_r': snow_albedo_pearson_r,
            'snow_albedo_pearson_p': snow_albedo_pearson_p,
            'snow_albedo_ci_lower': snow_albedo_ci[0],
            'snow_albedo_ci_upper': snow_albedo_ci[1],
            'snow_albedo_spearman_r': snow_albedo_spearman_r,
            'snow_albedo_spearman_p': snow_albedo_spearman_p,
            'snow_albedo_r_squared': snow_albedo_r_squared,
            
            # Rainfall vs Albedo
            'rain_albedo_pearson_r': rain_albedo_pearson_r,
            'rain_albedo_pearson_p': rain_albedo_pearson_p,
            'rain_albedo_ci_lower': rain_albedo_ci[0],
            'rain_albedo_ci_upper': rain_albedo_ci[1],
            'rain_albedo_spearman_r': rain_albedo_spearman_r,
            'rain_albedo_spearman_p': rain_albedo_spearman_p,
            'rain_albedo_r_squared': rain_albedo_r_squared,
            
            # BC AOD vs Temperature
            'bc_temp_pearson_r': bc_temp_pearson_r,
            'bc_temp_pearson_p': bc_temp_pearson_p,
            'bc_temp_ci_lower': bc_temp_ci[0],
            'bc_temp_ci_upper': bc_temp_ci[1],
            'bc_temp_spearman_r': bc_temp_spearman_r,
            'bc_temp_spearman_p': bc_temp_spearman_p,
            'bc_temp_r_squared': bc_temp_r_squared,
            
            # Multiple regression results (4-predictor model)
            'multiple_r_squared': multiple_r_squared,
            'adjusted_r_squared': adjusted_r_squared,
            'f_statistic': f_statistic,
            'bc_aod_coefficient': bc_aod_coef,
            'temperature_coefficient': temp_coef,
            'total_precip_coefficient': total_precip_coef,
            'snowfall_coefficient': snowfall_coef,
            'intercept': intercept,
            
            # Standardized coefficients (beta coefficients)
            'bc_aod_beta_standardized': bc_aod_beta,
            'temperature_beta_standardized': temp_beta,
            'total_precip_beta_standardized': total_precip_beta,
            'snowfall_beta_standardized': snowfall_beta,
            
            # Variable statistics
            'bc_aod_mean': clean_data['bc_aod_regional'].mean(),
            'bc_aod_std': clean_data['bc_aod_regional'].std(),
            'temperature_mean': clean_data['temperature_c'].mean(),
            'temperature_std': clean_data['temperature_c'].std(),
            'total_precip_mean': clean_data['total_precip_mm'].mean(),
            'total_precip_std': clean_data['total_precip_mm'].std(),
            'snowfall_mean': clean_data['snowfall_mm'].mean(),
            'snowfall_std': clean_data['snowfall_mm'].std(),
            'rainfall_mean': clean_data['rainfall_mm'].mean(),
            'rainfall_std': clean_data['rainfall_mm'].std(),
            'albedo_mean': clean_data['albedo_mean'].mean(),
            'albedo_std': clean_data['albedo_mean'].std(),
            
            # Legacy values for backward compatibility
            'pearson_r': bc_albedo_pearson_r,  # For existing scatter plot
            'pearson_p': bc_albedo_pearson_p,
            'pearson_ci_lower': bc_albedo_ci[0],
            'pearson_ci_upper': bc_albedo_ci[1],
            'spearman_r': bc_albedo_spearman_r,
            'spearman_p': bc_albedo_spearman_p,
            'r_squared': bc_albedo_r_squared
        }
        
        logger.info(f"BC AOD vs Albedo: Pearson r = {bc_albedo_pearson_r:.4f} (p = {bc_albedo_pearson_p:.4f})")
        logger.info(f"Temperature vs Albedo: Pearson r = {temp_albedo_pearson_r:.4f} (p = {temp_albedo_pearson_p:.4f})")
        logger.info(f"Total Precip vs Albedo: Pearson r = {total_precip_albedo_pearson_r:.4f} (p = {total_precip_albedo_pearson_p:.4f})")
        logger.info(f"Snowfall vs Albedo: Pearson r = {snow_albedo_pearson_r:.4f} (p = {snow_albedo_pearson_p:.4f})")
        logger.info(f"Rainfall vs Albedo: Pearson r = {rain_albedo_pearson_r:.4f} (p = {rain_albedo_pearson_p:.4f})")
        logger.info(f"Multiple regression (4-predictors) R² = {multiple_r_squared:.4f}, Adjusted R² = {adjusted_r_squared:.4f}")
        
        return stats_dict
    
    def _correlation_confidence_interval(self, r: float, n: int, alpha: float = 0.05) -> Tuple[float, float]:
        """Calculate confidence interval for correlation coefficient."""
        # Fisher transformation
        z = np.arctanh(r)
        se = 1 / np.sqrt(n - 3)
        z_crit = stats.norm.ppf(1 - alpha/2)
        
        # Confidence interval in z-space
        z_lower = z - z_crit * se
        z_upper = z + z_crit * se
        
        # Transform back to r-space
        r_lower = np.tanh(z_lower)
        r_upper = np.tanh(z_upper)
        
        return (r_lower, r_upper)
    
    def create_scatter_plot(self, data: pd.DataFrame, stats: Dict[str, Any]) -> None:
        """Create scatter plot with regression line and temperature coloring."""
        logger.info("Creating enhanced scatter plot with temperature coloring...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot colored by temperature
        scatter = ax.scatter(data['bc_aod_regional'], data['albedo_mean'], 
                           c=data['temperature_c'], s=60, alpha=0.7, 
                           cmap='coolwarm', edgecolors='black', linewidth=0.5)
        
        # Add regression line
        z = np.polyfit(data['bc_aod_regional'], data['albedo_mean'], 1)
        p = np.poly1d(z)
        x_reg = np.linspace(data['bc_aod_regional'].min(), data['bc_aod_regional'].max(), 100)
        ax.plot(x_reg, p(x_reg), "darkblue", linestyle='--', alpha=0.8, linewidth=2, 
               label='BC AOD-Albedo Regression')
        
        # Formatting
        ax.set_xlabel('Black Carbon AOD (Regional)', fontsize=12)
        ax.set_ylabel('MODIS Albedo (Daily Mean)', fontsize=12)
        ax.set_title('Black Carbon AOD vs Surface Albedo (Colored by Temperature)\nHaig Glacier (June-September 2022-2024)', 
                    fontsize=14, fontweight='bold')
        
        # Add colorbar for temperature
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Air Temperature (°C)', fontsize=12)
        
        # Add statistics text
        stats_text = f'BC AOD-Albedo: r = {stats["bc_albedo_pearson_r"]:.4f} (p = {stats["bc_albedo_pearson_p"]:.4f})\n'
        stats_text += f'Multiple R² = {stats["multiple_r_squared"]:.4f}\n'
        stats_text += f'n = {stats["sample_size"]}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "bc_aod_albedo_scatter_temp.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Enhanced scatter plot saved")
    
    def create_precipitation_scatter_plots(self, data: pd.DataFrame, stats: Dict[str, Any]) -> None:
        """Create scatter plots for precipitation variables vs albedo."""
        logger.info("Creating precipitation vs albedo scatter plots...")
        
        # Create subplot layout for 3 precipitation plots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Total Precipitation vs Albedo
        ax1 = axes[0]
        scatter1 = ax1.scatter(data['total_precip_mm'], data['albedo_mean'], 
                             c=data['temperature_c'], s=60, alpha=0.7, 
                             cmap='coolwarm', edgecolors='black', linewidth=0.5)
        
        # Add regression line
        z1 = np.polyfit(data['total_precip_mm'], data['albedo_mean'], 1)
        p1 = np.poly1d(z1)
        x_reg1 = np.linspace(data['total_precip_mm'].min(), data['total_precip_mm'].max(), 100)
        ax1.plot(x_reg1, p1(x_reg1), "darkblue", linestyle='--', alpha=0.8, linewidth=2)
        
        ax1.set_xlabel('Total Precipitation (mm)', fontsize=12)
        ax1.set_ylabel('MODIS Albedo (Daily Mean)', fontsize=12)
        ax1.set_title(f'Total Precipitation vs Albedo\nr = {stats["total_precip_albedo_pearson_r"]:.4f} (p = {stats["total_precip_albedo_pearson_p"]:.4f})', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Snowfall vs Albedo
        ax2 = axes[1]
        scatter2 = ax2.scatter(data['snowfall_mm'], data['albedo_mean'], 
                             c=data['temperature_c'], s=60, alpha=0.7, 
                             cmap='coolwarm', edgecolors='black', linewidth=0.5)
        
        # Add regression line
        z2 = np.polyfit(data['snowfall_mm'], data['albedo_mean'], 1)
        p2 = np.poly1d(z2)
        x_reg2 = np.linspace(data['snowfall_mm'].min(), data['snowfall_mm'].max(), 100)
        ax2.plot(x_reg2, p2(x_reg2), "darkblue", linestyle='--', alpha=0.8, linewidth=2)
        
        ax2.set_xlabel('Snowfall (mm)', fontsize=12)
        ax2.set_ylabel('MODIS Albedo (Daily Mean)', fontsize=12)
        ax2.set_title(f'Snowfall vs Albedo\nr = {stats["snow_albedo_pearson_r"]:.4f} (p = {stats["snow_albedo_pearson_p"]:.4f})', 
                     fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Rainfall vs Albedo
        ax3 = axes[2]
        scatter3 = ax3.scatter(data['rainfall_mm'], data['albedo_mean'], 
                             c=data['temperature_c'], s=60, alpha=0.7, 
                             cmap='coolwarm', edgecolors='black', linewidth=0.5)
        
        # Add regression line
        z3 = np.polyfit(data['rainfall_mm'], data['albedo_mean'], 1)
        p3 = np.poly1d(z3)
        x_reg3 = np.linspace(data['rainfall_mm'].min(), data['rainfall_mm'].max(), 100)
        ax3.plot(x_reg3, p3(x_reg3), "darkblue", linestyle='--', alpha=0.8, linewidth=2)
        
        ax3.set_xlabel('Rainfall (mm)', fontsize=12)
        ax3.set_ylabel('MODIS Albedo (Daily Mean)', fontsize=12)
        ax3.set_title(f'Rainfall vs Albedo\nr = {stats["rain_albedo_pearson_r"]:.4f} (p = {stats["rain_albedo_pearson_p"]:.4f})', 
                     fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar for temperature (shared across all plots)
        cbar = fig.colorbar(scatter1, ax=axes, shrink=0.8, pad=0.02)
        cbar.set_label('Air Temperature (°C)', fontsize=12)
        
        plt.suptitle('Precipitation Variables vs Surface Albedo (Colored by Temperature)\nHaig Glacier (June-September 2022-2024)', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "precipitation_albedo_scatter.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Precipitation scatter plots saved")
    
    def create_correlation_matrix_heatmap(self, data: pd.DataFrame, stats: Dict[str, Any]) -> None:
        """Create correlation matrix heatmap for all five variables."""
        logger.info("Creating 5x5 correlation matrix heatmap...")
        
        # Create correlation matrix for all 5 variables
        corr_data = data[['bc_aod_regional', 'temperature_c', 'total_precip_mm', 'snowfall_mm', 'rainfall_mm', 'albedo_mean']].corr()
        
        # Create custom labels
        labels = ['BC AOD\n(Regional)', 'Air Temperature\n(°C)', 'Total Precip\n(mm)', 'Snowfall\n(mm)', 'Rainfall\n(mm)', 'MODIS Albedo\n(Daily Mean)']
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr_data, dtype=bool))  # Mask upper triangle
        im = ax.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal')
        
        # Set ticks and labels
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_yticklabels(labels, fontsize=10)
        
        # Rotate x-axis labels
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Add correlation values as text
        for i in range(len(labels)):
            for j in range(len(labels)):
                if i != j:  # Don't show text for diagonal
                    corr_val = corr_data.iloc[i, j]
                    
                    # Determine text color based on correlation strength
                    text_color = 'white' if abs(corr_val) > 0.5 else 'black'
                    
                    # Show correlation value
                    ax.text(j, i, f'{corr_val:.3f}', 
                           ha='center', va='center', color=text_color, fontsize=9, fontweight='bold')
                else:
                    ax.text(j, i, '1.000', ha='center', va='center', color='black', fontsize=9, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Pearson Correlation Coefficient', fontsize=12)
        
        # Title and formatting
        ax.set_title('Correlation Matrix: BC AOD, Temperature, Precipitation, and Albedo\nHaig Glacier (June-September 2022-2024)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        # Add sample size information
        ax.text(0.5, -0.15, f'n = {stats["sample_size"]} observations', 
               ha='center', va='center', transform=ax.transAxes, fontsize=11)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "correlation_matrix_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Correlation matrix heatmap saved")
    
    def create_multi_panel_scatter_grid(self, data: pd.DataFrame, stats: Dict[str, Any]) -> None:
        """Create multi-panel scatter plot grid for all pairwise relationships."""
        logger.info("Creating multi-panel scatter plot grid...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Pairwise Relationships: BC AOD, Temperature, and Albedo\nHaig Glacier (June-September 2022-2024)', 
                    fontsize=16, fontweight='bold')
        
        # BC AOD vs Albedo (colored by temperature)
        ax1 = axes[0, 0]
        scatter1 = ax1.scatter(data['bc_aod_regional'], data['albedo_mean'], 
                              c=data['temperature_c'], s=40, alpha=0.7, 
                              cmap='coolwarm', edgecolors='black', linewidth=0.3)
        z1 = np.polyfit(data['bc_aod_regional'], data['albedo_mean'], 1)
        p1 = np.poly1d(z1)
        x_reg1 = np.linspace(data['bc_aod_regional'].min(), data['bc_aod_regional'].max(), 100)
        ax1.plot(x_reg1, p1(x_reg1), "darkblue", linestyle='--', alpha=0.8, linewidth=2)
        ax1.set_xlabel('BC AOD (Regional)', fontsize=11)
        ax1.set_ylabel('MODIS Albedo', fontsize=11)
        ax1.set_title(f'BC AOD vs Albedo\nr = {stats["bc_albedo_pearson_r"]:.3f}, p = {stats["bc_albedo_pearson_p"]:.4f}', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Temperature vs Albedo (colored by BC AOD)
        ax2 = axes[0, 1]
        scatter2 = ax2.scatter(data['temperature_c'], data['albedo_mean'], 
                              c=data['bc_aod_regional'], s=40, alpha=0.7, 
                              cmap='viridis', edgecolors='black', linewidth=0.3)
        z2 = np.polyfit(data['temperature_c'], data['albedo_mean'], 1)
        p2 = np.poly1d(z2)
        x_reg2 = np.linspace(data['temperature_c'].min(), data['temperature_c'].max(), 100)
        ax2.plot(x_reg2, p2(x_reg2), "darkred", linestyle='--', alpha=0.8, linewidth=2)
        ax2.set_xlabel('Air Temperature (°C)', fontsize=11)
        ax2.set_ylabel('MODIS Albedo', fontsize=11)
        ax2.set_title(f'Temperature vs Albedo\nr = {stats["temp_albedo_pearson_r"]:.3f}, p = {stats["temp_albedo_pearson_p"]:.4f}', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # BC AOD vs Temperature (colored by albedo)
        ax3 = axes[1, 0]
        scatter3 = ax3.scatter(data['bc_aod_regional'], data['temperature_c'], 
                              c=data['albedo_mean'], s=40, alpha=0.7, 
                              cmap='plasma', edgecolors='black', linewidth=0.3)
        z3 = np.polyfit(data['bc_aod_regional'], data['temperature_c'], 1)
        p3 = np.poly1d(z3)
        x_reg3 = np.linspace(data['bc_aod_regional'].min(), data['bc_aod_regional'].max(), 100)
        ax3.plot(x_reg3, p3(x_reg3), "darkgreen", linestyle='--', alpha=0.8, linewidth=2)
        ax3.set_xlabel('BC AOD (Regional)', fontsize=11)
        ax3.set_ylabel('Air Temperature (°C)', fontsize=11)
        ax3.set_title(f'BC AOD vs Temperature\nr = {stats["bc_temp_pearson_r"]:.3f}, p = {stats["bc_temp_pearson_p"]:.4f}', fontsize=12)
        ax3.grid(True, alpha=0.3)
        
        # Multiple regression visualization (predicted vs observed)
        ax4 = axes[1, 1]
        
        # Calculate predicted values using multiple regression (numpy implementation)
        X_reg = np.column_stack([np.ones(len(data)), data['bc_aod_regional'], data['temperature_c']])
        y_reg = data['albedo_mean'].values
        try:
            beta_reg = np.linalg.solve(X_reg.T @ X_reg, X_reg.T @ y_reg)
            y_pred = X_reg @ beta_reg
        except np.linalg.LinAlgError:
            y_pred = np.full_like(y_reg, np.mean(y_reg))
        
        scatter4 = ax4.scatter(data['albedo_mean'], y_pred, 
                              c=data['temperature_c'], s=40, alpha=0.7, 
                              cmap='coolwarm', edgecolors='black', linewidth=0.3)
        
        # Add 1:1 line
        min_val = min(data['albedo_mean'].min(), y_pred.min())
        max_val = max(data['albedo_mean'].max(), y_pred.max())
        ax4.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=2, label='1:1 Line')
        
        ax4.set_xlabel('Observed Albedo', fontsize=11)
        ax4.set_ylabel('Predicted Albedo\n(BC AOD + Temperature)', fontsize=11)
        ax4.set_title(f'Multiple Regression Model\nR² = {stats["multiple_r_squared"]:.3f}, Adj R² = {stats["adjusted_r_squared"]:.3f}', fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add colorbars
        cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.set_label('Temperature (°C)', fontsize=10)
        
        cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.set_label('BC AOD', fontsize=10)
        
        cbar3 = fig.colorbar(scatter3, ax=ax3, shrink=0.8)
        cbar3.set_label('Albedo', fontsize=10)
        
        cbar4 = fig.colorbar(scatter4, ax=ax4, shrink=0.8)
        cbar4.set_label('Temperature (°C)', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "multi_panel_scatter_grid.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Multi-panel scatter grid saved")
    
    def create_time_series_plot(self, data: pd.DataFrame) -> None:
        """Create dual-axis time series plot."""
        logger.info("Creating time series plot...")
        
        # Sort data by date to ensure proper line plotting
        data_sorted = data.sort_values('date').reset_index(drop=True)
        
        fig, ax1 = plt.subplots(figsize=(14, 8))
        
        # Plot BC AOD on left axis (line only, no markers)
        color = 'tab:red'
        ax1.set_xlabel('Date', fontsize=12)
        ax1.set_ylabel('Black Carbon AOD (Regional)', color=color, fontsize=12)
        line1 = ax1.plot(data_sorted['date'], data_sorted['bc_aod_regional'], 
                        color=color, linewidth=2, alpha=0.8, label='BC AOD (Regional)')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, alpha=0.3)
        
        # Create second y-axis for albedo
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('MODIS Albedo (Daily Mean)', color=color, fontsize=12)
        line2 = ax2.plot(data_sorted['date'], data_sorted['albedo_mean'], 
                        color=color, linewidth=2, alpha=0.8, label='MODIS Albedo (Mean)')
        ax2.tick_params(axis='y', labelcolor=color)
        
        # Title and formatting
        plt.title('Time Series: Black Carbon AOD vs Surface Albedo\nHaig Glacier (June-September 2022-2024)', 
                 fontsize=14, fontweight='bold', pad=20)
        
        # Legend
        lines = line1 + line2
        labels = ['BC AOD (Regional)', 'MODIS Albedo (Mean)']
        ax1.legend(lines, labels, loc='upper right')
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "bc_aod_albedo_timeseries.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Time series plot saved")
    
    def create_precipitation_time_series_plot(self, data: pd.DataFrame) -> None:
        """Create precipitation time series plot with albedo."""
        logger.info("Creating precipitation time series plot...")
        
        # Sort data by date to ensure proper line plotting
        data_sorted = data.sort_values('date').reset_index(drop=True)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
        
        # Top panel: Precipitation components
        ax1.plot(data_sorted['date'], data_sorted['total_precip_mm'], 
                color='darkblue', linewidth=2, alpha=0.8, label='Total Precipitation')
        ax1.plot(data_sorted['date'], data_sorted['snowfall_mm'], 
                color='lightblue', linewidth=2, alpha=0.8, label='Snowfall')
        ax1.plot(data_sorted['date'], data_sorted['rainfall_mm'], 
                color='orange', linewidth=2, alpha=0.8, label='Rainfall')
        
        ax1.set_ylabel('Precipitation (mm)', fontsize=12)
        ax1.set_title('Precipitation Components vs Surface Albedo\nHaig Glacier (June-September 2022-2024)', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Bottom panel: Albedo
        ax2.plot(data_sorted['date'], data_sorted['albedo_mean'], 
                color='green', linewidth=2, alpha=0.8, label='MODIS Albedo (Mean)')
        
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('MODIS Albedo (Daily Mean)', fontsize=12)
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "precipitation_albedo_timeseries.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Precipitation time series plot saved")
    
    def create_yearly_time_series_plots(self, data: pd.DataFrame) -> None:
        """Create 3-panel yearly time series plots with BC AOD, Albedo, and Temperature."""
        logger.info("Creating yearly time series plots with temperature...")
        
        # Sort data by date to ensure proper line plotting
        data_sorted = data.sort_values('date').reset_index(drop=True)
        
        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        fig.suptitle('Yearly Time Series: Black Carbon AOD, Surface Albedo, and Temperature\nHaig Glacier (June-September)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        years = [2022, 2023, 2024]
        # Use consistent colors for all years (same as 2023 previously had)
        colors_bc = ['red', 'red', 'red']
        colors_albedo = ['blue', 'blue', 'blue']
        colors_temp = ['green', 'green', 'green']
        
        for i, year in enumerate(years):
            ax1 = axes[i]
            
            # Filter data for specific year
            year_data = data_sorted[data_sorted['year'] == year].copy()
            
            if len(year_data) > 0:
                # Plot BC AOD on left axis
                color_bc = colors_bc[i]
                ax1.set_ylabel('Black Carbon AOD (Regional)', color=color_bc, fontsize=11)
                line1 = ax1.plot(year_data['date'], year_data['bc_aod_regional'], 
                               color=color_bc, linewidth=2, alpha=0.8, 
                               label=f'BC AOD {year}')
                ax1.tick_params(axis='y', labelcolor=color_bc)
                ax1.grid(True, alpha=0.3)
                
                # Create second y-axis for albedo
                ax2 = ax1.twinx()
                color_albedo = colors_albedo[i]
                ax2.set_ylabel('MODIS Albedo (Daily Mean)', color=color_albedo, fontsize=11)
                line2 = ax2.plot(year_data['date'], year_data['albedo_mean'], 
                               color=color_albedo, linewidth=2, alpha=0.8, 
                               label=f'Albedo {year}')
                ax2.tick_params(axis='y', labelcolor=color_albedo)
                
                # Create third y-axis for temperature
                ax3 = ax1.twinx()
                # Offset the third axis
                ax3.spines['right'].set_position(('outward', 60))
                color_temp = colors_temp[i]
                ax3.set_ylabel('Air Temperature (°C)', color=color_temp, fontsize=11)
                line3 = ax3.plot(year_data['date'], year_data['temperature_c'], 
                               color=color_temp, linewidth=2, alpha=0.8, linestyle=':', 
                               label=f'Temperature {year}')
                ax3.tick_params(axis='y', labelcolor=color_temp)
                
                # Set title for each subplot with correlations
                bc_albedo_r = np.nan
                temp_albedo_r = np.nan
                bc_temp_r = np.nan
                
                if len(year_data) > 3:  # Need at least 4 points for correlation
                    try:
                        bc_albedo_r, bc_albedo_p = pearsonr(year_data['bc_aod_regional'], year_data['albedo_mean'])
                        temp_albedo_r, temp_albedo_p = pearsonr(year_data['temperature_c'], year_data['albedo_mean'])
                        bc_temp_r, bc_temp_p = pearsonr(year_data['bc_aod_regional'], year_data['temperature_c'])
                    except:
                        pass
                
                title_text = f'{year} (n = {len(year_data)} observations)\n'
                title_text += f'BC-Albedo: r = {bc_albedo_r:.3f} | Temp-Albedo: r = {temp_albedo_r:.3f} | BC-Temp: r = {bc_temp_r:.3f}'
                ax1.set_title(title_text, fontsize=11, pad=15)
                
                # Enhanced legend with all three variables
                lines = line1 + line2 + line3
                labels = [f'BC AOD {year}', f'Albedo {year}', f'Temperature {year}']
                ax1.legend(lines, labels, loc='upper left', fontsize=9, 
                          bbox_to_anchor=(0.02, 0.98))
                
                # Format x-axis - only show month-day for cleaner look
                ax1.tick_params(axis='x', rotation=45)
                
                # Add statistics box
                if len(year_data) > 3:
                    stats_text = f'Variable means:\n'
                    stats_text += f'BC AOD: {year_data["bc_aod_regional"].mean():.4f}\n'
                    stats_text += f'Albedo: {year_data["albedo_mean"].mean():.3f}\n'
                    stats_text += f'Temp: {year_data["temperature_c"].mean():.1f}°C'
                    
                    ax1.text(0.98, 0.02, stats_text, transform=ax1.transAxes, 
                           verticalalignment='bottom', horizontalalignment='right', fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))
                
            else:
                ax1.text(0.5, 0.5, f'{year}\nNo data available', 
                        ha='center', va='center', transform=ax1.transAxes, fontsize=12)
                ax1.set_ylabel('Black Carbon AOD (Regional)', fontsize=11)
                ax1.set_title(f'{year} (no data)', fontsize=12, pad=10)
        
        # Only label x-axis on bottom plot
        axes[-1].set_xlabel('Date', fontsize=12)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)  # Make room for main title
        plt.savefig(self.output_dir / "plots" / "bc_aod_albedo_temp_yearly_timeseries.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Enhanced yearly time series plots with temperature saved")
    
    def create_seasonal_analysis(self, data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Perform seasonal correlation analysis."""
        logger.info("Creating seasonal analysis...")
        
        seasonal_stats = {}
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Seasonal Analysis: BC AOD vs Albedo\nHaig Glacier (2022-2024)', 
                    fontsize=16, fontweight='bold')
        
        months = [6, 7, 8, 9]
        month_names = ['June', 'July', 'August', 'September']
        colors = ['skyblue', 'lightgreen', 'orange', 'lightcoral']
        
        for i, (month, month_name, color) in enumerate(zip(months, month_names, colors)):
            ax = axes[i//2, i%2]
            
            # Filter data for specific month
            month_data = data[data['month'] == month]
            
            if len(month_data) > 3:  # Need at least 4 points for correlation
                # Calculate correlation
                pearson_r, pearson_p = pearsonr(month_data['bc_aod_regional'], month_data['albedo_mean'])
                r_squared = pearson_r ** 2
                
                # Calculate precipitation correlations for this month
                temp_pearson_r, temp_pearson_p = pearsonr(month_data['temperature_c'], month_data['albedo_mean'])
                total_precip_pearson_r, total_precip_pearson_p = pearsonr(month_data['total_precip_mm'], month_data['albedo_mean'])
                snow_pearson_r, snow_pearson_p = pearsonr(month_data['snowfall_mm'], month_data['albedo_mean'])
                rain_pearson_r, rain_pearson_p = pearsonr(month_data['rainfall_mm'], month_data['albedo_mean'])
                
                seasonal_stats[month_name] = {
                    'sample_size': len(month_data),
                    'pearson_r': pearson_r,
                    'pearson_p': pearson_p,
                    'r_squared': r_squared,
                    'bc_aod_mean': month_data['bc_aod_regional'].mean(),
                    'temperature_mean': month_data['temperature_c'].mean(),
                    'total_precip_mean': month_data['total_precip_mm'].mean(),
                    'snowfall_mean': month_data['snowfall_mm'].mean(),
                    'rainfall_mean': month_data['rainfall_mm'].mean(),
                    'albedo_mean': month_data['albedo_mean'].mean(),
                    'temp_albedo_pearson_r': temp_pearson_r,
                    'temp_albedo_pearson_p': temp_pearson_p,
                    'total_precip_albedo_pearson_r': total_precip_pearson_r,
                    'total_precip_albedo_pearson_p': total_precip_pearson_p,
                    'snow_albedo_pearson_r': snow_pearson_r,
                    'snow_albedo_pearson_p': snow_pearson_p,
                    'rain_albedo_pearson_r': rain_pearson_r,
                    'rain_albedo_pearson_p': rain_pearson_p
                }
                
                # Create subplot
                ax.scatter(month_data['bc_aod_regional'], month_data['albedo_mean'], 
                          alpha=0.7, color=color, s=60)
                
                # Add regression line if significant correlation
                if len(month_data) > 1:
                    z = np.polyfit(month_data['bc_aod_regional'], month_data['albedo_mean'], 1)
                    p = np.poly1d(z)
                    x_reg = np.linspace(month_data['bc_aod_regional'].min(), 
                                      month_data['bc_aod_regional'].max(), 100)
                    ax.plot(x_reg, p(x_reg), "r--", alpha=0.8)
                
                # Formatting
                ax.set_title(f'{month_name}\nr = {pearson_r:.3f}, p = {pearson_p:.3f}, n = {len(month_data)}')
                ax.set_xlabel('BC AOD (Regional)')
                ax.set_ylabel('MODIS Albedo (Mean)')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'{month_name}\nInsufficient data\n(n = {len(month_data)})', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_xlabel('BC AOD (Regional)')
                ax.set_ylabel('MODIS Albedo (Mean)')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "bc_aod_albedo_seasonal.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Seasonal analysis completed")
        return seasonal_stats
    
    def create_residual_analysis(self, data: pd.DataFrame) -> None:
        """Create residual analysis plot."""
        logger.info("Creating residual analysis...")
        
        # Calculate fitted values and residuals
        z = np.polyfit(data['bc_aod_regional'], data['albedo_mean'], 1)
        fitted_values = np.polyval(z, data['bc_aod_regional'])
        residuals = data['albedo_mean'] - fitted_values
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Residual Analysis: BC AOD vs Albedo Relationship', fontsize=16, fontweight='bold')
        
        # Residuals vs fitted values
        ax1.scatter(fitted_values, residuals, alpha=0.6)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Fitted Values')
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Residuals')
        ax2.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax3.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax3.axvline(x=0, color='r', linestyle='--')
        ax3.set_xlabel('Residuals')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Residuals')
        ax3.grid(True, alpha=0.3)
        
        # Residuals vs BC AOD
        ax4.scatter(data['bc_aod_regional'], residuals, alpha=0.6)
        ax4.axhline(y=0, color='r', linestyle='--')
        ax4.set_xlabel('BC AOD (Regional)')
        ax4.set_ylabel('Residuals')
        ax4.set_title('Residuals vs BC AOD')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "bc_aod_albedo_residuals.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Residual analysis completed")
    
    def save_results(self, data: pd.DataFrame, stats: Dict[str, Any], seasonal_stats: Dict[str, Dict[str, Any]]) -> None:
        """Save analysis results to CSV files."""
        logger.info("Saving results...")
        
        # Save merged dataset
        data.to_csv(self.output_dir / "results" / "bc_aod_albedo_merged_data.csv", index=False)
        
        # Save overall statistics
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(self.output_dir / "results" / "bc_aod_albedo_correlation_stats.csv", index=False)
        
        # Save seasonal statistics
        seasonal_df = pd.DataFrame(seasonal_stats).T
        seasonal_df.index.name = 'month'
        seasonal_df.to_csv(self.output_dir / "results" / "bc_aod_albedo_seasonal_stats.csv")
        
        # Create summary report
        with open(self.output_dir / "results" / "analysis_summary.txt", 'w') as f:
            f.write("Black Carbon AOD vs MODIS Albedo Correlation Analysis\n")
            f.write("=" * 55 + "\n\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Dataset Information:\n")
            f.write(f"- Total paired observations: {stats['sample_size']}\n")
            f.write(f"- Time period: June-September 2022-2024\n")
            f.write(f"- Location: Haig Glacier\n\n")
            f.write(f"Overall Correlation Results:\n")
            f.write(f"- Pearson correlation: {stats['pearson_r']:.4f} (p = {stats['pearson_p']:.4f})\n")
            f.write(f"- Spearman correlation: {stats['spearman_r']:.4f} (p = {stats['spearman_p']:.4f})\n")
            f.write(f"- R-squared: {stats['r_squared']:.4f}\n")
            f.write(f"- 95% CI for Pearson r: [{stats['pearson_ci_lower']:.4f}, {stats['pearson_ci_upper']:.4f}]\n\n")
            f.write(f"Variable Statistics:\n")
            f.write(f"- BC AOD (Regional): {stats['bc_aod_mean']:.4f} ± {stats['bc_aod_std']:.4f}\n")
            f.write(f"- MODIS Albedo: {stats['albedo_mean']:.4f} ± {stats['albedo_std']:.4f}\n\n")
            f.write(f"Seasonal Results:\n")
            for month, month_stats in seasonal_stats.items():
                f.write(f"- {month}: r = {month_stats['pearson_r']:.3f} (p = {month_stats['pearson_p']:.3f}, n = {month_stats['sample_size']})\n")
        
        logger.info("Results saved successfully")
    
    def create_comparison_analysis(self, data_original: pd.DataFrame, data_filtered: pd.DataFrame, 
                                  stats_original: Dict[str, Any], stats_filtered: Dict[str, Any]) -> None:
        """Create comparison plots and analysis between original and filtered data."""
        logger.info("Creating comparison analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Comparison: Original vs Outlier-Filtered Data\nHaig Glacier BC AOD vs Albedo', 
                    fontsize=16, fontweight='bold')
        
        # Define colors for months
        month_colors = {6: 'lightgreen', 7: 'orange', 8: 'red', 9: 'brown'}
        month_names = {6: 'June', 7: 'July', 8: 'August', 9: 'September'}
        
        datasets = [data_original, data_filtered]
        stats_list = [stats_original, stats_filtered]
        titles = ['Original Data', 'Outlier-Filtered Data (2.5 SD)']
        
        for i, (data, stats, title) in enumerate(zip(datasets, stats_list, titles)):
            ax = axes[i]
            
            # Create scatter plot with discrete colors by month
            for month in [6, 7, 8, 9]:
                month_data = data[data['month'] == month]
                if len(month_data) > 0:
                    ax.scatter(month_data['bc_aod_regional'], month_data['albedo_mean'], 
                              alpha=0.7, c=month_colors[month], s=60, 
                              label=f'{month_names[month]} (n={len(month_data)})',
                              edgecolors='black', linewidth=0.5)
            
            # Add regression line
            z = np.polyfit(data['bc_aod_regional'], data['albedo_mean'], 1)
            p = np.poly1d(z)
            x_reg = np.linspace(data['bc_aod_regional'].min(), data['bc_aod_regional'].max(), 100)
            ax.plot(x_reg, p(x_reg), "darkblue", linestyle='--', alpha=0.8, linewidth=2)
            
            # Formatting
            ax.set_xlabel('Black Carbon AOD (Regional)', fontsize=12)
            ax.set_ylabel('MODIS Albedo (Daily Mean)', fontsize=12)
            ax.set_title(f'{title}\nr = {stats["pearson_r"]:.4f}, R² = {stats["r_squared"]:.4f}, n = {stats["sample_size"]}', 
                        fontsize=13, pad=10)
            
            # Add statistics text
            stats_text = f'p = {stats["pearson_p"]:.4f}'
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add legend only to first plot
            if i == 0:
                ax.legend(loc='upper right', fontsize=9)
            
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "bc_aod_albedo_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Comparison analysis completed")
    
    def run_complete_analysis(self) -> None:
        """Run the complete correlation analysis using original data only."""
        logger.info("Starting complete BC AOD vs Albedo correlation analysis...")
        
        # Load and process data
        data = self.load_and_process_data()
        
        # Calculate correlations
        stats = self.calculate_correlations(data)
        
        # Create visualizations
        self.create_scatter_plot(data, stats)  # Enhanced with temperature coloring
        self.create_precipitation_scatter_plots(data, stats)  # New: precipitation scatter plots
        self.create_correlation_matrix_heatmap(data, stats)  # Updated: 5x5 correlation matrix
        self.create_precipitation_time_series_plot(data)  # New: precipitation time series
        self.create_multi_panel_scatter_grid(data, stats)  # Multi-panel grid
        self.create_yearly_time_series_plots(data)
        seasonal_stats = self.create_seasonal_analysis(data)
        self.create_residual_analysis(data)
        
        # Save results
        self.save_results(data, stats, seasonal_stats)
        
        logger.info(f"Analysis complete! Results saved to: {self.output_dir}")
        print(f"\nEnhanced 5-Variable Correlation Analysis Results:")
        print(f"================================================")
        print(f"Sample size: {stats['sample_size']} paired observations with BC AOD, Temperature, Precipitation (3 vars), and Albedo")
        print(f"\nPairwise Correlations with Albedo:")
        print(f"- BC AOD vs Albedo:         r = {stats['bc_albedo_pearson_r']:.4f} (p = {stats['bc_albedo_pearson_p']:.4f})")
        print(f"- Temperature vs Albedo:    r = {stats['temp_albedo_pearson_r']:.4f} (p = {stats['temp_albedo_pearson_p']:.4f})")
        print(f"- Total Precip vs Albedo:   r = {stats['total_precip_albedo_pearson_r']:.4f} (p = {stats['total_precip_albedo_pearson_p']:.4f})")
        print(f"- Snowfall vs Albedo:       r = {stats['snow_albedo_pearson_r']:.4f} (p = {stats['snow_albedo_pearson_p']:.4f})")
        print(f"- Rainfall vs Albedo:       r = {stats['rain_albedo_pearson_r']:.4f} (p = {stats['rain_albedo_pearson_p']:.4f})")
        print(f"\nMultiple Regression Model (Albedo ~ BC AOD + Temperature + Total Precip + Snowfall):")
        print(f"- Multiple R²: {stats['multiple_r_squared']:.4f}")
        print(f"- Adjusted R²: {stats['adjusted_r_squared']:.4f}")
        print(f"- F-statistic: {stats['f_statistic']:.2f}")
        print(f"\nUnstandardized Coefficients:")
        print(f"- BC AOD coefficient:       {stats['bc_aod_coefficient']:.4f}")
        print(f"- Temperature coefficient:  {stats['temperature_coefficient']:.4f}")
        print(f"- Total Precip coefficient: {stats['total_precip_coefficient']:.4f}")
        print(f"- Snowfall coefficient:     {stats['snowfall_coefficient']:.4f}")
        print(f"\nStandardized Coefficients (Beta):")
        print(f"- BC AOD beta:         {stats['bc_aod_beta_standardized']:.4f}")
        print(f"- Temperature beta:    {stats['temperature_beta_standardized']:.4f}")
        print(f"- Total Precip beta:   {stats['total_precip_beta_standardized']:.4f}")
        print(f"- Snowfall beta:       {stats['snowfall_beta_standardized']:.4f}")
        print(f"\nVariable Statistics:")
        print(f"- BC AOD: {stats['bc_aod_mean']:.4f} ± {stats['bc_aod_std']:.4f}")
        print(f"- Temperature: {stats['temperature_mean']:.1f}°C ± {stats['temperature_std']:.1f}°C")
        print(f"- Total Precip: {stats['total_precip_mean']:.2f} ± {stats['total_precip_std']:.2f} mm")
        print(f"- Snowfall: {stats['snowfall_mean']:.2f} ± {stats['snowfall_std']:.2f} mm")
        print(f"- Rainfall: {stats['rainfall_mean']:.2f} ± {stats['rainfall_std']:.2f} mm")
        print(f"- Albedo: {stats['albedo_mean']:.4f} ± {stats['albedo_std']:.4f}")
        print(f"\nResults saved to: {self.output_dir}")


def main():
    """Main function to run the analysis."""
    # File paths
    merra2_path = r"D:\Downloads\Haig_Glacier_Climate_JuneSept_Daily_MERRA2_Speciated_AOD_10km_2 - Haig_Glacier_Climate_JuneSept_Daily_MERRA2_Speciated_AOD_10km_2.csv"
    modis_path = r"D:\Downloads\Haig_2022-01-01_to_2025-01-01.csv"
    
    # Initialize and run analysis
    analyzer = BCAlbedoCorrelationAnalyzer(merra2_path, modis_path)
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()