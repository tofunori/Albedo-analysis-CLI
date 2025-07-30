#!/usr/bin/env python3
"""
Pivot-based Main Analysis Script

This script implements the user's proven methodology from compare_albedo.ipynb that successfully 
produces 515 merged observations with proper statistical validation.

Key methodology differences from the original framework:
1. Uses pivot_table for data reshaping instead of complex temporal alignment
2. Simple pd.merge for temporal alignment instead of method-specific pairs
3. Residual-based outlier detection instead of Z-score
4. Processes the same CSV files that produce correct results
5. Replicates exact plotting and analysis style
"""

import argparse
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats as scipy_stats
from typing import Dict, Any, List, Tuple
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.helpers import load_config, setup_logging, ensure_directory_exists, get_timestamp
from src.data.pivot_based_loaders import (
    create_pivot_based_loader, 
    PivotBasedProcessor,
    AthabascaMultiProductLoader,
    AthabascaAWSLoader
)


class PivotBasedAlbedoAnalysis:
    """
    Main analysis class that replicates the user's proven methodology.
    
    This produces the same results as compare_albedo.ipynb with 515 merged observations.
    """
    
    def __init__(self, config_path: str):
        """Initialize with configuration."""
        self.config = load_config(config_path)
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize processor
        self.processor = PivotBasedProcessor(self.config)
        
        # Set up output directories
        self._setup_output_directories()
        
    def _setup_output_directories(self):
        """Create necessary output directories."""
        output_paths = [
            self.config['output']['results_path'],
            self.config['output']['plots_path'],
            self.config['output']['maps_path']
        ]
        
        for path in output_paths:
            ensure_directory_exists(path)
    
    def process_glacier(self, glacier_id: str, use_selected_pixels: bool = False) -> Dict[str, Any]:
        """Process a single glacier using the pivot-based methodology."""
        self.logger.info(f"Starting pivot-based analysis for glacier: {glacier_id}")
        
        try:
            # Load glacier configuration
            glacier_config = self._load_glacier_config(glacier_id)
            
            # Create glacier-specific output directories
            glacier_output_dir = self._create_glacier_output_dir(glacier_id)
            
            # Step 1: Load data using pivot-based loaders
            self.logger.info("Loading data using pivot-based approach...")
            modis_data, aws_data = self._load_pivot_based_data(glacier_config, glacier_id, use_selected_pixels)
            
            # Step 2: Apply Terra/Aqua merging (matches user's approach)
            self.logger.info("Applying Terra/Aqua merge...")
            modis_merged = self.processor.apply_terra_aqua_merge(modis_data)
            
            # Step 3: Create pivot table and merge with AWS (core methodology)
            self.logger.info("Creating pivot table and merging with AWS...")
            merged_data = self.processor.create_pivot_and_merge(modis_merged, aws_data)
            
            # Step 4: Statistical analysis (matches user's approach)
            self.logger.info("Performing statistical analysis...")
            statistics = self._perform_pivot_based_analysis(merged_data)
            
            # Step 4b: Outlier analysis (residual-based)
            self.logger.info("Performing outlier analysis...")
            outlier_stats = self._perform_outlier_analysis(merged_data)
            statistics['outlier_analysis'] = outlier_stats
            
            # Step 5: Generate visualizations (comprehensive suite)
            self.logger.info("Generating comprehensive visualization suite...")
            self._generate_comprehensive_plots(merged_data, statistics, glacier_id, glacier_output_dir)
            
            # Step 6: Export results
            self.logger.info("Exporting results...")
            results = self._export_pivot_based_results(statistics, merged_data, glacier_id, glacier_output_dir)
            
            self.logger.info(f"Pivot-based analysis completed for glacier: {glacier_id}")
            self.logger.info(f"Successfully processed {len(merged_data)} merged observations")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing glacier {glacier_id}: {e}")
            raise
    
    def _load_glacier_config(self, glacier_id: str) -> Dict[str, Any]:
        """Load configuration for specific glacier."""
        glacier_sites_config = load_config('config/glacier_sites.yaml')
        
        if glacier_id not in glacier_sites_config['glaciers']:
            raise ValueError(f"Glacier {glacier_id} not found in configuration")
        
        return glacier_sites_config['glaciers'][glacier_id]
    
    def _create_glacier_output_dir(self, glacier_id: str) -> str:
        """Create output directory for specific glacier."""
        timestamp = get_timestamp()
        glacier_dir = os.path.join(self.config['output']['base_path'], f"{glacier_id}_pivot_{timestamp}")
        
        subdirs = ['results', 'plots', 'maps']
        for subdir in subdirs:
            ensure_directory_exists(os.path.join(glacier_dir, subdir))
        
        return glacier_dir
    
    def _load_pivot_based_data(self, glacier_config: Dict[str, Any], glacier_id: str, use_selected_pixels: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load data using pivot-based loaders."""
        
        # Determine search paths based on glacier
        glacier_name = glacier_config['name'].lower().split()[0].lower()
        search_paths = [self.config['data']['modis_path']]
        
        # Add glacier-specific paths
        if glacier_name == 'athabasca' and 'athabasca_modis_path' in self.config['data']:
            search_paths.insert(0, self.config['data']['athabasca_modis_path'])  # Search Athabasca path first
        elif glacier_name == 'haig' and 'haig_modis_path' in self.config['data']:
            search_paths.insert(0, self.config['data']['haig_modis_path'])  # Search Haig path first
        elif glacier_name == 'coropuna' and 'coropuna_modis_path' in self.config['data']:
            search_paths.insert(0, self.config['data']['coropuna_modis_path'])  # Search Coropuna path first
        
        multiproduct_file = None
        
        # Search for MultiProduct file in all paths
        for search_path in search_paths:
            if not os.path.exists(search_path):
                continue
                
            for filename in os.listdir(search_path):
                # Look for MultiProduct files (with or without AWS) or Coropuna glacier files
                if (('MultiProduct' in filename and 
                     (glacier_name in filename.lower() or 
                      any(keyword in filename.lower() for keyword in ['multiproduct', 'multi_product']))) or
                    (glacier_name == 'coropuna' and 'coropuna_glacier' in filename.lower() and filename.endswith('.csv'))):
                    multiproduct_file = os.path.join(search_path, filename)
                    self.logger.info(f"Found MultiProduct file: {multiproduct_file}")
                    break
            
            if multiproduct_file:
                break
        
        if not multiproduct_file:
            raise FileNotFoundError(f"MultiProduct file not found for {glacier_config['name']} in paths: {search_paths}")
        
        # Load MODIS data
        modis_loader = create_pivot_based_loader("athabasca_multiproduct", self.config, glacier_config)
        modis_data = modis_loader.load_data(multiproduct_file)
        
        # Store original MODIS data for spatial mapping (before pixel selection)
        self._last_modis_data_original = modis_data.copy()
        
        # Apply pixel selection if requested
        if use_selected_pixels:
            self.logger.info(f"Applying pixel selection for {glacier_name}...")
            modis_data = self._apply_pixel_selection(modis_data, glacier_id, glacier_config)
            # Store selected pixels for spatial mapping
            self._last_modis_data_selected = modis_data.copy()
            self._use_selected_pixels = True
        else:
            self._last_modis_data_selected = modis_data.copy()
            self._use_selected_pixels = False
        
        # Load AWS data (use separate file to match user's methodology)
        aws_loader = create_pivot_based_loader("AWS", self.config, None)  # Don't pass glacier_config to avoid confusion
        
        # Use separate AWS file (matching user's approach)
        if 'aws' in glacier_config.get('data_files', {}):
            # Determine AWS search paths based on glacier
            aws_search_paths = [self.config['data']['aws_path']]
            if glacier_name == 'athabasca' and 'athabasca_aws_path' in self.config['data']:
                aws_search_paths.insert(0, self.config['data']['athabasca_aws_path'])  # Search Athabasca path first
            elif glacier_name == 'haig' and 'haig_aws_path' in self.config['data']:
                aws_search_paths.insert(0, self.config['data']['haig_aws_path'])  # Search Haig path first
            elif glacier_name == 'coropuna' and 'coropuna_aws_path' in self.config['data']:
                aws_search_paths.insert(0, self.config['data']['coropuna_aws_path'])  # Search Coropuna path first
            
            aws_file_path = None
            aws_filename = glacier_config['data_files']['aws']
            
            # Search for AWS file in all paths
            for aws_path in aws_search_paths:
                potential_path = os.path.join(aws_path, aws_filename)
                if os.path.exists(potential_path):
                    aws_file_path = potential_path
                    self.logger.info(f"Found AWS file: {aws_file_path}")
                    break
            
            if aws_file_path:
                aws_data = aws_loader.load_data(aws_file_path)
            else:
                raise FileNotFoundError(f"AWS file '{aws_filename}' not found in paths: {aws_search_paths}")
        else:
            self.logger.warning("No separate AWS file specified, trying integrated AWS")
            aws_data = aws_loader.load_data(multiproduct_file)
        
        self.logger.info(f"Loaded MODIS data: {len(modis_data)} records")
        self.logger.info(f"Loaded AWS data: {len(aws_data)} records")
        
        return modis_data, aws_data
    
    def _apply_pixel_selection(self, modis_data: pd.DataFrame, glacier_id: str, glacier_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply pixel selection algorithm to find 2 closest best performing pixels to AWS station."""
        try:
            import numpy as np
            
            # Get AWS station coordinates
            aws_stations = glacier_config.get('aws_stations', {})
            if not aws_stations:
                self.logger.warning(f"No AWS stations found for {glacier_id}, using standard selection")
                return self._apply_standard_pixel_selection(modis_data, glacier_id)
            
            # Use first AWS station coordinates
            aws_station = list(aws_stations.values())[0]
            aws_lat = aws_station['lat']
            aws_lon = aws_station['lon']
            
            # Get available pixels with their quality metrics and coordinates
            pixel_summary = modis_data.groupby('pixel_id').agg({
                'glacier_fraction': 'mean',
                'albedo': 'count',  # Number of observations per pixel
                'ndsi': 'mean',
                'latitude': 'first',  # Coordinates should be the same for each pixel
                'longitude': 'first'
            }).reset_index()
            
            pixel_summary.columns = ['pixel_id', 'avg_glacier_fraction', 'n_observations', 'avg_ndsi', 'latitude', 'longitude']
            
            # Filter pixels based on quality criteria
            # 1. Must have reasonable glacier fraction (>0.1)
            # 2. Must have sufficient observations (>10)
            quality_pixels = pixel_summary[
                (pixel_summary['avg_glacier_fraction'] > 0.1) & 
                (pixel_summary['n_observations'] > 10)
            ].copy()
            
            if len(quality_pixels) == 0:
                self.logger.warning(f"No quality pixels found for {glacier_id}, using all data")
                return modis_data
            
            # Calculate distance from each pixel to AWS station (Haversine formula)
            def haversine_distance(lat1, lon1, lat2, lon2):
                """Calculate distance in km between two points on Earth."""
                R = 6371  # Earth's radius in km
                
                lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                
                a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                c = 2 * np.arcsin(np.sqrt(a))
                return R * c
            
            quality_pixels['distance_to_aws'] = haversine_distance(
                quality_pixels['latitude'], 
                quality_pixels['longitude'],
                aws_lat, 
                aws_lon
            )
            
            # Sort by performance (glacier fraction) first, then by distance
            # This prioritizes performance while considering proximity
            quality_pixels = quality_pixels.sort_values([
                'avg_glacier_fraction',  # Higher is better (descending)
                'distance_to_aws'        # Lower is better (ascending)
            ], ascending=[False, True])
            
            # Select the 2 closest best performing pixels
            selected_pixels = quality_pixels.head(2)
            selected_pixel_ids = set(selected_pixels['pixel_id'])
            
            self.logger.info(f"Selected 2 closest best performing pixels from {len(modis_data['pixel_id'].unique())} total pixels")
            self.logger.info(f"AWS station: {aws_station['name']} at ({aws_lat:.4f}, {aws_lon:.4f})")
            
            for _, pixel in selected_pixels.iterrows():
                self.logger.info(f"  Pixel {pixel['pixel_id']}: glacier_fraction={pixel['avg_glacier_fraction']:.3f}, "
                               f"distance={pixel['distance_to_aws']:.2f}km, observations={pixel['n_observations']}")
            
            # Filter MODIS data to only include selected pixels
            filtered_data = modis_data[modis_data['pixel_id'].isin(selected_pixel_ids)].copy()
            
            self.logger.info(f"Filtered MODIS data from {len(modis_data)} to {len(filtered_data)} observations")
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error applying pixel selection for {glacier_id}: {e}")
            self.logger.warning("Falling back to using all pixels")
            return modis_data
    
    def _apply_standard_pixel_selection(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Fallback method for standard pixel selection when AWS coordinates are not available."""
        try:
            # Get available pixels and their quality metrics
            pixel_summary = modis_data.groupby('pixel_id').agg({
                'glacier_fraction': 'mean',
                'albedo': 'count',  # Number of observations per pixel
                'ndsi': 'mean'
            }).reset_index()
            
            pixel_summary.columns = ['pixel_id', 'avg_glacier_fraction', 'n_observations', 'avg_ndsi']
            
            # Filter pixels based on quality criteria
            quality_pixels = pixel_summary[
                (pixel_summary['avg_glacier_fraction'] > 0.1) & 
                (pixel_summary['n_observations'] > 10)
            ].copy()
            
            if len(quality_pixels) == 0:
                self.logger.warning(f"No quality pixels found for {glacier_id}, using all data")
                return modis_data
            
            # Select top 2 pixels by glacier fraction
            selected_pixels = quality_pixels.nlargest(2, 'avg_glacier_fraction')
            selected_pixel_ids = set(selected_pixels['pixel_id'])
            
            self.logger.info(f"Selected 2 best pixels (no AWS coordinates available)")
            
            # Filter MODIS data to only include selected pixels
            filtered_data = modis_data[modis_data['pixel_id'].isin(selected_pixel_ids)].copy()
            
            return filtered_data
            
        except Exception as e:
            self.logger.error(f"Error in standard pixel selection: {e}")
            return modis_data
    
    def _perform_pivot_based_analysis(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis matching the user's methodology."""
        statistics = {}
        
        # Get available MODIS methods
        modis_methods = [col for col in merged_data.columns if col in ['MCD43A3', 'MOD09GA', 'MOD10A1']]
        
        if not modis_methods:
            self.logger.warning("No MODIS methods found in merged data")
            return statistics
        
        self.logger.info(f"Analyzing methods: {modis_methods}")
        
        # Calculate metrics for each method (matching user's approach)
        statistics['method_comparison'] = {}
        
        for method in modis_methods:
            if method in merged_data.columns and 'AWS' in merged_data.columns:
                # Get valid data
                valid_data = merged_data[[method, 'AWS']].dropna()
                
                if len(valid_data) > 0:
                    aws_vals = valid_data['AWS']
                    modis_vals = valid_data[method]
                    
                    # Calculate comprehensive statistics (matching user's approach)
                    r, p = scipy_stats.pearsonr(modis_vals, aws_vals)
                    rmse = np.sqrt(np.mean((modis_vals - aws_vals)**2))
                    mae = np.mean(np.abs(modis_vals - aws_vals))
                    bias = np.mean(modis_vals - aws_vals)
                    
                    # Additional metrics from user's analysis
                    diff = modis_vals - aws_vals
                    abs_diff = np.abs(diff)
                    rel_diff = diff / aws_vals * 100
                    
                    # Regression analysis
                    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(modis_vals, aws_vals)
                    
                    metrics = {
                        'n_samples': len(valid_data),
                        'r': r,
                        'p': p,
                        'r_squared': r**2,
                        'rmse': rmse,
                        'mae': mae,
                        'bias': bias,
                        'med_abs_error': np.median(abs_diff),
                        'std_error': np.std(diff),
                        'mean_rel_bias': np.mean(rel_diff),
                        'mean_abs_rel_error': np.mean(np.abs(rel_diff)),
                        'slope': slope,
                        'intercept': intercept,
                        'slope_stderr': std_err,
                        'mean_obs': np.mean(aws_vals),
                        'mean_pred': np.mean(modis_vals),
                        'std_obs': np.std(aws_vals),
                        'std_pred': np.std(modis_vals),
                        'within_5pct': np.sum(np.abs(rel_diff) <= 5) / len(valid_data) * 100,
                        'within_10pct': np.sum(np.abs(rel_diff) <= 10) / len(valid_data) * 100,
                        'within_15pct': np.sum(np.abs(rel_diff) <= 15) / len(valid_data) * 100
                    }
                    
                    statistics['method_comparison'][method] = metrics
                    
                    self.logger.info(f"\n{method} Statistics:")
                    self.logger.info(f"  n samples: {metrics['n_samples']}")
                    self.logger.info(f"  r: {metrics['r']:.4f}")
                    self.logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                    self.logger.info(f"  Bias: {metrics['bias']:.4f}")
        
        # Create method ranking (matching user's approach)
        if len(statistics['method_comparison']) > 1:
            ranking_data = []
            for method, metrics in statistics['method_comparison'].items():
                ranking_data.append({
                    'method': method,
                    'r_squared': metrics['r_squared'],
                    'rmse': metrics['rmse'],
                    'bias': abs(metrics['bias']),
                    'correlation': metrics['r'],
                    'n_samples': metrics['n_samples']
                })
            
            ranking_df = pd.DataFrame(ranking_data)
            ranking_df['r2_rank'] = ranking_df['r_squared'].rank(ascending=False)
            ranking_df['rmse_rank'] = ranking_df['rmse'].rank()
            ranking_df['bias_rank'] = ranking_df['bias'].rank() 
            ranking_df['overall_rank'] = (ranking_df['r2_rank'] + ranking_df['rmse_rank'] + ranking_df['bias_rank']) / 3
            
            statistics['ranking'] = ranking_df.sort_values('overall_rank')
            
            self.logger.info(f"\nMethod Ranking (by R²):")
            for i, row in statistics['ranking'].iterrows():
                self.logger.info(f"  {int(row['overall_rank'])}. {row['method']} (R²: {row['r_squared']:.4f})")
        
        return statistics
    
    def _perform_outlier_analysis(self, merged_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform residual-based outlier analysis matching user's methodology."""
        outlier_results = {}
        
        # Get available MODIS methods
        modis_methods = [col for col in merged_data.columns if col in ['MCD43A3', 'MOD09GA', 'MOD10A1']]
        
        if not modis_methods or 'AWS' not in merged_data.columns:
            self.logger.warning("Insufficient data for outlier analysis")
            return outlier_results
        
        self.logger.info("Performing residual-based outlier detection (2.5σ threshold)")
        
        stats_with_outliers = {}
        stats_without_outliers = {}
        outlier_info = {}
        
        for method in modis_methods:
            # Get valid data pairs
            mask = merged_data[[method, 'AWS']].notna().all(axis=1)
            if mask.sum() == 0:
                continue
                
            x_all = merged_data.loc[mask, 'AWS']
            y_all = merged_data.loc[mask, method]
            
            # Stats with outliers
            r_all, _ = scipy_stats.pearsonr(x_all, y_all)
            rmse_all = np.sqrt(np.mean((y_all - x_all)**2))
            mae_all = np.mean(np.abs(y_all - x_all))
            bias_all = np.mean(y_all - x_all)
            stats_with_outliers[method] = {
                'n': len(x_all), 'r': r_all, 'rmse': rmse_all, 
                'mae': mae_all, 'bias': bias_all
            }
            
            # Remove residual outliers (user's approach)
            slope, intercept = np.polyfit(x_all, y_all, 1)
            predicted = slope * x_all + intercept
            residuals = y_all - predicted
            residual_threshold = 2.5 * residuals.std()
            residual_outliers = np.abs(residuals) > residual_threshold
            
            # Create proper outlier series
            outlier_series = pd.Series(residual_outliers, index=mask[mask].index).reindex(merged_data.index, fill_value=False)
            clean_mask = mask & ~outlier_series
            
            # Store outlier information
            n_outliers = residual_outliers.sum()
            outlier_info[method] = {
                'n_total': len(x_all),
                'n_outliers': n_outliers,
                'outlier_percentage': (n_outliers / len(x_all)) * 100,
                'residual_threshold': residual_threshold,
                'outlier_indices': mask[mask].index[residual_outliers].tolist()
            }
            
            if clean_mask.sum() > 0:
                x_clean = merged_data.loc[clean_mask, 'AWS']
                y_clean = merged_data.loc[clean_mask, method]
                
                # Stats without outliers
                r_clean, _ = scipy_stats.pearsonr(x_clean, y_clean)
                rmse_clean = np.sqrt(np.mean((y_clean - x_clean)**2))
                mae_clean = np.mean(np.abs(y_clean - x_clean))
                bias_clean = np.mean(y_clean - x_clean)
                stats_without_outliers[method] = {
                    'n': len(x_clean), 'r': r_clean, 'rmse': rmse_clean,
                    'mae': mae_clean, 'bias': bias_clean
                }
                
                # Calculate improvements
                r_improvement = ((r_clean - r_all) / abs(r_all)) * 100 if r_all != 0 else 0
                rmse_improvement = ((rmse_all - rmse_clean) / rmse_all) * 100 if rmse_all != 0 else 0
                
                outlier_info[method].update({
                    'r_improvement_pct': r_improvement,
                    'rmse_improvement_pct': rmse_improvement
                })
                
                self.logger.info(f"{method} outlier analysis:")
                self.logger.info(f"  Removed {n_outliers} outliers ({n_outliers/len(x_all)*100:.1f}%)")
                self.logger.info(f"  Correlation improvement: {r_improvement:.1f}%")
                self.logger.info(f"  RMSE improvement: {rmse_improvement:.1f}%")
        
        outlier_results = {
            'stats_with_outliers': stats_with_outliers,
            'stats_without_outliers': stats_without_outliers,
            'outlier_info': outlier_info
        }
        
        return outlier_results
    
    def _generate_user_style_plots(self, merged_data: pd.DataFrame, statistics: Dict[str, Any], 
                                  glacier_id: str, output_dir: str):
        """Generate plots exactly matching the user's style from compare_albedo.ipynb."""
        plots_dir = os.path.join(output_dir, 'plots')
        
        # Available methods
        modis_methods = [col for col in merged_data.columns if col in ['MCD43A3', 'MOD09GA', 'MOD10A1']]
        
        if not modis_methods or 'AWS' not in merged_data.columns:
            self.logger.warning("Insufficient data for plotting")
            return
        
        # Extract month data for seasonal coloring (matching user's approach)
        months = merged_data.index.month
        unique_months = sorted(months.unique())
        month_names = {6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct'}
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_months)))
        
        # Method colors (matching user's notebook)
        method_colors = {'MCD43A3': 'darkblue', 'MOD09GA': 'darkred', 'MOD10A1': 'darkgreen'}
        
        # Create comprehensive scatter plot analysis (matching user's cell)
        fig, axes = plt.subplots(1, len(modis_methods), figsize=(6*len(modis_methods), 6))
        if len(modis_methods) == 1:
            axes = [axes]
        
        fig.suptitle(f'{glacier_id.upper()} Glacier - MODIS vs AWS Albedo Comparison\n(Pivot-Based Analysis)', 
                     fontsize=16, fontweight='bold')
        
        for i, method in enumerate(modis_methods):
            ax = axes[i]
            
            # Get valid data for this method
            valid_data = merged_data[[method, 'AWS']].dropna()
            
            if len(valid_data) == 0:
                ax.text(0.5, 0.5, f'No valid data\nfor {method}', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot data points colored by season (matching user's approach)
            for month, color in zip(unique_months, colors):
                if month in month_names:  # Only plot months that exist in data
                    month_mask = merged_data.index.month == month
                    month_data = valid_data[valid_data.index.isin(merged_data[month_mask].index)]
                    
                    if len(month_data) > 0:
                        month_label = month_names[month]
                        ax.scatter(month_data[method], month_data['AWS'], 
                                 c=[color], alpha=0.6, s=30, 
                                 label=month_label if i == 0 else "")
            
            # Add regression line (matching user's approach)
            if method in statistics.get('method_comparison', {}):
                stats_data = statistics['method_comparison'][method]
                slope = stats_data['slope']
                intercept = stats_data['intercept']
                
                x_min, x_max = valid_data[method].min(), valid_data[method].max()
                x_range = np.linspace(x_min, x_max, 50)
                y_pred = slope * x_range + intercept
                ax.plot(x_range, y_pred, color=method_colors.get(method, 'black'), 
                       linewidth=2, alpha=0.8, 
                       label=f'Regression (y={slope:.3f}x+{intercept:.3f})')
            
            # Add 1:1 line
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1, label='1:1 line')
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel('Satellite Albedo')
            ax.set_ylabel('AWS Albedo')
            ax.set_title(f'{method} vs AWS')
            ax.grid(True, alpha=0.3)
            
            # Add statistics text box (matching user's style)
            if method in statistics.get('method_comparison', {}):
                stats_data = statistics['method_comparison'][method]
                # Show method non-null count (like user's notebook) instead of valid pairs
                method_nonull_count = merged_data[method].count()
                stats_text = (f"{method}:\n"
                             f"n = {method_nonull_count}\n"
                             f"r = {stats_data['r']:.3f}\n"
                             f"RMSE = {stats_data['rmse']:.3f}\n"
                             f"MAE = {stats_data['mae']:.3f}\n"
                             f"Bias = {stats_data['bias']:.3f}")
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
                        fontsize=8, family='monospace')
            
            # Add legend only for the first plot
            if i == 0:
                # Create seasonal legend
                seasonal_legend = [plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color, markersize=6, 
                                            label=month_names.get(month, f'M{month}'))
                                for month, color in zip(unique_months, colors) 
                                if month in month_names]
                seasonal_legend.append(plt.Line2D([0], [0], color='k', linestyle='--', alpha=0.5, label='1:1 line'))
                ax.legend(handles=seasonal_legend, loc='lower right', fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(plots_dir, f"{glacier_id}_pivot_based_comprehensive_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created comprehensive analysis plot: {plot_path}")
    
    def _generate_comprehensive_plots(self, merged_data: pd.DataFrame, statistics: Dict[str, Any], 
                                    glacier_id: str, output_dir: str):
        """Generate comprehensive suite of visualization plots for albedo analysis."""
        plots_dir = os.path.join(output_dir, 'plots')
        
        # Available methods
        modis_methods = [col for col in merged_data.columns if col in ['MCD43A3', 'MOD09GA', 'MOD10A1']]
        
        if not modis_methods or 'AWS' not in merged_data.columns:
            self.logger.warning("Insufficient data for plotting")
            return
        
        try:
            # 1. Original user-style comprehensive scatterplot analysis
            self._generate_user_style_plots(merged_data, statistics, glacier_id, output_dir)
            
            # 2. Comprehensive summary figure (multi-panel analysis)
            self._create_comprehensive_summary_figure(merged_data, statistics, modis_methods, glacier_id, plots_dir)
            
            # 3. Temporal analysis plots
            self._create_time_series_plots(merged_data, modis_methods, glacier_id, plots_dir)
            
            # 4. Distribution comparison plots
            self._create_distribution_plots(merged_data, modis_methods, glacier_id, plots_dir)
            
            # 5. Outlier analysis plots (if outlier analysis is available)
            if 'outlier_analysis' in statistics:
                self._create_outlier_analysis_plots(merged_data, statistics['outlier_analysis'], modis_methods, glacier_id, plots_dir)
            
            # 6. Seasonal analysis plots
            self._create_seasonal_analysis_plots(merged_data, modis_methods, glacier_id, plots_dir)
            
            # 7. Correlation and bias analysis
            self._create_correlation_bias_plots(merged_data, statistics, modis_methods, glacier_id, plots_dir)
            
            # 8. Generate spatial maps (if mapping data available)
            self._create_spatial_maps(glacier_id, output_dir)
            
            self.logger.info(f"Generated comprehensive visualization suite with 7 different plot types")
            
        except Exception as e:
            self.logger.error(f"Error generating comprehensive plots: {e}")
            # Fallback to basic plotting
            self._generate_user_style_plots(merged_data, statistics, glacier_id, output_dir)
    
    def _create_comprehensive_summary_figure(self, merged_data: pd.DataFrame, statistics: Dict[str, Any], 
                                           modis_methods: List[str], glacier_id: str, plots_dir: str):
        """Create a comprehensive multi-panel summary figure."""
        # Create figure with 2-row layout (removed temporal evolution plot)
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # Method colors
        method_colors = {'MCD43A3': 'darkblue', 'MOD09GA': 'darkred', 'MOD10A1': 'darkgreen', 'AWS': 'black'}
        
        # Panel 1-3: Individual scatterplots (top row)
        for i, method in enumerate(modis_methods[:3]):
            ax = fig.add_subplot(gs[0, i])
            valid_data = merged_data[[method, 'AWS']].dropna()
            
            if len(valid_data) > 0:
                ax.scatter(valid_data[method], valid_data['AWS'], 
                          alpha=0.6, s=25, color=method_colors.get(method, 'blue'), 
                          edgecolors='black', linewidth=0.3)
                
                # Add regression line
                if method in statistics.get('method_comparison', {}):
                    stats_data = statistics['method_comparison'][method]
                    slope, intercept = stats_data['slope'], stats_data['intercept']
                    x_range = np.linspace(valid_data[method].min(), valid_data[method].max(), 50)
                    y_pred = slope * x_range + intercept
                    ax.plot(x_range, y_pred, color='red', linewidth=2, alpha=0.8)
                
                # Add 1:1 line
                ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
                
                # Statistics text
                if method in statistics.get('method_comparison', {}):
                    stats_data = statistics['method_comparison'][method]
                    stats_text = f"r={stats_data['r']:.3f}\nRMSE={stats_data['rmse']:.3f}\nn={stats_data['n_samples']}"
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           verticalalignment='top', fontsize=8)
            
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel(f'{method} Albedo')
            ax.set_ylabel('AWS Albedo' if i == 0 else '')
            ax.set_title(f'{method} vs AWS')
            ax.grid(True, alpha=0.3)
        
        # Panel 4: Performance comparison table (top right)
        ax_table = fig.add_subplot(gs[0, 3])
        ax_table.axis('off')
        
        if statistics.get('method_comparison'):
            table_data = []
            for method in modis_methods:
                if method in statistics['method_comparison']:
                    stats = statistics['method_comparison'][method]
                    table_data.append([
                        method,
                        f"{stats['r']:.3f}",
                        f"{stats['rmse']:.3f}",
                        f"{stats['bias']:.3f}",
                        f"{stats['n_samples']}"
                    ])
            
            if table_data:
                table = ax_table.table(cellText=table_data,
                                     colLabels=['Method', 'r', 'RMSE', 'Bias', 'n'],
                                     cellLoc='center', loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 2)
        
        ax_table.set_title('Performance Summary')
        
        # Panel 5: Distribution comparison (middle left span 2)
        ax_dist = fig.add_subplot(gs[1, :2])
        
        plot_data = []
        labels = []
        colors_list = []
        
        for method in ['AWS'] + modis_methods:
            if method in merged_data.columns:
                data_clean = merged_data[method].dropna()
                if len(data_clean) > 0:
                    plot_data.append(data_clean)
                    labels.append(f"{method}\n(n={len(data_clean)})")
                    colors_list.append(method_colors.get(method, 'gray'))
        
        if plot_data:
            box_plot = ax_dist.boxplot(plot_data, labels=labels, patch_artist=True)
            for box, color in zip(box_plot['boxes'], colors_list):
                box.set_facecolor(color)
                box.set_alpha(0.6)
        
        ax_dist.set_ylabel('Albedo')
        ax_dist.set_title('Albedo Distribution Comparison')
        ax_dist.grid(True, alpha=0.3)
        
        # Panel 6: Bias analysis (middle right span 2)
        ax_bias = fig.add_subplot(gs[1, 2:])
        
        bias_data = []
        bias_labels = []
        bias_colors = []
        
        for method in modis_methods:
            if method in merged_data.columns and 'AWS' in merged_data.columns:
                common_data = merged_data[[method, 'AWS']].dropna()
                if len(common_data) > 0:
                    bias = common_data[method] - common_data['AWS']
                    bias_data.append(bias)
                    bias_labels.append(f"{method}\n(n={len(bias)})")
                    bias_colors.append(method_colors.get(method, 'gray'))
        
        if bias_data:
            bias_box_plot = ax_bias.boxplot(bias_data, labels=bias_labels, patch_artist=True)
            for box, color in zip(bias_box_plot['boxes'], bias_colors):
                box.set_facecolor(color)
                box.set_alpha(0.6)
            ax_bias.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        
        ax_bias.set_ylabel('Albedo Bias (MODIS - AWS)')
        ax_bias.set_title('Bias Distribution Analysis')
        ax_bias.grid(True, alpha=0.3)
        
        
        # Overall title
        fig.suptitle(f'{glacier_id.upper()} Glacier - Comprehensive Albedo Analysis Summary', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # Save the plot
        plot_path = os.path.join(plots_dir, f"{glacier_id}_02_comprehensive_summary_figure.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created comprehensive summary figure: {plot_path}")
    
    def _create_time_series_plots(self, merged_data: pd.DataFrame, modis_methods: List[str], glacier_id: str, plots_dir: str):
        """Create temporal analysis plot with separate panels for each method."""
        method_colors = {'MCD43A3': 'darkblue', 'MOD09GA': 'darkred', 'MOD10A1': 'darkgreen', 'AWS': 'black'}
        
        # Create multi-panel layout - one panel per method
        all_methods = ['AWS'] + modis_methods
        n_methods = len(all_methods)
        
        fig, axes = plt.subplots(n_methods, 1, figsize=(15, 3 * n_methods), sharex=True)
        if n_methods == 1:
            axes = [axes]
        
        # Plot each method in its own panel
        ts_data = merged_data[['AWS'] + modis_methods].reset_index()
        
        for i, method in enumerate(all_methods):
            ax = axes[i]
            
            if method in ts_data.columns:
                # Get all daily data points (no aggregation)
                daily_data = ts_data[['date', method]].dropna()
                
                if len(daily_data) > 0:
                    dates = pd.to_datetime(daily_data['date'])
                    values = daily_data[method]
                    
                    # Use smaller scatter points for cleaner look
                    ax.scatter(dates, values, 
                             color=method_colors.get(method, 'gray'),
                             alpha=0.6, s=4, edgecolors='none')
                    
                    # Add data count to panel
                    ax.text(0.02, 0.95, f'n = {len(daily_data)}', 
                           transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            ax.set_ylabel('Albedo')
            ax.set_title(f'{method} Daily Observations', fontsize=12, color=method_colors.get(method, 'black'))
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)  # Standard albedo range
        
        # Only show x-axis label on bottom panel
        axes[-1].set_xlabel('Date')
        
        fig.suptitle(f'{glacier_id.upper()} Glacier - Temporal Analysis (All Daily Observations)', 
                    fontsize=16, fontweight='bold')
        fig.autofmt_xdate()
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, f"{glacier_id}_03_temporal_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created temporal analysis plot: {plot_path}")
    
    def _create_distribution_plots(self, merged_data: pd.DataFrame, modis_methods: List[str], glacier_id: str, plots_dir: str):
        """Create distribution comparison plots."""
        method_colors = {'MCD43A3': 'darkblue', 'MOD09GA': 'darkred', 'MOD10A1': 'darkgreen', 'AWS': 'black'}
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Box plots
        plot_data = []
        labels = []
        colors_list = []
        
        for method in ['AWS'] + modis_methods:
            if method in merged_data.columns:
                data_clean = merged_data[method].dropna()
                if len(data_clean) > 0:
                    plot_data.append(data_clean)
                    labels.append(f"{method}\n(n={len(data_clean)})")
                    colors_list.append(method_colors.get(method, 'gray'))
        
        if plot_data:
            box_plot = ax1.boxplot(plot_data, labels=labels, patch_artist=True)
            for box, color in zip(box_plot['boxes'], colors_list):
                box.set_facecolor(color)
                box.set_alpha(0.6)
        
        ax1.set_ylabel('Albedo')
        ax1.set_title('Albedo Distribution Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Violin plots
        if plot_data:
            parts = ax2.violinplot(plot_data, positions=range(1, len(plot_data)+1), showmeans=True)
            for i, (pc, color) in enumerate(zip(parts['bodies'], colors_list)):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
        
        ax2.set_xticks(range(1, len(labels)+1))
        ax2.set_xticklabels(labels)
        ax2.set_ylabel('Albedo')
        ax2.set_title('Albedo Distribution Density')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Histograms
        for method in ['AWS'] + modis_methods:
            if method in merged_data.columns:
                data_clean = merged_data[method].dropna()
                if len(data_clean) > 0:
                    ax3.hist(data_clean, bins=30, alpha=0.6, 
                           label=f"{method} (n={len(data_clean)})",
                           color=method_colors.get(method, 'gray'),
                           density=True)
        
        ax3.set_xlabel('Albedo')
        ax3.set_ylabel('Density')
        ax3.set_title('Albedo Distribution Histograms')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Cumulative distribution
        for method in ['AWS'] + modis_methods:
            if method in merged_data.columns:
                data_clean = merged_data[method].dropna().sort_values()
                if len(data_clean) > 0:
                    y_vals = np.arange(1, len(data_clean) + 1) / len(data_clean)
                    ax4.plot(data_clean, y_vals, 
                           label=f"{method}", color=method_colors.get(method, 'gray'),
                           linewidth=2)
        
        ax4.set_xlabel('Albedo')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Cumulative Distribution Functions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(f'{glacier_id.upper()} Glacier - Distribution Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, f"{glacier_id}_04_distribution_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created distribution analysis plot: {plot_path}")
    
    def _create_outlier_analysis_plots(self, merged_data: pd.DataFrame, outlier_analysis: Dict[str, Any], 
                                     modis_methods: List[str], glacier_id: str, plots_dir: str):
        """Create outlier analysis before/after plots."""
        method_colors = {'MCD43A3': 'darkblue', 'MOD09GA': 'darkred', 'MOD10A1': 'darkgreen'}
        
        fig, axes = plt.subplots(2, len(modis_methods), figsize=(6*len(modis_methods), 10))
        if len(modis_methods) == 1:
            axes = axes.reshape(2, 1)
        
        for i, method in enumerate(modis_methods):
            if method in outlier_analysis.get('outlier_info', {}):
                outlier_info = outlier_analysis['outlier_info'][method]
                
                # Get data
                valid_mask = merged_data[[method, 'AWS']].notna().all(axis=1)
                x_data = merged_data.loc[valid_mask, 'AWS']
                y_data = merged_data.loc[valid_mask, method]
                
                # Identify outliers
                outlier_indices = outlier_info.get('outlier_indices', [])
                outlier_mask = merged_data.index.isin(outlier_indices)
                
                # Panel 1: Before outlier removal
                ax1 = axes[0, i]
                
                # Plot all points
                ax1.scatter(x_data, y_data, alpha=0.6, s=25, 
                          color=method_colors.get(method, 'blue'))
                
                # Highlight outliers
                if len(outlier_indices) > 0:
                    outlier_x = merged_data.loc[outlier_mask & valid_mask, 'AWS']
                    outlier_y = merged_data.loc[outlier_mask & valid_mask, method]
                    ax1.scatter(outlier_x, outlier_y, color='red', s=50, alpha=0.8, 
                              marker='x')
                
                # Add regression line
                if len(x_data) > 1:
                    slope, intercept = np.polyfit(x_data, y_data, 1)
                    x_range = np.linspace(x_data.min(), x_data.max(), 50)
                    y_pred = slope * x_range + intercept
                    ax1.plot(x_range, y_pred, color='black', linewidth=2, alpha=0.8)
                
                ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
                ax1.set_xlim(0, 1)
                ax1.set_ylim(0, 1)
                ax1.set_xlabel('AWS Albedo')
                ax1.set_ylabel(f'{method} Albedo')
                ax1.set_title(f'{method} - Before Outlier Removal')
                ax1.grid(True, alpha=0.3)
                
                # Add statistics
                if method in outlier_analysis.get('stats_with_outliers', {}):
                    stats = outlier_analysis['stats_with_outliers'][method]
                    stats_text = (f"r={stats['r']:.3f}\n"
                                f"RMSE={stats['rmse']:.3f}\n"
                                f"n={stats['n']}\n"
                                f"MAE={stats['mae']:.3f}\n"
                                f"Bias={stats['bias']:.3f}")
                    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           verticalalignment='top', fontsize=8)
                
                # Panel 2: After outlier removal
                ax2 = axes[1, i]
                
                # Plot only non-outlier points
                clean_mask = valid_mask & ~outlier_mask
                if clean_mask.sum() > 0:
                    x_clean = merged_data.loc[clean_mask, 'AWS']
                    y_clean = merged_data.loc[clean_mask, method]
                    
                    ax2.scatter(x_clean, y_clean, alpha=0.6, s=25, 
                              color=method_colors.get(method, 'blue'))
                    
                    # Add regression line for clean data
                    if len(x_clean) > 1:
                        slope_clean, intercept_clean = np.polyfit(x_clean, y_clean, 1)
                        x_range = np.linspace(x_clean.min(), x_clean.max(), 50)
                        y_pred = slope_clean * x_range + intercept_clean
                        ax2.plot(x_range, y_pred, color='black', linewidth=2, alpha=0.8)
                
                ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
                ax2.set_xlim(0, 1)
                ax2.set_ylim(0, 1)
                ax2.set_xlabel('AWS Albedo')
                ax2.set_ylabel(f'{method} Albedo')
                ax2.set_title(f'{method} - After Outlier Removal')
                ax2.grid(True, alpha=0.3)
                
                # Add improved statistics
                if method in outlier_analysis.get('stats_without_outliers', {}):
                    stats = outlier_analysis['stats_without_outliers'][method]
                    improvement = outlier_info.get('r_improvement_pct', 0)
                    stats_text = (f"r={stats['r']:.3f}\n"
                                f"RMSE={stats['rmse']:.3f}\n"
                                f"n={stats['n']}\n"
                                f"MAE={stats['mae']:.3f}\n"
                                f"Bias={stats['bias']:.3f}\n"
                                f"Δr={improvement:.1f}%")
                    ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes,
                           bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                           verticalalignment='top', fontsize=8)
        
        fig.suptitle(f'{glacier_id.upper()} Glacier - Outlier Analysis (2.5σ threshold)', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, f"{glacier_id}_05_outlier_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created outlier analysis plot: {plot_path}")
    
    def _create_seasonal_analysis_plots(self, merged_data: pd.DataFrame, modis_methods: List[str], 
                                      glacier_id: str, plots_dir: str):
        """Create seasonal analysis plots."""
        method_colors = {'MCD43A3': 'darkblue', 'MOD09GA': 'darkred', 'MOD10A1': 'darkgreen', 'AWS': 'black'}
        
        # Add seasonal information
        seasonal_data = merged_data.copy()
        seasonal_data['month'] = seasonal_data.index.month
        seasonal_data['season'] = seasonal_data['month'].map({
            6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct'
        })
        
        # Filter to available seasons
        seasonal_data = seasonal_data.dropna(subset=['season'])
        
        if seasonal_data.empty:
            self.logger.warning("No seasonal data available for plotting")
            return
        
        # Create seasonal plots in 2x2 layout (no trend analysis)
        seasons = sorted(seasonal_data['season'].unique())
        
        # Use 2x2 layout for clean monthly distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Panel 1-4: Monthly box plots for each season
        for i, season in enumerate(seasons):
            if i < 4:  # Only show first 4 seasons in 2x2 layout
                ax = axes[i]
                season_data = seasonal_data[seasonal_data['season'] == season]
                
                plot_data = []
                labels = []
                colors_list = []
                
                for method in ['AWS'] + modis_methods:
                    if method in season_data.columns:
                        data_clean = season_data[method].dropna()
                        if len(data_clean) > 0:
                            plot_data.append(data_clean)
                            labels.append(f"{method}\n(n={len(data_clean)})")
                            colors_list.append(method_colors.get(method, 'gray'))
                
                if plot_data:
                    box_plot = ax.boxplot(plot_data, labels=labels, patch_artist=True)
                    for box, color in zip(box_plot['boxes'], colors_list):
                        box.set_facecolor(color)
                        box.set_alpha(0.6)
                
                ax.set_ylabel('Albedo')
                ax.set_title(f'{season} Albedo Distribution')
                ax.grid(True, alpha=0.3)
        
        # Hide any unused panels in 2x2 layout
        for i in range(len(seasons), 4):
            if i < len(axes):
                axes[i].set_visible(False)
        
        fig.suptitle(f'{glacier_id.upper()} Glacier - Seasonal Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, f"{glacier_id}_06_seasonal_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created seasonal analysis plot: {plot_path}")
    
    def _create_correlation_bias_plots(self, merged_data: pd.DataFrame, statistics: Dict[str, Any], 
                                     modis_methods: List[str], glacier_id: str, plots_dir: str):
        """Create correlation matrix and bias analysis plots."""
        method_colors = {'MCD43A3': 'darkblue', 'MOD09GA': 'darkred', 'MOD10A1': 'darkgreen', 'AWS': 'black'}
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Correlation matrix
        corr_data = merged_data[['AWS'] + modis_methods].corr()
        
        im = ax1.imshow(corr_data.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax1.set_xticks(range(len(corr_data.columns)))
        ax1.set_yticks(range(len(corr_data.columns)))
        ax1.set_xticklabels(corr_data.columns)
        ax1.set_yticklabels(corr_data.columns)
        
        # Add correlation values
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                text = ax1.text(j, i, f'{corr_data.values[i, j]:.3f}',
                              ha="center", va="center", color="black", fontweight='bold')
        
        ax1.set_title('Correlation Matrix')
        cbar = plt.colorbar(im, ax=ax1)
        cbar.set_label('Correlation Coefficient')
        
        # Panel 2: Method performance comparison
        if statistics.get('method_comparison'):
            methods = list(statistics['method_comparison'].keys())
            metrics = ['r', 'rmse', 'bias']
            
            x = np.arange(len(methods))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                values = [statistics['method_comparison'][method].get(metric, 0) for method in methods]
                ax2.bar(x + i*width, values, width, label=metric.upper(),
                       color=plt.cm.Set3(i/len(metrics)), alpha=0.7)
            
            ax2.set_xlabel('Methods')
            ax2.set_ylabel('Metric Value')
            ax2.set_title('Method Performance Comparison')
            ax2.set_xticks(x + width)
            ax2.set_xticklabels(methods)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Panel 3: Bias vs. AWS values
        for method in modis_methods:
            if method in merged_data.columns:
                common_data = merged_data[[method, 'AWS']].dropna()
                if len(common_data) > 0:
                    bias = common_data[method] - common_data['AWS']
                    ax3.scatter(common_data['AWS'], bias, alpha=0.6, s=20,
                              label=method, color=method_colors.get(method, 'gray'))
        
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax3.set_xlabel('AWS Albedo')
        ax3.set_ylabel('Bias (MODIS - AWS)')
        ax3.set_title('Bias vs. AWS Albedo')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: RMSE vs. sample size
        if statistics.get('method_comparison'):
            methods = []
            rmse_values = []
            n_samples = []
            colors = []
            
            for method in modis_methods:
                if method in statistics['method_comparison']:
                    stats = statistics['method_comparison'][method]
                    methods.append(method)
                    rmse_values.append(stats.get('rmse', 0))
                    n_samples.append(stats.get('n_samples', 0))
                    colors.append(method_colors.get(method, 'gray'))
            
            ax4.scatter(n_samples, rmse_values, s=100, c=colors, alpha=0.7)
            
            for i, method in enumerate(methods):
                ax4.annotate(method, (n_samples[i], rmse_values[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
            
            ax4.set_xlabel('Sample Size')
            ax4.set_ylabel('RMSE')
            ax4.set_title('RMSE vs. Sample Size')
            ax4.grid(True, alpha=0.3)
        
        fig.suptitle(f'{glacier_id.upper()} Glacier - Correlation and Bias Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        plot_path = os.path.join(plots_dir, f"{glacier_id}_07_correlation_bias_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Created correlation and bias analysis plot: {plot_path}")
    
    def _create_spatial_maps(self, glacier_id: str, output_dir: str):
        """Generate spatial maps for the glacier analysis."""
        try:
            from pathlib import Path
            
            # Create maps directory if it doesn't exist
            maps_dir = os.path.join(output_dir, 'maps')
            os.makedirs(maps_dir, exist_ok=True)
            
            self.logger.info("Generating spatial maps...")
            
            # Create a simple spatial visualization using merged data if available
            self._create_simple_pixel_map(glacier_id, maps_dir)
            
            self.logger.info(f"Generated spatial maps in: {maps_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating spatial maps: {e}")
    
    def _create_simple_pixel_map(self, glacier_id: str, maps_dir: str):
        """Create a comprehensive pixel location map highlighting selected analysis pixels."""
        try:
            # Check if we have the required data
            if not (hasattr(self, '_last_modis_data_original') and self._last_modis_data_original is not None):
                self.logger.warning("No original MODIS data available for spatial mapping")
                return
            
            original_data = self._last_modis_data_original
            selected_data = getattr(self, '_last_modis_data_selected', original_data)
            use_selected_pixels = getattr(self, '_use_selected_pixels', False)
            
            if 'pixel_id' not in original_data.columns:
                self.logger.warning("No pixel_id column found - cannot create pixel map")
                return
            
            # Get unique pixel locations for all available pixels
            all_pixels = original_data.groupby('pixel_id').agg({
                'longitude': 'first',
                'latitude': 'first',
                'glacier_fraction': 'mean' if 'glacier_fraction' in original_data.columns else 'first'
            }).reset_index()
            
            # Get unique pixel locations for selected pixels (used in analysis)
            selected_pixel_ids = set()
            if use_selected_pixels and not selected_data.empty:
                selected_pixels = selected_data.groupby('pixel_id').agg({
                    'longitude': 'first',
                    'latitude': 'first',
                    'glacier_fraction': 'mean' if 'glacier_fraction' in selected_data.columns else 'first'
                }).reset_index()
                selected_pixel_ids = set(selected_pixels['pixel_id'])
            else:
                selected_pixels = all_pixels.copy()
                selected_pixel_ids = set(all_pixels['pixel_id'])
            
            # Create comprehensive map
            fig, ax = plt.subplots(1, 1, figsize=(14, 12))
            
            # Try to load and plot glacier mask
            try:
                from src.analysis.glacier_mapping_simple import MultiGlacierMapperSimple
                mapper = MultiGlacierMapperSimple()
                mask_gdf = mapper.load_glacier_mask(glacier_id)
                
                if mask_gdf is not None:
                    # Plot glacier mask with blue outline and light blue fill
                    mask_gdf.plot(ax=ax, facecolor='lightblue', alpha=0.4,
                                 edgecolor='blue', linewidth=2, 
                                 label='Glacier Boundary')
            except Exception as e:
                self.logger.warning(f"Could not load glacier mask: {e}")
            
            # Plot ALL available pixels as background (smaller, gray)
            unselected_pixels = all_pixels[~all_pixels['pixel_id'].isin(selected_pixel_ids)]
            if not unselected_pixels.empty:
                ax.scatter(unselected_pixels['longitude'], unselected_pixels['latitude'], 
                         c='lightgray', s=40, alpha=0.5, edgecolors='gray', linewidth=0.5,
                         label=f'Available Pixels ({len(unselected_pixels)})', zorder=1)
            
            # Plot SELECTED pixels (larger, colored by glacier fraction)
            if not selected_pixels.empty and 'glacier_fraction' in selected_pixels.columns:
                scatter = ax.scatter(selected_pixels['longitude'], selected_pixels['latitude'], 
                               c=selected_pixels['glacier_fraction'], 
                               cmap='Blues', s=150, alpha=0.9, edgecolors='black', linewidth=2,
                               label=f'Selected for Analysis ({len(selected_pixels)})', zorder=3)
                cbar = plt.colorbar(scatter, ax=ax, label='Glacier Fraction', shrink=0.8)
                cbar.ax.tick_params(labelsize=10)
            else:
                ax.scatter(selected_pixels['longitude'], selected_pixels['latitude'], 
                         c='darkblue', s=150, alpha=0.9, edgecolors='black', linewidth=2,
                         label=f'Selected for Analysis ({len(selected_pixels)})', zorder=3)
            
            # Try to add AWS station location
            aws_lat, aws_lon, station_name = None, None, "AWS Station"
            try:
                from src.analysis.glacier_mapping_simple import MultiGlacierMapperSimple
                mapper = MultiGlacierMapperSimple()
                aws_coords = mapper.get_aws_coordinates(glacier_id)
                
                if aws_coords:
                    # Get the first AWS station for this glacier
                    for station_id, coords in aws_coords.items():
                        if coords.get('lat') is not None and coords.get('lon') is not None:
                            aws_lat = coords['lat']
                            aws_lon = coords['lon']
                            station_name = coords.get('name', 'AWS Station')
                            
                            # Plot AWS station as red star
                            ax.scatter(aws_lon, aws_lat, c='red', s=300, marker='*', 
                                     edgecolors='black', linewidth=2, 
                                     label=station_name, zorder=10)
                            self.logger.info(f"Added AWS station {station_name} at ({aws_lat}, {aws_lon})")
                            break
                    else:
                        self.logger.warning(f"No valid AWS coordinates found for {glacier_id}")
                else:
                    self.logger.warning(f"No AWS coordinates returned for {glacier_id}")
            except Exception as e:
                self.logger.warning(f"Could not add AWS coordinates: {e}")
            
            # Add analysis information text box
            if use_selected_pixels:
                analysis_text = f"Pixel Selection Analysis:\n"
                analysis_text += f"• Using {len(selected_pixels)} of {len(all_pixels)} available pixels\n"
                analysis_text += f"• Selection: 2 closest best-performing pixels to AWS\n"
                analysis_text += f"• Criteria: High glacier fraction + proximity to AWS"
                
                # Add distance information if we have AWS coordinates
                if aws_lat is not None and aws_lon is not None:
                    analysis_text += f"\n• AWS Station: {station_name}"
                    
                    # Calculate distances for selected pixels
                    if not selected_pixels.empty:
                        import numpy as np
                        def haversine_distance(lat1, lon1, lat2, lon2):
                            R = 6371  # Earth's radius in km
                            lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
                            dlat = lat2 - lat1
                            dlon = lon2 - lon1
                            a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
                            c = 2 * np.arcsin(np.sqrt(a))
                            return R * c
                        
                        distances = [haversine_distance(row['latitude'], row['longitude'], aws_lat, aws_lon) 
                                   for _, row in selected_pixels.iterrows()]
                        analysis_text += f"\n• Pixel distances: {', '.join([f'{d:.1f}km' for d in distances])}"
                
                # Add text box
                ax.text(0.02, 0.98, analysis_text, transform=ax.transAxes,
                        verticalalignment='top', horizontalalignment='left',
                        bbox=dict(boxstyle='round,pad=0.6', facecolor='wheat', alpha=0.9),
                        fontsize=9, family='monospace')
            
            # Formatting
            ax.set_xlabel('Longitude', fontsize=12)
            ax.set_ylabel('Latitude', fontsize=12) 
            
            # Enhanced title based on pixel selection mode
            if use_selected_pixels:
                title = f'{glacier_id.title()} Glacier - Analysis Pixels Highlighted'
            else:
                title = f'{glacier_id.title()} Glacier - All Available Pixels'
            ax.set_title(title, fontsize=14, fontweight='bold')
            
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=10)
            
            # Save the enhanced map
            output_path = os.path.join(maps_dir, f'{glacier_id}_comprehensive_map.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.logger.info(f"Created comprehensive pixel map with analysis highlighting: {output_path}")
                
        except Exception as e:
            self.logger.error(f"Error creating comprehensive pixel map: {e}")
    
    def _export_pivot_based_results(self, statistics: Dict[str, Any], merged_data: pd.DataFrame,
                                   glacier_id: str, output_dir: str) -> Dict[str, Any]:
        """Export results to files."""
        results_dir = os.path.join(output_dir, 'results')
        
        # Export comparison metrics
        if 'method_comparison' in statistics:
            comparison_df = pd.DataFrame(statistics['method_comparison']).T
            comparison_file = os.path.join(results_dir, f"{glacier_id}_pivot_method_comparison.csv")
            comparison_df.to_csv(comparison_file)
            self.logger.info(f"Exported method comparison: {comparison_file}")
        
        # Export ranking if available
        if 'ranking' in statistics:
            ranking_file = os.path.join(results_dir, f"{glacier_id}_pivot_method_ranking.csv")
            statistics['ranking'].to_csv(ranking_file, index=False)
            self.logger.info(f"Exported method ranking: {ranking_file}")
        
        # Export outlier analysis if available
        if 'outlier_analysis' in statistics:
            outlier_analysis = statistics['outlier_analysis']
            
            # Export with/without outliers comparison
            if 'stats_with_outliers' in outlier_analysis and 'stats_without_outliers' in outlier_analysis:
                comparison_data = []
                for method in ['MCD43A3', 'MOD09GA', 'MOD10A1']:
                    if method in outlier_analysis['stats_with_outliers']:
                        with_stats = outlier_analysis['stats_with_outliers'][method]
                        without_stats = outlier_analysis['stats_without_outliers'].get(method, {})
                        outlier_info = outlier_analysis['outlier_info'].get(method, {})
                        
                        comparison_data.append({
                            'method': method,
                            'condition': 'with_outliers',
                            'n': with_stats['n'],
                            'r': with_stats['r'],
                            'rmse': with_stats['rmse'],
                            'mae': with_stats['mae'],
                            'bias': with_stats['bias']
                        })
                        
                        if without_stats:
                            comparison_data.append({
                                'method': method,
                                'condition': 'without_outliers',
                                'n': without_stats['n'],
                                'r': without_stats['r'],
                                'rmse': without_stats['rmse'],
                                'mae': without_stats['mae'],
                                'bias': without_stats['bias']
                            })
                            
                            # Add improvement row
                            comparison_data.append({
                                'method': method,
                                'condition': 'improvement',
                                'n': outlier_info.get('n_outliers', 0),
                                'r': outlier_info.get('r_improvement_pct', 0),
                                'rmse': outlier_info.get('rmse_improvement_pct', 0),
                                'mae': '',
                                'bias': ''
                            })
                
                outlier_comparison_df = pd.DataFrame(comparison_data)
                outlier_file = os.path.join(results_dir, f"{glacier_id}_pivot_outlier_analysis.csv")
                outlier_comparison_df.to_csv(outlier_file, index=False)
                self.logger.info(f"Exported outlier analysis: {outlier_file}")
        
        # Export merged data for verification
        merged_file = os.path.join(results_dir, f"{glacier_id}_pivot_merged_data.csv")
        merged_data.to_csv(merged_file)
        self.logger.info(f"Exported merged data: {merged_file}")
        
        # Create summary
        summary = {
            'glacier_id': glacier_id,
            'analysis_type': 'pivot_based',
            'analysis_timestamp': get_timestamp(),
            'output_directory': output_dir,
            'n_merged_observations': len(merged_data),
            'available_methods': [col for col in merged_data.columns if col in ['MCD43A3', 'MOD09GA', 'MOD10A1']],
            'date_range': {
                'start': merged_data.index.min().strftime('%Y-%m-%d'),
                'end': merged_data.index.max().strftime('%Y-%m-%d')
            },
            'statistics': statistics
        }
        
        return summary


def main():
    """Main function to run the pivot-based analysis."""
    parser = argparse.ArgumentParser(description='Pivot-Based MODIS Albedo Analysis Framework')
    parser.add_argument('--glacier', type=str, help='Glacier ID to process')
    parser.add_argument('--all-glaciers', action='store_true', 
                       help='Process all glaciers in configuration')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--output-summary', type=str, 
                       help='Output file for summary results')
    
    args = parser.parse_args()
    
    if not args.glacier and not args.all_glaciers:
        parser.error("Must specify either --glacier or --all-glaciers")
    
    try:
        # Initialize pipeline
        pipeline = PivotBasedAlbedoAnalysis(args.config)
        
        # Process glaciers
        all_results = []
        
        if args.glacier:
            # Process single glacier
            result = pipeline.process_glacier(args.glacier)
            all_results.append(result)
            
        elif args.all_glaciers:
            # Process all glaciers
            glacier_sites_config = load_config('config/glacier_sites.yaml')
            glacier_ids = list(glacier_sites_config['glaciers'].keys())
            
            for glacier_id in glacier_ids:
                try:
                    result = pipeline.process_glacier(glacier_id)
                    all_results.append(result)
                except Exception as e:
                    logging.error(f"Failed to process {glacier_id}: {e}")
                    continue
        
        # Export summary if requested
        if args.output_summary and all_results:
            summary_df = pd.DataFrame([
                {
                    'glacier_id': result['glacier_id'],
                    'analysis_type': result['analysis_type'],
                    'analysis_timestamp': result['analysis_timestamp'],
                    'n_merged_observations': result['n_merged_observations'],
                    'output_directory': result['output_directory']
                }
                for result in all_results
            ])
            summary_df.to_csv(args.output_summary, index=False)
            print(f"Summary exported to: {args.output_summary}")
        
        print(f"\nPivot-based analysis completed for {len(all_results)} glacier(s)")
        
        # Print key results
        for result in all_results:
            print(f"\n{result['glacier_id']}:")
            print(f"  - {result['n_merged_observations']} merged observations")
            print(f"  - Methods: {', '.join(result['available_methods'])}")
            print(f"  - Date range: {result['date_range']['start']} to {result['date_range']['end']}")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()