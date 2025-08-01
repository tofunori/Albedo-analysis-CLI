#!/usr/bin/env python3
"""
Method Performance Bar Chart Generator

This standalone script creates the exact 3×4 method performance comparison
showing Correlation, RMSE, Bias, and MAE metrics across glaciers and MODIS methods.
It includes the complete data pipeline from raw CSV files to publication-ready bar charts.

Features:
- 3×4 bar chart matrix (3 glaciers × 4 metrics)
- Method-specific color coding with best performance highlighting
- Gold star indicators for best performing methods
- Comprehensive statistical analysis (R, RMSE, MAE, Bias)
- Support for all 3 glaciers: Athabasca, Haig, Coropuna

Author: Generated from Albedo Analysis Framework
Date: 2025-07-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'data_paths': {
        'athabasca': {
            'modis': "D:/Documents/Projects/athabasca_analysis/data/csv/Athabasca_Terra_Aqua_MultiProduct_2014-01-01_to_2021-01-01.csv",
            'aws': "D:/Documents/Projects/athabasca_analysis/data/csv/iceAWS_Atha_albedo_daily_20152020_filled_clean.csv"
        },
        'haig': {
            'modis': "D:/Documents/Projects/Haig_analysis/data/csv/Haig_MODIS_Pixel_Analysis_MultiProduct_2002_to_2016_fraction.csv",
            'aws': "D:/Documents/Projects/Haig_analysis/data/csv/HaigAWS_daily_2002_2015_gapfilled.csv"
        },
        'coropuna': {
            'modis': "D:/Documents/Projects/Coropuna_glacier/data/csv/coropuna_glacier_2014-01-01_to_2025-01-01.csv",
            'aws': "D:/Documents/Projects/Coropuna_glacier/data/csv/COROPUNA_simple.csv"
        }
    },
    'aws_stations': {
        'athabasca': {'lat': 52.1949, 'lon': -117.2431, 'name': 'Athabasca AWS'},
        'haig': {'lat': 50.7186, 'lon': -115.3433, 'name': 'Haig AWS'},
        'coropuna': {'lat': -15.5181, 'lon': -72.6617, 'name': 'Coropuna AWS'}
    },
    'colors': {
        'athabasca': '#1f77b4',  # Blue
        'haig': '#ff7f0e',       # Orange  
        'coropuna': '#2ca02c',   # Green
        'MOD09GA': '#9467bd',    # Purple (Terra)
        'MYD09GA': '#17becf',    # Cyan (Aqua)
        'MCD43A3': '#d62728',    # Red
        'MOD10A1': '#8c564b',    # Brown (Terra)
        'MYD10A1': '#e377c2',    # Pink (Aqua)
        'mcd43a3': '#d62728',    # Red
        'mod09ga': '#9467bd',    # Purple (Terra)
        'myd09ga': '#17becf',    # Cyan (Aqua)
        'mod10a1': '#8c564b',    # Brown (Terra)
        'myd10a1': '#e377c2'     # Pink (Aqua)
    },
    'methods': ['MOD09GA', 'MCD43A3', 'MOD10A1'],
    'method_mapping': {
        # Map different case variations to standard format
        'mcd43a3': 'MCD43A3',
        'MCD43A3': 'MCD43A3',
        'mod09ga': 'MOD09GA', 
        'MOD09GA': 'MOD09GA',
        'mod10a1': 'MOD10A1',
        'MOD10A1': 'MOD10A1'
    },
    'outlier_threshold': 2.5,
    'quality_filters': {
        'min_glacier_fraction': 0.1,
        'min_observations': 10
    },
    'visualization': {
        'figsize': (16, 12),
        'dpi': 300,
        'style': 'seaborn-v0_8'
    }
}


class DataLoader:
    """Handles loading and preprocessing of MODIS and AWS data for all glaciers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_glacier_data(self, glacier_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load MODIS and AWS data for a specific glacier."""
        logger.info(f"Loading data for {glacier_id} glacier...")
        
        paths = self.config['data_paths'][glacier_id]
        
        # Load MODIS data
        modis_data = self._load_modis_data(paths['modis'], glacier_id)
        
        # Load AWS data
        aws_data = self._load_aws_data(paths['aws'], glacier_id)
        
        logger.info(f"Loaded {len(modis_data):,} MODIS and {len(aws_data):,} AWS records for {glacier_id}")
        
        return modis_data, aws_data
    
    def _load_modis_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        """Load MODIS data with glacier-specific parsing."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"MODIS data file not found: {file_path}")
        
        logger.info(f"Loading MODIS data from: {file_path}")
        data = pd.read_csv(file_path)
        
        # Convert date to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Glacier-specific processing
        if glacier_id == 'coropuna':
            # Coropuna has method column - already in long format
            if 'method' in data.columns and 'albedo' in data.columns:
                logger.info("Coropuna data is in long format")
                # Apply method mapping to standardize names
                data['method'] = data['method'].map(self.config['method_mapping']).fillna(data['method'])
                return data
        
        # For other glaciers, check if conversion to long format is needed
        if 'method' not in data.columns:
            logger.info(f"Converting {glacier_id} data to long format")
            data = self._convert_to_long_format(data, glacier_id)
        else:
            # Data already has method column, just apply mapping
            logger.info(f"{glacier_id} data already in long format")
            data['method'] = data['method'].map(self.config['method_mapping']).fillna(data['method'])
        
        return data
    
    def _convert_to_long_format(self, data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Convert wide format MODIS data to long format."""
        long_format_rows = []
        
        # Define method mappings based on available columns
        method_columns = {}
        for col in data.columns:
            if 'MOD09GA' in col and 'albedo' in col:
                method_columns['MOD09GA'] = col
            elif 'MYD09GA' in col and 'albedo' in col:
                method_columns['MYD09GA'] = col
            elif 'MOD10A1' in col and 'albedo' in col:
                method_columns['MOD10A1'] = col
            elif 'MYD10A1' in col and 'albedo' in col:
                method_columns['MYD10A1'] = col
            elif 'MCD43A3' in col and 'albedo' in col:
                method_columns['MCD43A3'] = col
            elif col in ['MOD09GA', 'MYD09GA', 'MOD10A1', 'MYD10A1', 'MCD43A3']:
                method_columns[col] = col
        
        for method, col_name in method_columns.items():
            if col_name not in data.columns:
                continue
                
            # Extract data for this method
            method_data = data[data[col_name].notna()][['pixel_id', 'date', col_name]].copy()
            
            if len(method_data) > 0:
                method_data['method'] = method
                method_data['albedo'] = method_data[col_name]
                method_data = method_data.drop(columns=[col_name])
                
                # Add spatial coordinates if available
                for coord_col in ['longitude', 'latitude']:
                    if coord_col in data.columns:
                        method_data[coord_col] = data.loc[method_data.index, coord_col]
                
                # Add glacier fraction if available
                glacier_frac_cols = [c for c in data.columns if 'glacier_fraction' in c.lower()]
                if glacier_frac_cols:
                    method_data['glacier_fraction'] = data.loc[method_data.index, glacier_frac_cols[0]]
                
                long_format_rows.append(method_data)
        
        if long_format_rows:
            return pd.concat(long_format_rows, ignore_index=True)
        else:
            logger.error(f"No valid method data found for {glacier_id}")
            return pd.DataFrame()
    
    def _load_aws_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        """Load AWS data with glacier-specific parsing."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"AWS data file not found: {file_path}")
        
        logger.info(f"Loading AWS data from: {file_path}")
        
        if glacier_id == 'haig':
            # Haig has special format: semicolon separated, European decimal
            aws_data = pd.read_csv(file_path, sep=';', skiprows=6, decimal=',')
            aws_data.columns = aws_data.columns.str.strip()
            
            # Process Year and Day columns to create datetime
            aws_data = aws_data.dropna(subset=['Year', 'Day'])
            aws_data['Year'] = aws_data['Year'].astype(int)
            aws_data['Day'] = aws_data['Day'].astype(int)
            
            # Convert Day of Year to datetime
            aws_data['date'] = pd.to_datetime(
                aws_data['Year'].astype(str) + '-01-01'
            ) + pd.to_timedelta(aws_data['Day'] - 1, unit='D')
            
            # Find albedo column
            albedo_cols = [col for col in aws_data.columns if 'albedo' in col.lower()]
            if albedo_cols:
                albedo_col = albedo_cols[0]
                aws_data['Albedo'] = pd.to_numeric(aws_data[albedo_col], errors='coerce')
            else:
                raise ValueError(f"No albedo column found in Haig AWS data")
                
        elif glacier_id == 'coropuna':
            # Coropuna format: Timestamp, Albedo
            aws_data = pd.read_csv(file_path)
            aws_data['date'] = pd.to_datetime(aws_data['Timestamp'])
            
        elif glacier_id == 'athabasca':
            # Athabasca format: Time, Albedo
            aws_data = pd.read_csv(file_path)
            aws_data['date'] = pd.to_datetime(aws_data['Time'])
        
        # Clean and validate data
        aws_data = aws_data[['date', 'Albedo']].copy()
        aws_data = aws_data.dropna(subset=['Albedo'])
        aws_data = aws_data[aws_data['Albedo'] > 0]  # Remove invalid albedo values
        aws_data = aws_data.drop_duplicates().sort_values('date').reset_index(drop=True)
        
        return aws_data


class PixelSelector:
    """Implements intelligent pixel selection based on distance to AWS stations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def select_best_pixels(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Select best pixels for analysis based on AWS distance and glacier fraction."""
        logger.info(f"Applying pixel selection for {glacier_id}...")
        
        # Get AWS station coordinates
        aws_station = self.config['aws_stations'][glacier_id]
        aws_lat, aws_lon = aws_station['lat'], aws_station['lon']
        
        # Get available pixels with their quality metrics
        pixel_summary = modis_data.groupby('pixel_id').agg({
            'glacier_fraction': 'mean',
            'albedo': 'count',
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        pixel_summary.columns = ['pixel_id', 'avg_glacier_fraction', 'n_observations', 'latitude', 'longitude']
        
        # Apply quality filters
        quality_filters = self.config['quality_filters']
        quality_pixels = pixel_summary[
            (pixel_summary['avg_glacier_fraction'] > quality_filters['min_glacier_fraction']) & 
            (pixel_summary['n_observations'] > quality_filters['min_observations'])
        ].copy()
        
        if len(quality_pixels) == 0:
            logger.warning(f"No quality pixels found for {glacier_id}, using all data")
            return modis_data
        
        # Calculate distance to AWS station using Haversine formula
        quality_pixels['distance_to_aws'] = self._haversine_distance(
            quality_pixels['latitude'], quality_pixels['longitude'], aws_lat, aws_lon
        )
        
        # For Athabasca (small dataset), use all pixels
        if glacier_id == 'athabasca':
            selected_pixel_ids = quality_pixels['pixel_id'].tolist()
            logger.info(f"Using all {len(selected_pixel_ids)} pixels for {glacier_id} (small dataset)")
        else:
            # Sort by glacier fraction (descending) then distance (ascending)
            quality_pixels = quality_pixels.sort_values([
                'avg_glacier_fraction', 'distance_to_aws'
            ], ascending=[False, True])
            
            # Select the best performing pixel
            selected_pixels = quality_pixels.head(1)
            selected_pixel_ids = selected_pixels['pixel_id'].tolist()
            
            logger.info(f"Selected {len(selected_pixel_ids)} best pixel(s) for {glacier_id}")
            for _, pixel in selected_pixels.iterrows():
                logger.info(f"  Pixel {pixel['pixel_id']}: "
                           f"glacier_fraction={pixel['avg_glacier_fraction']:.3f}, "
                           f"distance={pixel['distance_to_aws']:.2f}km, "
                           f"observations={pixel['n_observations']}")
        
        # Filter MODIS data to selected pixels
        filtered_data = modis_data[modis_data['pixel_id'].isin(selected_pixel_ids)].copy()
        logger.info(f"Filtered MODIS data from {len(modis_data)} to {len(filtered_data)} observations")
        
        return filtered_data
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance using Haversine formula."""
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c


class DataProcessor:
    """Handles AWS-MODIS data merging and statistical processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def merge_and_process(self, modis_data: pd.DataFrame, aws_data: pd.DataFrame, 
                         glacier_id: str) -> pd.DataFrame:
        """Merge AWS and MODIS data and calculate statistics for each method."""
        logger.info(f"Merging and processing data for {glacier_id}...")
        
        results = []
        
        # Filter to only include methods we want to analyze
        target_methods = self.config['methods']
        available_methods = [m for m in modis_data['method'].unique() if m in target_methods]
        
        for method in available_methods:
            # Filter MODIS data for this method
            method_data = modis_data[modis_data['method'] == method].copy()
            
            if len(method_data) == 0:
                logger.warning(f"No {method} data found for {glacier_id}")
                continue
            
            # Merge with AWS data on date
            merged = method_data.merge(aws_data, on='date', how='inner')
            
            if len(merged) < 3:  # Need minimum data points
                logger.warning(f"Insufficient {method} data for {glacier_id}: {len(merged)} points")
                continue
            
            # Apply outlier filtering
            aws_clean, modis_clean = self._apply_outlier_filtering(
                merged['Albedo'].values, merged['albedo'].values
            )
            
            if len(aws_clean) < 3:
                logger.warning(f"Insufficient {method} data after outlier filtering for {glacier_id}")
                continue
            
            # Calculate statistics
            stats = self._calculate_statistics(aws_clean, modis_clean)
            
            results.append({
                'glacier_id': glacier_id,
                'method': method,
                'aws_values': aws_clean,
                'modis_values': modis_clean,
                **stats
            })
            
            logger.info(f"Processed {method} for {glacier_id}: "
                       f"{len(aws_clean)} samples, r={stats['r']:.3f}")
        
        return pd.DataFrame(results)
    
    def _apply_outlier_filtering(self, aws_vals: np.ndarray, modis_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply 2.5σ outlier filtering to AWS-MODIS pairs."""
        if len(aws_vals) < 3:
            return aws_vals, modis_vals
        
        # Calculate residuals
        residuals = modis_vals - aws_vals
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        # Apply threshold
        threshold = self.config['outlier_threshold'] * std_residual
        mask = np.abs(residuals - mean_residual) <= threshold
        
        return aws_vals[mask], modis_vals[mask]
    
    def _calculate_statistics(self, aws_vals: np.ndarray, modis_vals: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive statistics between AWS and MODIS values."""
        if len(aws_vals) == 0:
            return {
                'r': np.nan, 'correlation': np.nan, 'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 
                'n_samples': 0, 'p_value': 1.0
            }
        
        # Basic statistics
        correlation = np.corrcoef(aws_vals, modis_vals)[0, 1] if len(aws_vals) > 1 else np.nan
        
        # Error metrics
        residuals = modis_vals - aws_vals
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        bias = np.mean(residuals)
        
        # Statistical significance
        if len(aws_vals) > 2 and not np.isnan(correlation):
            t_stat = correlation * np.sqrt((len(aws_vals) - 2) / (1 - correlation**2))
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(aws_vals) - 2))
        else:
            p_value = 1.0
        
        return {
            'r': correlation if not np.isnan(correlation) else 0.0,
            'correlation': correlation if not np.isnan(correlation) else 0.0,
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'n_samples': len(aws_vals),
            'p_value': p_value
        }


class MethodPerformanceVisualizer:
    """Creates the 3×4 method performance bar chart matrix."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def create_performance_matrix(self, processed_data: List[pd.DataFrame], 
                                 output_path: Optional[str] = None) -> plt.Figure:
        """Create the 3×4 method performance bar chart matrix."""
        logger.info("Creating method performance bar chart matrix...")
        
        # Set matplotlib style
        try:
            plt.style.use(self.config['visualization']['style'])
        except:
            logger.warning("Could not set plotting style, using default")
        
        # Create 3x4 subplot layout (3 glaciers x 4 metrics)
        fig, axes = plt.subplots(3, 4, figsize=self.config['visualization']['figsize'])
        
        # Enhanced title with pixel selection information
        main_title = 'Method Performance by Glacier and Metric'
        subtitle = "Selected Best Pixels: 2/1/1 (Closest to AWS Stations)"
        fig.suptitle(f'{main_title}\n{subtitle}', fontsize=16, fontweight='bold')
        
        metrics = ['r', 'rmse', 'bias', 'mae']
        metric_titles = ['Correlation (r)', 'RMSE', 'Bias', 'MAE']
        glaciers = ['athabasca', 'coropuna', 'haig']  # Match your image order
        
        # Create combined dataframe for easier processing
        all_data = pd.concat(processed_data, ignore_index=True) if processed_data else pd.DataFrame()
        
        # Create plots for each glacier-metric combination
        for glacier_idx, glacier_id in enumerate(glaciers):
            glacier_data = all_data[all_data['glacier_id'] == glacier_id] if not all_data.empty else pd.DataFrame()
            
            for metric_idx, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
                ax = axes[glacier_idx, metric_idx]
                
                if not glacier_data.empty:
                    # Prepare data for this glacier-metric combination
                    method_values = []
                    method_labels = []
                    colors = []
                    
                    # Get available methods for this glacier, sorted
                    available_methods = sorted(glacier_data['method'].unique())
                    
                    for method in available_methods:
                        method_data = glacier_data[glacier_data['method'] == method]
                        if not method_data.empty:
                            method_values.append(method_data[metric].iloc[0])
                            method_labels.append(method)
                            colors.append(self.config['colors'].get(method, 'gray'))
                    
                    if method_values:  # If we have data to plot
                        # Create bar chart
                        bars = ax.bar(range(len(method_labels)), method_values, 
                                     color=colors, alpha=0.7, edgecolor='black', linewidth=1)
                        
                        # Add value labels on bars
                        for i, (bar, value) in enumerate(zip(bars, method_values)):
                            height = bar.get_height()
                            # Position text above or below bar depending on value
                            if metric == 'bias' and value < 0:
                                va = 'top'
                                y_pos = height - abs(height)*0.02
                            else:
                                va = 'bottom'
                                y_pos = height + abs(height)*0.02
                            
                            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                                   f'{value:.3f}', ha='center', va=va, 
                                   fontsize=9, fontweight='bold')
                        
                        # Highlight best performing method
                        if len(method_values) > 1:
                            if metric == 'r':  # Higher is better
                                best_idx = method_values.index(max(method_values))
                            elif metric in ['rmse', 'mae']:  # Lower is better
                                best_idx = method_values.index(min(method_values))
                            elif metric == 'bias':  # Closest to zero is better
                                best_idx = method_values.index(min(method_values, key=abs))
                            
                            # Add gold highlighting to best method
                            bars[best_idx].set_edgecolor('gold')
                            bars[best_idx].set_linewidth(3)
                            
                            # Add gold star
                            best_value = method_values[best_idx]
                            star_y = best_value + abs(best_value)*0.08 if best_value >= 0 else best_value - abs(best_value)*0.08
                            ax.text(best_idx, star_y, '★', ha='center', va='center', 
                                   fontsize=12, color='gold', fontweight='bold')
                        
                        # Customize subplot
                        ax.set_xticks(range(len(method_labels)))
                        ax.set_xticklabels(method_labels, rotation=45, ha='right')
                        ax.grid(True, alpha=0.3, axis='y')
                        
                        # Set y-axis limits for better comparison across glaciers
                        if metric == 'r':
                            ax.set_ylim(0, 1)
                        elif metric in ['rmse', 'mae']:
                            # Set common scale for error metrics
                            all_values = all_data[metric].values
                            ax.set_ylim(0, max(all_values) * 1.15)
                        elif metric == 'bias':
                            # Center bias around zero
                            all_values = all_data[metric].values
                            max_abs = max(abs(all_values)) if len(all_values) > 0 else 0.1
                            ax.set_ylim(-max_abs * 1.2, max_abs * 1.2)
                            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                    else:
                        # No data available
                        ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                               ha='center', va='center', fontsize=10, style='italic')
                else:
                    # No data for this glacier
                    ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes,
                           ha='center', va='center', fontsize=10, style='italic')
                
                # Add glacier name to leftmost plots
                if metric_idx == 0:
                    ax.set_ylabel(f'{glacier_id.title()}\n{metric_title}', fontweight='bold')
                else:
                    ax.set_ylabel(metric_title)
                
                # Add metric title to top row
                if glacier_idx == 0:
                    ax.set_title(metric_title, fontweight='bold')
        
        # Add legend for method colors (show all possible methods)
        legend_methods = ['MOD09GA', 'MCD43A3', 'MOD10A1']
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=self.config['colors'].get(method, 'gray'), 
                                       edgecolor='black', alpha=0.7, label=method) 
                          for method in legend_methods]
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='none', 
                                           edgecolor='gold', linewidth=3, 
                                           label='Best Performance'))
        
        fig.legend(legend_elements, [elem.get_label() for elem in legend_elements], 
                  loc='center', bbox_to_anchor=(0.5, 0.02), ncol=4, fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.08)  # Make room for legend
        
        # Save figure if path provided
        if output_path:
            fig.savefig(output_path, dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight', facecolor='white')
            logger.info(f"Method performance matrix saved to: {output_path}")
        
        return fig


def main():
    """Main execution function."""
    logger.info("Starting Method Performance Bar Chart Generation")
    
    # Initialize components
    data_loader = DataLoader(CONFIG)
    pixel_selector = PixelSelector(CONFIG)
    data_processor = DataProcessor(CONFIG)
    visualizer = MethodPerformanceVisualizer(CONFIG)
    
    # Process each glacier
    all_processed_data = []
    
    for glacier_id in ['athabasca', 'haig', 'coropuna']:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {glacier_id.upper()} Glacier")
            logger.info(f"{'='*50}")
            
            # Load data
            modis_data, aws_data = data_loader.load_glacier_data(glacier_id)
            
            # Apply pixel selection
            selected_modis = pixel_selector.select_best_pixels(modis_data, glacier_id)
            
            # Process and merge data
            processed = data_processor.merge_and_process(selected_modis, aws_data, glacier_id)
            
            if not processed.empty:
                all_processed_data.append(processed)
                logger.info(f"Successfully processed {glacier_id}: {len(processed)} methods")
            else:
                logger.warning(f"No processed data for {glacier_id}")
                
        except Exception as e:
            logger.error(f"Error processing {glacier_id}: {e}")
            continue
    
    # Create visualization
    if all_processed_data:
        logger.info(f"\n{'='*50}")
        logger.info("Creating Method Performance Bar Chart Matrix")
        logger.info(f"{'='*50}")
        
        # Generate output filename
        output_path = "method_performance_bar_chart_matrix.png"
        
        # Create the plot
        fig = visualizer.create_performance_matrix(all_processed_data, output_path)
        
        # Show the plot
        plt.show()
        
        logger.info(f"\nSUCCESS: Method performance matrix generated and saved to {output_path}")
        logger.info(f"Total glaciers processed: {len(all_processed_data)}")
        
    else:
        logger.error("No data could be processed for any glacier")


if __name__ == "__main__":
    main()