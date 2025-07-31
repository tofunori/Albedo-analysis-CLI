#!/usr/bin/env python3
"""
AWS vs MODIS Albedo Scatterplot Matrix Generator

Modular pipeline: DATA LOADING → PIXEL SELECTION → DATA PROCESSING → VISUALIZATION
Supports 3 glaciers with intelligent pixel selection and comprehensive statistics.
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
    'colors': {'athabasca': '#1f77b4', 'haig': '#ff7f0e', 'coropuna': '#2ca02c'},
    'methods': ['MCD43A3', 'MOD09GA', 'MOD10A1'],
    'method_mapping': {
        'mcd43a3': 'MCD43A3', 'MCD43A3': 'MCD43A3', 'mod09ga': 'MOD09GA', 'MOD09GA': 'MOD09GA',
        'myd09ga': 'MOD09GA', 'MYD09GA': 'MOD09GA', 'mod10a1': 'MOD10A1', 'MOD10A1': 'MOD10A1',
        'myd10a1': 'MOD10A1', 'MYD10A1': 'MOD10A1'
    },
    'outlier_threshold': 2.5,
    'quality_filters': {'min_glacier_fraction': 0.1, 'min_observations': 10},
    'visualization': {'figsize': (15, 12), 'dpi': 300, 'style': 'seaborn-v0_8'}
}



class DataLoader:
    """Handles loading and preprocessing of MODIS and AWS data for all glaciers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_glacier_data(self, glacier_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess MODIS and AWS data for a specific glacier."""
        if glacier_id not in self.config['data_paths']:
            raise ValueError(f"Unknown glacier ID: {glacier_id}")
        
        paths = self.config['data_paths'][glacier_id]
        modis_data = self._load_modis_data(paths['modis'], glacier_id)
        aws_data = self._load_aws_data(paths['aws'], glacier_id)
        
        logger.info(f"Loaded {glacier_id}: {len(modis_data):,} MODIS, {len(aws_data):,} AWS records")
        return modis_data, aws_data
    
    def _load_modis_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        """Load MODIS data with glacier-specific parsing."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"MODIS data file not found: {file_path}")
        
        data = pd.read_csv(file_path)
        data['date'] = pd.to_datetime(data['date'], errors='coerce')
        data = data.dropna(subset=['date'])
        
        # Process based on format
        if glacier_id == 'coropuna' and 'method' in data.columns and 'albedo' in data.columns:
            data['method'] = data['method'].map(self.config['method_mapping']).fillna(data['method'])
        elif 'method' not in data.columns:
            data = self._convert_to_long_format(data)
        else:
            data['method'] = data['method'].map(self.config['method_mapping']).fillna(data['method'])
        
        return self._validate_modis_data(data)
    
    def _convert_to_long_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert wide format MODIS data to long format."""
        method_columns = {}
        for col in data.columns:
            col_upper = col.upper()
            if 'MOD09GA' in col_upper and 'ALBEDO' in col_upper:
                method_columns['MOD09GA'] = col
            elif 'MOD10A1' in col_upper and 'ALBEDO' in col_upper:
                method_columns['MOD10A1'] = col
            elif 'MCD43A3' in col_upper and 'ALBEDO' in col_upper:
                method_columns['MCD43A3'] = col
            elif col in ['MOD09GA', 'MOD10A1', 'MCD43A3']:
                method_columns[col] = col
        
        long_format_rows = []
        for method, col_name in method_columns.items():
            if col_name not in data.columns:
                continue
                
            method_data = data[data[col_name].notna()][['pixel_id', 'date', col_name]].copy()
            if len(method_data) > 0:
                method_data['method'] = method
                method_data['albedo'] = method_data[col_name]
                method_data = method_data.drop(columns=[col_name])
                
                # Add coordinates and glacier fraction if available
                for coord_col in ['longitude', 'latitude']:
                    if coord_col in data.columns:
                        method_data[coord_col] = data.loc[method_data.index, coord_col]
                
                glacier_frac_cols = [c for c in data.columns if 'glacier_fraction' in c.lower()]
                if glacier_frac_cols:
                    method_data['glacier_fraction'] = data.loc[method_data.index, glacier_frac_cols[0]]
                
                long_format_rows.append(method_data)
        
        return pd.concat(long_format_rows, ignore_index=True) if long_format_rows else pd.DataFrame()
    
    def _validate_modis_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean MODIS data."""
        data = data[(data['albedo'] >= 0) & (data['albedo'] <= 1) & data['albedo'].notna()]
        return data.drop_duplicates().sort_values(['pixel_id', 'date']).reset_index(drop=True)
    
    def _load_aws_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        """Load AWS data with glacier-specific parsing."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"AWS data file not found: {file_path}")
        
        loaders = {
            'haig': self._load_haig_aws,
            'coropuna': self._load_coropuna_aws,
            'athabasca': self._load_athabasca_aws
        }
        
        aws_data = loaders[glacier_id](file_path)
        return self._validate_aws_data(aws_data)
    
    def _load_haig_aws(self, file_path: str) -> pd.DataFrame:
        """Load Haig AWS data with special format handling."""
        aws_data = pd.read_csv(file_path, sep=';', skiprows=6, decimal=',')
        aws_data.columns = aws_data.columns.str.strip()
        aws_data = aws_data.dropna(subset=['Year', 'Day'])
        aws_data['Year'] = aws_data['Year'].astype(int)
        aws_data['Day'] = aws_data['Day'].astype(int)
        aws_data['date'] = pd.to_datetime(aws_data['Year'].astype(str) + '-01-01') + pd.to_timedelta(aws_data['Day'] - 1, unit='D')
        
        albedo_cols = [col for col in aws_data.columns if 'albedo' in col.lower()]
        if not albedo_cols:
            raise ValueError("No albedo column found in Haig AWS data")
        aws_data['Albedo'] = pd.to_numeric(aws_data[albedo_cols[0]], errors='coerce')
        return aws_data
    
    def _load_coropuna_aws(self, file_path: str) -> pd.DataFrame:
        """Load Coropuna AWS data."""
        aws_data = pd.read_csv(file_path)
        aws_data['date'] = pd.to_datetime(aws_data['Timestamp'])
        return aws_data
    
    def _load_athabasca_aws(self, file_path: str) -> pd.DataFrame:
        """Load Athabasca AWS data."""
        aws_data = pd.read_csv(file_path)
        aws_data['date'] = pd.to_datetime(aws_data['Time'])
        return aws_data
    
    def _validate_aws_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean AWS data."""
        data = data[['date', 'Albedo']].copy()
        data = data.dropna(subset=['Albedo'])
        data = data[(data['Albedo'] > 0) & (data['Albedo'] <= 1)]
        return data.drop_duplicates().sort_values('date').reset_index(drop=True)



class PixelSelector:
    """Implements intelligent pixel selection based on proximity to AWS stations and quality metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def select_best_pixels(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Select best pixels for AWS-MODIS comparison analysis."""
        if len(modis_data) == 0:
            return modis_data
        
        # Calculate pixel metrics and apply quality filters
        pixel_summary = self._calculate_pixel_metrics(modis_data)
        quality_pixels = self._apply_quality_filters(pixel_summary, glacier_id)
        
        if len(quality_pixels) == 0:
            logger.warning(f"No quality pixels found for {glacier_id}, using all data")
            return modis_data
        
        # Calculate distances and select pixels
        quality_pixels = self._calculate_aws_distances(quality_pixels, glacier_id)
        selected_pixel_ids = self._apply_selection_strategy(quality_pixels, glacier_id)
        
        # Filter to selected pixels
        filtered_data = modis_data[modis_data['pixel_id'].isin(selected_pixel_ids)].copy()
        logger.info(f"Selected {len(selected_pixel_ids)} pixels for {glacier_id}: {len(modis_data)} → {len(filtered_data)} observations")
        
        return filtered_data
    
    def _calculate_pixel_metrics(self, modis_data: pd.DataFrame) -> pd.DataFrame:
        """Calculate quality metrics for each pixel."""
        pixel_summary = modis_data.groupby('pixel_id').agg({
            'glacier_fraction': ['mean', 'std'],
            'albedo': 'count',
            'latitude': 'first',
            'longitude': 'first',
            'date': ['min', 'max']
        }).reset_index()
        
        pixel_summary.columns = ['pixel_id', 'avg_glacier_fraction', 'std_glacier_fraction',
                                'n_observations', 'latitude', 'longitude', 'start_date', 'end_date']
        
        pixel_summary['avg_glacier_fraction'] = pixel_summary['avg_glacier_fraction'].fillna(0.5)
        pixel_summary['std_glacier_fraction'] = pixel_summary['std_glacier_fraction'].fillna(0.0)
        
        return pixel_summary
    
    def _apply_quality_filters(self, pixel_summary: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Apply quality filters to remove poor-quality pixels."""
        quality_filters = self.config['quality_filters']
        quality_pixels = pixel_summary[
            (pixel_summary['avg_glacier_fraction'] >= quality_filters['min_glacier_fraction']) &
            (pixel_summary['n_observations'] >= quality_filters['min_observations'])
        ].dropna(subset=['latitude', 'longitude']).copy()
        
        logger.info(f"Quality filtering for {glacier_id}: {len(pixel_summary)} → {len(quality_pixels)} pixels")
        return quality_pixels
    
    def _calculate_aws_distances(self, quality_pixels: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Calculate distance from each pixel to the AWS station."""
        aws_station = self.config['aws_stations'][glacier_id]
        aws_lat, aws_lon = aws_station['lat'], aws_station['lon']
        
        quality_pixels['distance_to_aws_km'] = self._haversine_distance(
            quality_pixels['latitude'], quality_pixels['longitude'], aws_lat, aws_lon
        )
        
        return quality_pixels
    
    def _apply_selection_strategy(self, quality_pixels: pd.DataFrame, glacier_id: str) -> List[str]:
        """Apply glacier-specific pixel selection strategy."""
        if glacier_id == 'athabasca':
            # Use all quality pixels for small dataset
            selected_pixel_ids = quality_pixels['pixel_id'].tolist()
        else:
            # Select best pixel using composite ranking
            ranked_pixels = self._rank_pixels_by_quality(quality_pixels)
            selected_pixel_ids = ranked_pixels.head(1)['pixel_id'].tolist()
        
        return selected_pixel_ids
    
    def _rank_pixels_by_quality(self, quality_pixels: pd.DataFrame) -> pd.DataFrame:
        """Rank pixels by composite quality score (glacier fraction + distance)."""
        pixels = quality_pixels.copy()
        
        # Normalize metrics to 0-1 scale
        pixels['glacier_fraction_norm'] = (
            (pixels['avg_glacier_fraction'] - pixels['avg_glacier_fraction'].min()) /
            (pixels['avg_glacier_fraction'].max() - pixels['avg_glacier_fraction'].min())
        ).fillna(0.5)
        
        pixels['distance_score'] = (
            (pixels['distance_to_aws_km'].max() - pixels['distance_to_aws_km']) /
            (pixels['distance_to_aws_km'].max() - pixels['distance_to_aws_km'].min())
        ).fillna(0.5)
        
        # Composite score: 60% glacier fraction, 40% distance
        pixels['composite_score'] = 0.6 * pixels['glacier_fraction_norm'] + 0.4 * pixels['distance_score']
        
        return pixels.sort_values(['composite_score', 'distance_to_aws_km'], ascending=[False, True])
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate great circle distance using Haversine formula."""
        R = 6371.0  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))


# =============================================================================
# DATA PROCESSING MODULE
# =============================================================================

class DataProcessor:
    """Handles AWS-MODIS data merging, outlier filtering, and statistical analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def merge_and_process(self, modis_data: pd.DataFrame, aws_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Merge AWS and MODIS data and calculate comprehensive statistics."""
        if len(modis_data) == 0 or len(aws_data) == 0:
            logger.warning(f"Insufficient data for {glacier_id}: MODIS={len(modis_data)}, AWS={len(aws_data)}")
            return pd.DataFrame()
        
        results = []
        
        # Process each MODIS method independently
        for method in self.config['methods']:
            try:
                method_result = self._process_single_method(modis_data, aws_data, method, glacier_id)
                if method_result is not None:
                    results.append(method_result)
                    logger.info(f"Processed {method} for {glacier_id}: {method_result['n_samples']} samples, r={method_result['correlation']:.3f}")
            except Exception as e:
                logger.error(f"Error processing {method} for {glacier_id}: {e}")
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def _process_single_method(self, modis_data: pd.DataFrame, aws_data: pd.DataFrame, method: str, glacier_id: str) -> Optional[Dict[str, Any]]:
        """Process a single MODIS method against AWS data."""
        method_data = modis_data[modis_data['method'] == method].copy()
        if len(method_data) == 0:
            return None
        
        # Merge with AWS data on date
        merged = method_data.merge(aws_data, on='date', how='inner').sort_values('date')
        if len(merged) < 3:
            return None
        
        # Apply outlier filtering
        aws_clean, modis_clean = self._apply_outlier_filtering(merged['Albedo'].values, merged['albedo'].values, method, glacier_id)
        if len(aws_clean) < 3:
            return None
        
        # Calculate statistics
        stats = self._calculate_comprehensive_statistics(aws_clean, modis_clean)
        
        return {'glacier_id': glacier_id, 'method': method, 'aws_values': aws_clean, 'modis_values': modis_clean, **stats}
    
    def _apply_outlier_filtering(self, aws_vals: np.ndarray, modis_vals: np.ndarray, method: str, glacier_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Apply statistical outlier filtering to AWS-MODIS pairs."""
        if len(aws_vals) < 3:
            return aws_vals, modis_vals
        
        residuals = modis_vals - aws_vals
        threshold = self.config['outlier_threshold'] * np.std(residuals)
        outlier_mask = np.abs(residuals - np.mean(residuals)) <= threshold
        
        return aws_vals[outlier_mask], modis_vals[outlier_mask]
    
    def _calculate_comprehensive_statistics(self, aws_vals: np.ndarray, modis_vals: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive statistics between AWS and MODIS values."""
        if len(aws_vals) == 0:
            return {'correlation': np.nan, 'r_squared': np.nan, 'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 'n_samples': 0, 'p_value': 1.0}
        
        correlation = _handle_correlation_calculation(aws_vals, modis_vals)
        r_squared = correlation**2 if not np.isnan(correlation) else np.nan
        residuals = modis_vals - aws_vals
        
        return {
            'correlation': correlation if not np.isnan(correlation) else 0.0,
            'r_squared': r_squared if not np.isnan(r_squared) else 0.0,
            'p_value': _handle_significance_calculation(aws_vals, modis_vals, correlation),
            'n_samples': len(aws_vals),
            'rmse': np.sqrt(np.mean(residuals**2)),
            'mae': np.mean(np.abs(residuals)),
            'bias': np.mean(residuals)
        }


# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

class ScatterplotVisualizer:
    """Creates publication-ready AWS vs MODIS scatterplot matrices."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def create_scatterplot_matrix(self, processed_data: List[pd.DataFrame], output_path: Optional[str] = None) -> plt.Figure:
        """Create the complete 3×3 AWS vs MODIS scatterplot matrix."""
        if not processed_data:
            return plt.figure()
        
        # Setup and create figure
        self._setup_plotting_style()
        fig, axes = plt.subplots(3, 3, figsize=self.config['visualization']['figsize'])
        fig.suptitle('AWS vs MODIS Albedo Correlations\nSelected Best Pixels: 2/1/1 (Closest to AWS Stations)', fontsize=16, fontweight='bold', y=0.98)
        
        # Populate subplots
        glaciers = ['athabasca', 'coropuna', 'haig']
        methods = self.config['methods']
        
        for i, glacier_id in enumerate(glaciers):
            glacier_data = self._get_glacier_data(processed_data, glacier_id)
            for j, method in enumerate(methods):
                self._create_subplot(axes[i, j], glacier_data, glacier_id, method)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)
        
        if output_path:
            self._save_figure(fig, output_path)
        
        return fig
    
    def _setup_plotting_style(self):
        """Setup matplotlib style and parameters."""
        try:
            plt.style.use(self.config['visualization']['style'])
        except Exception:
            pass
        plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8})
    
    def _get_glacier_data(self, processed_data: List[pd.DataFrame], glacier_id: str) -> pd.DataFrame:
        """Extract data for a specific glacier."""
        for df in processed_data:
            if not df.empty and df['glacier_id'].iloc[0] == glacier_id:
                return df
        return pd.DataFrame()
    
    def _create_subplot(self, ax: plt.Axes, glacier_data: pd.DataFrame, glacier_id: str, method: str):
        """Create a single subplot for a glacier-method combination."""
        # Basic configuration
        ax.set_title(f'{glacier_id.title()} - {method}', fontsize=12, fontweight='bold')
        ax.set_xlabel('AWS Albedo', fontsize=10)
        ax.set_ylabel('MODIS Albedo', fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        
        # Get method data
        method_data = None
        if not glacier_data.empty:
            method_rows = glacier_data[glacier_data['method'] == method]
            if not method_rows.empty:
                method_data = method_rows.iloc[0]
        
        if method_data is not None:
            # Create scatterplot with data
            aws_vals, modis_vals = method_data['aws_values'], method_data['modis_values']
            color = self.config['colors'][glacier_id]
            
            ax.scatter(aws_vals, modis_vals, alpha=0.6, s=20, color=color, zorder=3)
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=1, zorder=1)
            
            # Add trend line
            if len(aws_vals) > 1:
                try:
                    coeffs = np.polyfit(aws_vals, modis_vals, 1)
                    trend_line = np.poly1d(coeffs)
                    x_range = np.linspace(aws_vals.min(), aws_vals.max(), 100)
                    ax.plot(x_range, trend_line(x_range), 'r-', alpha=0.8, linewidth=1.5, zorder=2)
                except Exception:
                    pass
            
            # Add statistics
            stats_text = (f'R = {method_data["correlation"]:.3f}\n'
                         f'R² = {method_data["r_squared"]:.3f}\n'
                         f'RMSE = {method_data["rmse"]:.3f}\n'
                         f'MAE = {method_data["mae"]:.3f}\n'
                         f'Bias = {method_data["bias"]:.3f}\n'
                         f'n = {method_data["n_samples"]}')
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), fontsize=8, zorder=4)
        else:
            # No data placeholder
            ax.text(0.5, 0.5, 'No data available', transform=ax.transAxes, ha='center', va='center', fontsize=10, style='italic')
    
    def _save_figure(self, fig: plt.Figure, output_path: str):
        """Save figure to file."""
        try:
            fig.savefig(output_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight', facecolor='white')
            logger.info(f"Scatterplot matrix saved to: {output_path}")
        except Exception as e:
            logger.error(f"Error saving figure: {e}")


# =============================================================================
# PIPELINE ORCHESTRATION
# =============================================================================

class AnalysisPipeline:
    """Orchestrates the complete AWS vs MODIS analysis pipeline."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_loader = DataLoader(config)
        self.pixel_selector = PixelSelector(config)
        self.data_processor = DataProcessor(config)
        self.visualizer = ScatterplotVisualizer(config)
        
    def run_complete_analysis(self, output_path: Optional[str] = None) -> bool:
        """Execute the complete analysis pipeline for all glaciers."""
        logger.info("="*60)
        logger.info("STARTING AWS vs MODIS ANALYSIS PIPELINE")
        logger.info("="*60)
        
        try:
            all_processed_data = []
            for glacier_id in self.config['data_paths'].keys():
                processed_data = self._process_single_glacier(glacier_id)
                if processed_data is not None:
                    all_processed_data.append(processed_data)
            
            if all_processed_data:
                success = self._generate_visualization(all_processed_data, output_path)
                self._log_final_summary(all_processed_data)
                return success
            else:
                logger.error("No data could be processed for any glacier")
                return False
                
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            return False
    
    def _process_single_glacier(self, glacier_id: str) -> Optional[pd.DataFrame]:
        """Process a single glacier through the complete pipeline."""
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"PROCESSING {glacier_id.upper()} GLACIER")
            logger.info(f"{'='*50}")
            
            modis_data, aws_data = self.data_loader.load_glacier_data(glacier_id)
            selected_modis = self.pixel_selector.select_best_pixels(modis_data, glacier_id)
            processed = self.data_processor.merge_and_process(selected_modis, aws_data, glacier_id)
            
            if not processed.empty:
                logger.info(f"Successfully processed {glacier_id}: {len(processed)} method(s)")
                return processed
            else:
                logger.warning(f"No processed data for {glacier_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing {glacier_id}: {e}")
            return None
    
    def _generate_visualization(self, all_processed_data: List[pd.DataFrame], output_path: Optional[str]) -> bool:
        """Generate the final visualization."""
        try:
            logger.info(f"\n{'='*50}")
            logger.info("GENERATING VISUALIZATION")
            logger.info(f"{'='*50}")
            
            if output_path is None:
                output_path = "aws_vs_modis_scatterplot_matrix.png"
            
            self.visualizer.create_scatterplot_matrix(all_processed_data, output_path)
            plt.show()
            return True
            
        except Exception as e:
            logger.error(f"Error generating visualization: {e}")
            return False
    
    def _log_final_summary(self, all_processed_data: List[pd.DataFrame]):
        """Log final analysis summary."""
        logger.info(f"\n{'='*50}")
        logger.info("ANALYSIS SUMMARY")
        logger.info(f"{'='*50}")
        
        total_combinations = 0
        successful_combinations = 0
        
        for processed_data in all_processed_data:
            glacier_id = processed_data['glacier_id'].iloc[0]
            n_methods = len(processed_data)
            total_combinations += len(self.config['methods'])
            successful_combinations += n_methods
            
            logger.info(f"{glacier_id.title()}: {n_methods}/{len(self.config['methods'])} methods processed")
            for _, row in processed_data.iterrows():
                logger.info(f"  - {row['method']}: r={row['correlation']:.3f}, n={row['n_samples']}, p={row['p_value']:.3f}")
        
        success_rate = 100 * successful_combinations / total_combinations
        logger.info(f"\nOverall success rate: {successful_combinations}/{total_combinations} ({success_rate:.1f}%)")
        logger.info("Analysis pipeline completed successfully!")


# Fix bare except statements
def _handle_correlation_calculation(aws_vals: np.ndarray, modis_vals: np.ndarray) -> float:
    """Safe correlation calculation with proper exception handling."""
    if len(aws_vals) <= 1:
        return np.nan
    
    try:
        correlation_matrix = np.corrcoef(aws_vals, modis_vals)
        return correlation_matrix[0, 1]
    except (ValueError, RuntimeWarning, FloatingPointError):
        return np.nan


def _handle_significance_calculation(aws_vals: np.ndarray, modis_vals: np.ndarray, 
                                   correlation: float) -> float:
    """Safe significance calculation with proper exception handling."""
    if len(aws_vals) <= 2 or np.isnan(correlation):
        return 1.0
    
    try:
        n = len(aws_vals)
        t_stat = correlation * np.sqrt((n - 2) / (1 - correlation**2))
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
        return p_value
    except (ValueError, RuntimeWarning, FloatingPointError):
        return 1.0



# =============================================================================
# MAIN EXECUTION FUNCTIONS
# =============================================================================

def main():
    """Main execution function - runs the complete analysis pipeline."""
    logger.info("Starting AWS vs MODIS Scatterplot Matrix Generation")
    pipeline = AnalysisPipeline(CONFIG)
    success = pipeline.run_complete_analysis()
    
    if success:
        logger.info("Analysis completed successfully!")
    else:
        logger.error("Analysis failed - check logs for details")
    
    return success


if __name__ == "__main__":
    main()