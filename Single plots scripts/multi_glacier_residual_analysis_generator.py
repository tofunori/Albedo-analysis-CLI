#!/usr/bin/env python3
"""
Multi-Glacier Residual Analysis Generator

Creates 3×3 residual analysis showing AWS vs MODIS method error patterns
across all three glaciers.
"""

# =============================================================================
# IMPORTS
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================
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
        'athabasca': {'lat': 52.1949, 'lon': -117.2431},
        'haig': {'lat': 50.7124, 'lon': -115.3018},
        'coropuna': {'lat': -15.5361, 'lon': -72.5997}
    },
    'colors': {'MCD43A3': '#2ca02c', 'MOD09GA': '#1f77b4', 'MOD10A1': '#ff7f0e'},
    'methods': ['MCD43A3', 'MOD09GA', 'MOD10A1'],
    'method_mapping': {
        'mcd43a3': 'MCD43A3', 'MCD43A3': 'MCD43A3', 'mod09ga': 'MOD09GA', 
        'MOD09GA': 'MOD09GA', 'myd09ga': 'MOD09GA', 'MYD09GA': 'MOD09GA',
        'mod10a1': 'MOD10A1', 'MOD10A1': 'MOD10A1', 'myd10a1': 'MOD10A1', 'MYD10A1': 'MOD10A1'
    },
    'outlier_threshold': 2.5,
    'quality_filters': {'min_glacier_fraction': 0.1, 'min_observations': 10},
    'visualization': {'figsize': (18, 12), 'dpi': 300, 'style': 'seaborn-v0_8'}
}


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


class DataLoader:
    """Loads and preprocesses MODIS and AWS data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_glacier_data(self, glacier_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load MODIS and AWS data for a specific glacier."""
        logger.info(f"Loading data for {glacier_id}...")
        
        paths = self.config['data_paths'][glacier_id]
        modis_data = self._load_modis_data(paths['modis'], glacier_id)
        aws_data = self._load_aws_data(paths['aws'], glacier_id)
        
        logger.info(f"Loaded {len(modis_data):,} MODIS and {len(aws_data):,} AWS records")
        return modis_data, aws_data
    
    def _load_modis_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        """Load MODIS data with glacier-specific parsing."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"MODIS data file not found: {file_path}")
        
        data = pd.read_csv(file_path)
        data['date'] = pd.to_datetime(data['date'])
        
        if glacier_id == 'coropuna' and 'method' in data.columns:
            data['method'] = data['method'].map(self.config['method_mapping']).fillna(data['method'])
            return data
        
        if 'method' not in data.columns:
            data = self._convert_to_long_format(data)
        else:
            data['method'] = data['method'].map(self.config['method_mapping']).fillna(data['method'])
        
        return data
    
    def _convert_to_long_format(self, data: pd.DataFrame) -> pd.DataFrame:
        """Convert wide format MODIS data to long format."""
        long_format_rows = []
        method_columns = {}
        
        for col in data.columns:
            if any(method in col for method in ['MOD09GA', 'MYD09GA', 'MOD10A1', 'MYD10A1', 'MCD43A3']):
                if 'albedo' in col:
                    if 'MOD09GA' in col or 'MYD09GA' in col:
                        method_columns['MOD09GA'] = col
                    elif 'MOD10A1' in col or 'MYD10A1' in col:
                        method_columns['MOD10A1'] = col
                    elif 'MCD43A3' in col:
                        method_columns['MCD43A3'] = col
            elif col in ['MOD09GA', 'MYD09GA', 'MOD10A1', 'MYD10A1', 'MCD43A3']:
                standard_method = self.config['method_mapping'].get(col, col)
                method_columns[standard_method] = col
        
        for method, col_name in method_columns.items():
            if col_name not in data.columns:
                continue
                
            method_data = data[data[col_name].notna()][['pixel_id', 'date', col_name]].copy()
            
            if len(method_data) > 0:
                method_data['method'] = method
                method_data['albedo'] = method_data[col_name]
                method_data = method_data.drop(columns=[col_name])
                
                for coord_col in ['longitude', 'latitude']:
                    if coord_col in data.columns:
                        method_data[coord_col] = data.loc[method_data.index, coord_col]
                
                glacier_frac_cols = [c for c in data.columns if 'glacier_fraction' in c.lower()]
                if glacier_frac_cols:
                    method_data['glacier_fraction'] = data.loc[method_data.index, glacier_frac_cols[0]]
                
                long_format_rows.append(method_data)
        
        return pd.concat(long_format_rows, ignore_index=True) if long_format_rows else pd.DataFrame()
    
    def _load_aws_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        """Load AWS data with glacier-specific parsing."""
        if not Path(file_path).exists():
            raise FileNotFoundError(f"AWS data file not found: {file_path}")
        
        if glacier_id == 'haig':
            aws_data = pd.read_csv(file_path, sep=';', skiprows=6, decimal=',')
            aws_data.columns = aws_data.columns.str.strip()
            aws_data = aws_data.dropna(subset=['Year', 'Day'])
            aws_data['Year'] = aws_data['Year'].astype(int)
            aws_data['Day'] = aws_data['Day'].astype(int)
            aws_data['date'] = pd.to_datetime(aws_data['Year'].astype(str) + '-01-01') + pd.to_timedelta(aws_data['Day'] - 1, unit='D')
            
            albedo_cols = [col for col in aws_data.columns if 'albedo' in col.lower()]
            if albedo_cols:
                aws_data['Albedo'] = pd.to_numeric(aws_data[albedo_cols[0]], errors='coerce')
            else:
                raise ValueError("No albedo column found in Haig AWS data")
                
        elif glacier_id == 'coropuna':
            aws_data = pd.read_csv(file_path)
            aws_data['date'] = pd.to_datetime(aws_data['Timestamp'])
            
        elif glacier_id == 'athabasca':
            aws_data = pd.read_csv(file_path)
            aws_data['date'] = pd.to_datetime(aws_data['Time'])
        
        aws_data = aws_data[['date', 'Albedo']].copy()
        aws_data = aws_data.dropna(subset=['Albedo'])
        aws_data = aws_data[aws_data['Albedo'] > 0]
        return aws_data.drop_duplicates().sort_values('date').reset_index(drop=True)


# =============================================================================
# PIXEL SELECTION
# =============================================================================


class PixelSelector:
    """Selects best pixels based on distance to AWS stations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def select_best_pixels(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Select best pixels for analysis."""
        logger.info(f"Applying pixel selection for {glacier_id}...")
        
        aws_station = self.config['aws_stations'][glacier_id]
        aws_lat, aws_lon = aws_station['lat'], aws_station['lon']
        
        pixel_summary = modis_data.groupby('pixel_id').agg({
            'glacier_fraction': 'mean',
            'albedo': 'count',
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        pixel_summary.columns = ['pixel_id', 'avg_glacier_fraction', 'n_observations', 'latitude', 'longitude']
        
        quality_filters = self.config['quality_filters']
        quality_pixels = pixel_summary[
            (pixel_summary['avg_glacier_fraction'] > quality_filters['min_glacier_fraction']) & 
            (pixel_summary['n_observations'] > quality_filters['min_observations'])
        ].copy()
        
        if len(quality_pixels) == 0:
            logger.warning(f"No quality pixels found for {glacier_id}, using all data")
            return modis_data
        
        quality_pixels['distance_to_aws'] = self._haversine_distance(
            quality_pixels['latitude'], quality_pixels['longitude'], aws_lat, aws_lon
        )
        
        if glacier_id == 'athabasca':
            selected_pixel_ids = quality_pixels['pixel_id'].tolist()
            logger.info(f"Using all {len(selected_pixel_ids)} pixels for {glacier_id}")
        else:
            quality_pixels = quality_pixels.sort_values(['avg_glacier_fraction', 'distance_to_aws'], ascending=[False, True])
            selected_pixels = quality_pixels.head(1)
            selected_pixel_ids = selected_pixels['pixel_id'].tolist()
            
            for _, pixel in selected_pixels.iterrows():
                logger.info(f"Selected pixel {pixel['pixel_id']}: "
                           f"fraction={pixel['avg_glacier_fraction']:.3f}, "
                           f"distance={pixel['distance_to_aws']:.2f}km")
        
        filtered_data = modis_data[modis_data['pixel_id'].isin(selected_pixel_ids)].copy()
        logger.info(f"Filtered from {len(modis_data)} to {len(filtered_data)} observations")
        
        return filtered_data
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance using Haversine formula."""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c


# =============================================================================
# RESIDUAL DATA PROCESSING
# =============================================================================


class ResidualDataProcessor:
    """Handles AWS-MODIS data merging and residual calculations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def process_residual_data(self, modis_data: pd.DataFrame, aws_data: pd.DataFrame, 
                            glacier_id: str) -> pd.DataFrame:
        """Process and merge data to calculate residuals."""
        logger.info(f"Processing residual data for {glacier_id}...")
        
        results = []
        available_methods = modis_data['method'].unique()
        
        for method in available_methods:
            method_data = modis_data[modis_data['method'] == method].copy()
            
            if len(method_data) == 0:
                continue
            
            merged = method_data.merge(aws_data, on='date', how='inner')
            
            if len(merged) < 3:
                logger.warning(f"Insufficient {method} data for {glacier_id}: {len(merged)} points")
                continue
            
            aws_clean, modis_clean, dates_clean = self._apply_outlier_filtering(
                merged['Albedo'].values, merged['albedo'].values, merged['date'].values
            )
            
            if len(aws_clean) < 3:
                continue
            
            residuals = modis_clean - aws_clean
            
            clean_data = pd.DataFrame({
                'date': dates_clean,
                'aws_albedo': aws_clean,
                'modis_albedo': modis_clean,
                'residuals': residuals,
                'method': method,
                'glacier_id': glacier_id
            })
            
            results.append(clean_data)
            
            logger.info(f"Processed {method} for {glacier_id}: {len(aws_clean)} samples, "
                       f"mean residual = {np.mean(residuals):.4f}")
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def _apply_outlier_filtering(self, aws_vals: np.ndarray, modis_vals: np.ndarray, 
                               dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Apply 2.5σ outlier filtering to AWS-MODIS pairs."""
        if len(aws_vals) < 3:
            return aws_vals, modis_vals, dates
        
        residuals = modis_vals - aws_vals
        mean_residual = np.mean(residuals)
        std_residual = np.std(residuals)
        
        threshold = self.config['outlier_threshold'] * std_residual
        mask = np.abs(residuals - mean_residual) <= threshold
        
        return aws_vals[mask], modis_vals[mask], dates[mask]


# =============================================================================
# VISUALIZATION
# =============================================================================


class ResidualAnalysisVisualizer:
    """Creates the 3×3 residual analysis visualization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.colors = config['colors']
        
    def create_residual_analysis(self, residual_data_dict: Dict[str, pd.DataFrame], 
                               output_path: Optional[str] = None) -> plt.Figure:
        """Create the 3×3 residual analysis visualization."""
        logger.info("Creating 3×3 residual analysis visualization...")
        
        try:
            plt.style.use(self.config['visualization']['style'])
        except Exception:
            logger.warning("Could not set plotting style, using default")
        
        fig, axes = plt.subplots(3, 3, figsize=self.config['visualization']['figsize'])
        
        fig.suptitle('Multi-Glacier Residual Analysis\\nMODIS - AWS Albedo vs AWS Albedo', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        glaciers = ['athabasca', 'haig', 'coropuna']
        methods = self.config['methods']
        
        for glacier_idx, glacier_id in enumerate(glaciers):
            glacier_data = residual_data_dict.get(glacier_id, pd.DataFrame())
            
            for method_idx, method in enumerate(methods):
                ax = axes[glacier_idx, method_idx]
                
                if not glacier_data.empty:
                    method_data = glacier_data[glacier_data['method'] == method]
                    
                    if not method_data.empty:
                        self._create_residual_plot(ax, method_data, glacier_id, method)
                    else:
                        self._plot_no_data(ax)
                else:
                    self._plot_no_data(ax)
                
                self._set_subplot_labels(ax, glacier_idx, method_idx, glacier_id, method)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        
        if output_path:
            fig.savefig(output_path, dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight', facecolor='white')
            logger.info(f"Residual analysis saved to: {output_path}")
        
        return fig
    
    def _create_residual_plot(self, ax, method_data: pd.DataFrame, glacier_id: str, method: str):
        """Create residual plot for a single glacier-method combination."""
        aws_values = method_data['aws_albedo'].values
        residuals = method_data['residuals'].values
        color = self.colors.get(method, '#1f77b4')
        
        ax.scatter(aws_values, residuals, c=color, alpha=0.6, s=25, 
                  edgecolors='white', linewidth=0.5)
        
        n_samples = len(method_data)
        mean_residual = np.mean(residuals)
        
        ax.text(0.02, 0.95, f'n={n_samples}', transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.text(0.98, 0.95, f'μ={mean_residual:.3f}', transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    def _plot_no_data(self, ax):
        """Plot no data message."""
        ax.text(0.5, 0.5, 'No data\\navailable', transform=ax.transAxes,
               ha='center', va='center', fontsize=12, style='italic')
    
    def _set_subplot_labels(self, ax, glacier_idx: int, method_idx: int, glacier_id: str, method: str):
        """Set subplot labels and styling."""
        if glacier_idx == 0:
            ax.set_title(method, fontsize=12, fontweight='bold')
        
        if method_idx == 0:
            ax.set_ylabel(f'{glacier_id.title()} Glacier\\nResiduals (MODIS - AWS)', 
                         fontsize=10, fontweight='bold')
        else:
            ax.set_ylabel('Residuals (MODIS - AWS)', fontsize=9)
        
        if glacier_idx == 2:
            ax.set_xlabel('AWS Albedo', fontsize=10)
        
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.3, 0.3)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1)
        ax.grid(True, alpha=0.3)


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Main execution function."""
    logger.info("Starting Multi-Glacier Residual Analysis Generation")
    
    data_loader = DataLoader(CONFIG)
    pixel_selector = PixelSelector(CONFIG)
    data_processor = ResidualDataProcessor(CONFIG)
    visualizer = ResidualAnalysisVisualizer(CONFIG)
    
    all_residual_data = {}
    
    for glacier_id in ['athabasca', 'haig', 'coropuna']:
        try:
            logger.info(f"\\n{'='*50}")
            logger.info(f"Processing {glacier_id.upper()} Glacier")
            logger.info(f"{'='*50}")
            
            modis_data, aws_data = data_loader.load_glacier_data(glacier_id)
            selected_modis = pixel_selector.select_best_pixels(modis_data, glacier_id)
            residual_data = data_processor.process_residual_data(selected_modis, aws_data, glacier_id)
            
            if not residual_data.empty:
                all_residual_data[glacier_id] = residual_data
                
                for method in CONFIG['methods']:
                    method_data = residual_data[residual_data['method'] == method]
                    if not method_data.empty:
                        mean_residual = method_data['residuals'].mean()
                        std_residual = method_data['residuals'].std()
                        logger.info(f"  {method}: n={len(method_data)}, "
                                   f"mean_residual={mean_residual:.4f}, "
                                   f"std_residual={std_residual:.4f}")
                
                logger.info(f"✅ Successfully processed {glacier_id}")
            else:
                logger.warning(f"⚠️  No residual data for {glacier_id}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {glacier_id}: {e}")
            continue
    
    if all_residual_data:
        logger.info(f"\\n{'='*60}")
        logger.info("Creating Multi-Glacier Residual Analysis Visualization")
        logger.info(f"{'='*60}")
        
        output_path = "multi_glacier_residual_analysis.png"
        visualizer.create_residual_analysis(all_residual_data, output_path)
        plt.show()
        
        logger.info(f"\\n✅ SUCCESS: Residual analysis generated and saved to {output_path}")
        logger.info(f"📊 Total glaciers processed: {len(all_residual_data)}")
        
    else:
        logger.error("❌ No residual data could be processed for any glacier")


if __name__ == "__main__":
    main()