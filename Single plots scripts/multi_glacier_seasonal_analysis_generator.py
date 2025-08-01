#!/usr/bin/env python3
"""
Multi-Glacier Seasonal Analysis Generator

Creates 3×4 seasonal boxplot analysis showing AWS vs MODIS method comparisons 
across summer months (June-September) for all three glaciers.
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
    'colors': {
        'AWS': '#d62728', 'MCD43A3': '#2ca02c', 'MOD09GA': '#1f77b4', 
        'MYD09GA': '#1f77b4', 'MOD10A1': '#ff7f0e', 'MYD10A1': '#ff7f0e',
        'mcd43a3': '#2ca02c', 'mod09ga': '#1f77b4', 'myd09ga': '#1f77b4',
        'mod10a1': '#ff7f0e', 'myd10a1': '#ff7f0e'
    },
    'methods': ['AWS', 'MCD43A3', 'MOD09GA', 'MOD10A1'],
    'method_mapping': {
        'mcd43a3': 'MCD43A3', 'MCD43A3': 'MCD43A3', 'mod09ga': 'MOD09GA', 
        'MOD09GA': 'MOD09GA', 'myd09ga': 'MOD09GA', 'MYD09GA': 'MOD09GA',
        'mod10a1': 'MOD10A1', 'MOD10A1': 'MOD10A1', 'myd10a1': 'MOD10A1', 'MYD10A1': 'MOD10A1'
    },
    'seasonal_months': [6, 7, 8, 9],
    'month_names': {6: 'June', 7: 'July', 8: 'August', 9: 'September'},
    'outlier_threshold': 2.5,
    'quality_filters': {'min_glacier_fraction': 0.1, 'min_observations': 10},
    'visualization': {'figsize': (20, 15), 'dpi': 300, 'style': 'seaborn-v0_8'}
}


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

class DataLoader:
    """Handles loading and preprocessing of MODIS and AWS data for all glaciers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_glacier_data(self, glacier_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading data for {glacier_id} glacier...")
        
        paths = self.config['data_paths'][glacier_id]
        modis_data = self._load_modis_data(paths['modis'], glacier_id)
        aws_data = self._load_aws_data(paths['aws'], glacier_id)
        
        logger.info(f"Loaded {len(modis_data):,} MODIS and {len(aws_data):,} AWS records for {glacier_id}")
        return modis_data, aws_data
    
    def _load_modis_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"MODIS data file not found: {file_path}")
        
        logger.info(f"Loading MODIS data from: {file_path}")
        data = pd.read_csv(file_path)
        data['date'] = pd.to_datetime(data['date'])
        
        # Glacier-specific processing
        if glacier_id == 'coropuna' and 'method' in data.columns and 'albedo' in data.columns:
            logger.info("Coropuna data is in long format")
            data['method'] = data['method'].map(self.config['method_mapping']).fillna(data['method'])
            return data
        
        # Convert to long format if needed
        if 'method' not in data.columns:
            logger.info(f"Converting {glacier_id} data to long format")
            data = self._convert_to_long_format(data, glacier_id)
        else:
            logger.info(f"{glacier_id} data already in long format")
            data['method'] = data['method'].map(self.config['method_mapping']).fillna(data['method'])
        
        return data
    
    def _convert_to_long_format(self, data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        long_format_rows = []
        
        # Map columns to standard methods
        method_columns = {}
        for col in data.columns:
            if any(method in col and 'albedo' in col for method in ['MOD09GA', 'MYD09GA', 'MOD10A1', 'MYD10A1', 'MCD43A3']):
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
                
                # Add coordinates and glacier fraction if available
                for coord_col in ['longitude', 'latitude']:
                    if coord_col in data.columns:
                        method_data[coord_col] = data.loc[method_data.index, coord_col]
                
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
        if not Path(file_path).exists():
            raise FileNotFoundError(f"AWS data file not found: {file_path}")
        
        logger.info(f"Loading AWS data from: {file_path}")
        
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
        
        # Clean and validate data
        aws_data = aws_data[['date', 'Albedo']].copy()
        aws_data = aws_data.dropna(subset=['Albedo'])
        aws_data = aws_data[aws_data['Albedo'] > 0]
        aws_data = aws_data.drop_duplicates().sort_values('date').reset_index(drop=True)
        
        return aws_data


# =============================================================================
# PIXEL SELECTION
# =============================================================================

class PixelSelector:
    """Implements intelligent pixel selection based on distance to AWS stations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def select_best_pixels(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        logger.info(f"Applying pixel selection for {glacier_id}...")
        
        aws_station = self.config['aws_stations'][glacier_id]
        aws_lat, aws_lon = aws_station['lat'], aws_station['lon']
        
        # Get pixel quality metrics
        pixel_summary = modis_data.groupby('pixel_id').agg({
            'glacier_fraction': 'mean', 'albedo': 'count', 
            'latitude': 'first', 'longitude': 'first'
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
        
        # Calculate distance to AWS station
        quality_pixels['distance_to_aws'] = self._haversine_distance(
            quality_pixels['latitude'], quality_pixels['longitude'], aws_lat, aws_lon
        )
        
        # Pixel selection strategy
        if glacier_id == 'athabasca':
            selected_pixel_ids = quality_pixels['pixel_id'].tolist()
            logger.info(f"Using all {len(selected_pixel_ids)} pixels for {glacier_id} (small dataset)")
        else:
            quality_pixels = quality_pixels.sort_values(['avg_glacier_fraction', 'distance_to_aws'], ascending=[False, True])
            selected_pixels = quality_pixels.head(1)
            selected_pixel_ids = selected_pixels['pixel_id'].tolist()
            
            logger.info(f"Selected {len(selected_pixel_ids)} best pixel(s) for {glacier_id}")
            for _, pixel in selected_pixels.iterrows():
                logger.info(f"  Pixel {pixel['pixel_id']}: glacier_fraction={pixel['avg_glacier_fraction']:.3f}, "
                           f"distance={pixel['distance_to_aws']:.2f}km, observations={pixel['n_observations']}")
        
        # Filter MODIS data
        filtered_data = modis_data[modis_data['pixel_id'].isin(selected_pixel_ids)].copy()
        logger.info(f"Filtered MODIS data from {len(modis_data)} to {len(filtered_data)} observations")
        
        return filtered_data
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c


# =============================================================================
# SEASONAL DATA PROCESSING
# =============================================================================

class SeasonalDataProcessor:
    """Handles AWS-MODIS data merging and seasonal filtering."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def process_seasonal_data(self, modis_data: pd.DataFrame, aws_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        logger.info(f"Processing seasonal data for {glacier_id}...")
        
        # Add month columns and filter for seasonal months
        modis_data['month'] = modis_data['date'].dt.month
        aws_data['month'] = aws_data['date'].dt.month
        
        seasonal_months = self.config['seasonal_months']
        modis_seasonal = modis_data[modis_data['month'].isin(seasonal_months)].copy()
        aws_seasonal = aws_data[aws_data['month'].isin(seasonal_months)].copy()
        
        logger.info(f"Filtered to seasonal months {seasonal_months}: {len(modis_seasonal)} MODIS, {len(aws_seasonal)} AWS records")
        
        results = []
        available_methods = modis_seasonal['method'].unique()
        
        # Process each method
        for method in available_methods:
            method_data = modis_seasonal[modis_seasonal['method'] == method].copy()
            
            if len(method_data) == 0:
                logger.warning(f"No seasonal {method} data found for {glacier_id}")
                continue
            
            merged = method_data.merge(aws_seasonal, on='date', how='inner')
            
            if len(merged) < 3:
                logger.warning(f"Insufficient seasonal {method} data for {glacier_id}: {len(merged)} points")
                continue
            
            # Apply outlier filtering
            aws_clean, modis_clean, dates_clean = self._apply_outlier_filtering(
                merged['Albedo'].values, merged['albedo'].values, merged['date'].values
            )
            
            if len(aws_clean) < 3:
                logger.warning(f"Insufficient {method} data after outlier filtering for {glacier_id}")
                continue
            
            # Create clean dataset
            clean_data = pd.DataFrame({
                'date': dates_clean, 'aws_albedo': aws_clean, 'modis_albedo': modis_clean,
                'method': method, 'glacier_id': glacier_id
            })
            clean_data['month'] = pd.to_datetime(clean_data['date']).dt.month
            
            results.append(clean_data)
            logger.info(f"Processed {method} for {glacier_id}: {len(aws_clean)} clean samples")
        
        # Add AWS reference data
        aws_reference = aws_seasonal.copy()
        aws_reference['method'] = 'AWS'
        aws_reference['glacier_id'] = glacier_id
        aws_reference = aws_reference.rename(columns={'Albedo': 'aws_albedo'})
        aws_reference['modis_albedo'] = aws_reference['aws_albedo']
        
        if not aws_reference.empty:
            results.append(aws_reference[['date', 'aws_albedo', 'modis_albedo', 'method', 'glacier_id', 'month']])
        
        return pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    
    def _apply_outlier_filtering(self, aws_vals: np.ndarray, modis_vals: np.ndarray, dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

class SeasonalAnalysisVisualizer:
    """Creates the 3×4 seasonal analysis visualization."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.colors = config['colors']
        
    def create_seasonal_analysis(self, seasonal_data_dict: Dict[str, pd.DataFrame], output_path: Optional[str] = None) -> plt.Figure:
        logger.info("Creating 3×4 seasonal analysis visualization...")
        
        # Set matplotlib style
        try:
            plt.style.use(self.config['visualization']['style'])
        except Exception:
            logger.warning("Could not set plotting style, using default")
        
        # Create 3×4 subplot layout
        fig, axes = plt.subplots(3, 4, figsize=self.config['visualization']['figsize'])
        fig.suptitle('Multi-Glacier Seasonal Analysis\nAWS vs MODIS Method Comparison (June-September)', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        glaciers = ['athabasca', 'haig', 'coropuna']
        months = self.config['seasonal_months']
        month_names = self.config['month_names']
        
        # Create plots for each glacier-month combination
        for glacier_idx, glacier_id in enumerate(glaciers):
            glacier_data = seasonal_data_dict.get(glacier_id, pd.DataFrame())
            
            for month_idx, month in enumerate(months):
                ax = axes[glacier_idx, month_idx]
                
                if not glacier_data.empty:
                    month_data = glacier_data[glacier_data['month'] == month]
                    if not month_data.empty:
                        self._create_monthly_boxplots(ax, month_data, glacier_id, month)
                    else:
                        ax.text(0.5, 0.5, 'No data\navailable', transform=ax.transAxes, ha='center', va='center', fontsize=12, style='italic')
                else:
                    ax.text(0.5, 0.5, 'No data\navailable', transform=ax.transAxes, ha='center', va='center', fontsize=12, style='italic')
                
                # Set subplot labels
                if glacier_idx == 0:
                    ax.set_title(month_names[month], fontsize=14, fontweight='bold')
                
                if month_idx == 0:
                    ax.set_ylabel(f'{glacier_id.title()} Glacier\nAlbedo', fontsize=12, fontweight='bold')
                else:
                    ax.set_ylabel('Albedo', fontsize=10)
                
                ax.set_ylim(0.0, 1.0)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.88)
        
        if output_path:
            fig.savefig(output_path, dpi=self.config['visualization']['dpi'], bbox_inches='tight', facecolor='white')
            logger.info(f"Seasonal analysis saved to: {output_path}")
        
        return fig
    
    def _create_monthly_boxplots(self, ax, month_data: pd.DataFrame, glacier_id: str, month: int):
        methods = self.config['methods']
        available_methods = []
        boxplot_data = []
        colors_to_use = []
        sample_sizes = []
        
        # Prepare data for each method
        for method in methods:
            method_subset = month_data[month_data['method'] == method]
            
            if not method_subset.empty:
                values = method_subset['aws_albedo'].dropna() if method == 'AWS' else method_subset['modis_albedo'].dropna()
                
                if len(values) > 0:
                    available_methods.append(method)
                    boxplot_data.append(values)
                    colors_to_use.append(self.colors[method])
                    sample_sizes.append(len(values))
        
        if not boxplot_data:
            ax.text(0.5, 0.5, 'No data\navailable', transform=ax.transAxes, ha='center', va='center', fontsize=10, style='italic')
            return
        
        # Create and style boxplots
        box_plots = ax.boxplot(boxplot_data, patch_artist=True, 
                              labels=[f'{method}\n(n={n})' for method, n in zip(available_methods, sample_sizes)])
        
        for patch, color in zip(box_plots['boxes'], colors_to_use):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        for element in ['whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box_plots[element], color='black')
        
        ax.tick_params(axis='x', rotation=45)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    logger.info("Starting Multi-Glacier Seasonal Analysis Generation")
    
    # Initialize components
    data_loader = DataLoader(CONFIG)
    pixel_selector = PixelSelector(CONFIG)
    data_processor = SeasonalDataProcessor(CONFIG)
    visualizer = SeasonalAnalysisVisualizer(CONFIG)
    
    # Process each glacier
    all_seasonal_data = {}
    
    for glacier_id in ['athabasca', 'haig', 'coropuna']:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {glacier_id.upper()} Glacier")
            logger.info(f"{'='*50}")
            
            # Data processing pipeline
            modis_data, aws_data = data_loader.load_glacier_data(glacier_id)
            selected_modis = pixel_selector.select_best_pixels(modis_data, glacier_id)
            seasonal_data = data_processor.process_seasonal_data(selected_modis, aws_data, glacier_id)
            
            if not seasonal_data.empty:
                all_seasonal_data[glacier_id] = seasonal_data
                
                # Log monthly summary
                for month in CONFIG['seasonal_months']:
                    month_data = seasonal_data[seasonal_data['month'] == month]
                    methods_available = month_data['method'].unique()
                    logger.info(f"  {CONFIG['month_names'][month]}: {len(methods_available)} methods, {len(month_data)} total observations")
                
                logger.info(f"✅ Successfully processed {glacier_id}")
            else:
                logger.warning(f"⚠️  No seasonal data for {glacier_id}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {glacier_id}: {e}")
            continue
    
    # Create visualization
    if all_seasonal_data:
        logger.info(f"\n{'='*60}")
        logger.info("Creating Multi-Glacier Seasonal Analysis Visualization")
        logger.info(f"{'='*60}")
        
        output_path = "multi_glacier_seasonal_analysis.png"
        visualizer.create_seasonal_analysis(all_seasonal_data, output_path)
        
        plt.show()
        
        logger.info(f"\n✅ SUCCESS: Seasonal analysis generated and saved to {output_path}")
        logger.info(f"📊 Total glaciers processed: {len(all_seasonal_data)}")
        
    else:
        logger.error("❌ No seasonal data could be processed for any glacier")


if __name__ == "__main__":
    main()