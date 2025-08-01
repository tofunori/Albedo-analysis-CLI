#!/usr/bin/env python3
"""
Multi-Glacier Temporal Analysis Generator

Creates temporal time series plots showing AWS and all 3 MODIS methods over time.
Layout: 3 glaciers × 4 methods time series revealing temporal patterns and method stability.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

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
        'haig': {'lat': 50.7124, 'lon': -115.3018, 'name': 'Haig AWS'},
        'coropuna': {'lat': -15.5361, 'lon': -72.5997, 'name': 'Coropuna AWS'}
    },
    'methods': {
        'AWS': {'color': '#1f77b4', 'marker': 'o', 'label': 'AWS'},
        'MCD43A3': {'color': '#2ca02c', 'marker': '^', 'label': 'MCD43A3'},
        'MOD09GA': {'color': '#ff7f0e', 'marker': 's', 'label': 'MOD09GA'},  
        'MOD10A1': {'color': '#d62728', 'marker': 'D', 'label': 'MOD10A1'}
    },
    'method_mapping': {
        'mcd43a3': 'MCD43A3', 'MCD43A3': 'MCD43A3',
        'mod09ga': 'MOD09GA', 'MOD09GA': 'MOD09GA',
        'myd09ga': 'MOD09GA', 'MYD09GA': 'MOD09GA',
        'mod10a1': 'MOD10A1', 'MOD10A1': 'MOD10A1',
        'myd10a1': 'MOD10A1', 'MYD10A1': 'MOD10A1'
    },
    'outlier_threshold': 2.5,
    'quality_filters': {'min_glacier_fraction': 0.1, 'min_observations': 10},
    'visualization': {'figsize': (20, 12), 'dpi': 300, 'style': 'seaborn-v0_8'}
}


# ============================================================================
# DATA LOADING
# ============================================================================

class DataLoader:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_glacier_data(self, glacier_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        logger.info(f"Loading data for {glacier_id} glacier...")
        paths = self.config['data_paths'][glacier_id]
        
        modis_data = self._load_modis_data(paths['modis'], glacier_id)
        aws_data = self._load_aws_data(paths['aws'], glacier_id)
        
        logger.info(f"Loaded {len(modis_data):,} MODIS and {len(aws_data):,} AWS records")
        return modis_data, aws_data
    
    def _load_modis_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"MODIS data file not found: {file_path}")
        
        data = pd.read_csv(file_path)
        data['date'] = pd.to_datetime(data['date'])
        
        if glacier_id == 'coropuna' and 'method' in data.columns and 'albedo' in data.columns:
            data['method'] = data['method'].map(self.config['method_mapping']).fillna(data['method'])
            return data
        
        if 'method' not in data.columns:
            data = self._convert_to_long_format(data, glacier_id)
        else:
            data['method'] = data['method'].map(self.config['method_mapping']).fillna(data['method'])
        
        return data
    
    def _convert_to_long_format(self, data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        long_format_rows = []
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
            if col_name in data.columns:
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
        if not Path(file_path).exists():
            raise FileNotFoundError(f"AWS data file not found: {file_path}")
        
        if glacier_id == 'haig':
            aws_data = pd.read_csv(file_path, sep=';', skiprows=6, decimal=',')
            aws_data.columns = aws_data.columns.str.strip()
            aws_data = aws_data.dropna(subset=['Year', 'Day'])
            
            aws_data['date'] = pd.to_datetime(
                aws_data['Year'].astype(int).astype(str) + '-01-01'
            ) + pd.to_timedelta(aws_data['Day'].astype(int) - 1, unit='D')
            
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


# ============================================================================
# PIXEL SELECTION
# ============================================================================

class PixelSelector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def select_best_pixels(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        logger.info(f"Applying pixel selection for {glacier_id}...")
        
        aws_station = self.config['aws_stations'][glacier_id]
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
            quality_pixels['latitude'], quality_pixels['longitude'], 
            aws_station['lat'], aws_station['lon']
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
                           f"glacier_fraction={pixel['avg_glacier_fraction']:.3f}, "
                           f"distance={pixel['distance_to_aws']:.2f}km")
        
        filtered_data = modis_data[modis_data['pixel_id'].isin(selected_pixel_ids)].copy()
        logger.info(f"Filtered MODIS data from {len(modis_data)} to {len(filtered_data)} observations")
        return filtered_data
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        return R * 2 * np.arcsin(np.sqrt(a))


# ============================================================================
# TEMPORAL DATA PROCESSING
# ============================================================================

class TemporalDataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def process_temporal_data(self, modis_data: pd.DataFrame, aws_data: pd.DataFrame, 
                            glacier_id: str) -> pd.DataFrame:
        logger.info(f"Processing temporal data for {glacier_id}...")
        
        required_methods = ['MCD43A3', 'MOD09GA', 'MOD10A1']
        available_methods = modis_data['method'].unique()
        
        missing_methods = [m for m in required_methods if m not in available_methods]
        if missing_methods:
            logger.warning(f"Missing methods for {glacier_id}: {missing_methods}")
            required_methods = [m for m in required_methods if m in available_methods]
        
        if len(required_methods) == 0:
            logger.error(f"No required MODIS methods available for {glacier_id}")
            return pd.DataFrame()
        
        all_data = []
        
        for method in required_methods:
            method_modis = modis_data[modis_data['method'] == method].copy()
            
            if len(method_modis) == 0:
                continue
            
            merged = method_modis.merge(aws_data, on='date', how='left')
            
            if len(merged) < 3:
                continue
            
            has_both = merged.dropna(subset=['Albedo', 'albedo'])
            
            if len(has_both) >= 3:
                aws_clean, modis_clean, dates_clean = self._apply_outlier_filtering(
                    has_both['Albedo'].values, has_both['albedo'].values, has_both['date'].values
                )
                
                filtered_pairs = pd.DataFrame({
                    'date': dates_clean,
                    'aws_albedo': aws_clean,
                    'modis_albedo': modis_clean,
                    'method': method,
                    'glacier_id': glacier_id
                })
                all_data.append(filtered_pairs)
                logger.info(f"Processed {method}: {len(aws_clean)} paired samples")
            else:
                modis_only = pd.DataFrame({
                    'date': merged['date'],
                    'aws_albedo': np.nan,
                    'modis_albedo': merged['albedo'],
                    'method': method,
                    'glacier_id': glacier_id
                })
                all_data.append(modis_only)
                logger.info(f"Processed {method}: {len(merged)} MODIS-only samples")
        
        aws_reference = pd.DataFrame({
            'date': aws_data['date'],
            'aws_albedo': aws_data['Albedo'],
            'modis_albedo': aws_data['Albedo'],
            'method': 'AWS',
            'glacier_id': glacier_id
        })
        all_data.append(aws_reference)
        
        if not all_data:
            return pd.DataFrame()
        
        final_data = pd.concat(all_data, ignore_index=True)
        final_data = final_data.sort_values(['date', 'method']).reset_index(drop=True)
        
        logger.info(f"Temporal data for {glacier_id}: {len(final_data)} total observations")
        return final_data
    
    def _apply_outlier_filtering(self, aws_vals: np.ndarray, modis_vals: np.ndarray, 
                               dates: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if len(aws_vals) < 3:
            return aws_vals, modis_vals, dates
        
        residuals = modis_vals - aws_vals
        threshold = self.config['outlier_threshold'] * np.std(residuals)
        mask = np.abs(residuals - np.mean(residuals)) <= threshold
        
        return aws_vals[mask], modis_vals[mask], dates[mask]


# ============================================================================
# VISUALIZATION
# ============================================================================

class TemporalSeriesVisualizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.methods = config['methods']
        
    def create_temporal_analysis(self, temporal_data_dict: Dict[str, pd.DataFrame], 
                               output_path: Optional[str] = None) -> plt.Figure:
        logger.info("Creating temporal analysis visualization...")
        
        try:
            plt.style.use(self.config['visualization']['style'])
        except Exception:
            logger.warning("Could not set plotting style, using default")
        
        fig, axes = plt.subplots(3, 1, figsize=self.config['visualization']['figsize'])
        
        fig.suptitle('Multi-Glacier Temporal Analysis\nAWS and MODIS Albedo Time Series', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        glaciers = ['athabasca', 'haig', 'coropuna']
        glacier_names = {'athabasca': 'Athabasca Glacier', 'haig': 'Haig Glacier', 'coropuna': 'Coropuna Glacier'}
        
        for glacier_idx, glacier_id in enumerate(glaciers):
            ax = axes[glacier_idx]
            glacier_data = temporal_data_dict.get(glacier_id, pd.DataFrame())
            
            if not glacier_data.empty:
                self._create_temporal_plot(ax, glacier_data, glacier_id)
            else:
                ax.text(0.5, 0.5, 'No complete case data\navailable', transform=ax.transAxes,
                       ha='center', va='center', fontsize=14, style='italic')
            
            ax.set_ylabel(f'{glacier_names[glacier_id]}\nAlbedo', fontsize=12, fontweight='bold')
            
            if glacier_idx == 2:
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
            
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3)
        
        handles, labels = [], []
        for method, props in self.methods.items():
            handle = plt.Line2D([0], [0], color=props['color'], marker=props['marker'], 
                              linestyle='-', markersize=8, linewidth=2, label=props['label'])
            handles.append(handle)
            labels.append(props['label'])
        
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=4, fontsize=12, framealpha=0.9)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.12)
        
        if output_path:
            fig.savefig(output_path, dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight', facecolor='white')
            logger.info(f"Temporal analysis saved to: {output_path}")
        
        return fig
    
    def _create_temporal_plot(self, ax, glacier_data: pd.DataFrame, glacier_id: str):
        plot_order = ['AWS', 'MOD09GA', 'MOD10A1', 'MCD43A3']
        
        for method in plot_order:
            method_data = glacier_data[glacier_data['method'] == method]
            
            if not method_data.empty:
                method_data = method_data.sort_values('date')
                props = self.methods.get(method, {'color': 'gray', 'marker': 'o'})
                
                albedo_col = 'aws_albedo' if method == 'AWS' else 'modis_albedo'
                valid_data = method_data.dropna(subset=[albedo_col])
                
                if valid_data.empty:
                    continue
                    
                if method == 'AWS':
                    # AWS: segmented lines with gaps
                    dates_reset = valid_data['date'].reset_index(drop=True)
                    albedo_reset = valid_data[albedo_col].reset_index(drop=True)
                    
                    date_diffs = dates_reset.diff().dt.days
                    gap_indices = date_diffs[date_diffs > 10].index.tolist()
                    
                    start_idx = 0
                    for gap_idx in gap_indices + [len(dates_reset)]:
                        if gap_idx > start_idx:
                            ax.plot(dates_reset.iloc[start_idx:gap_idx], 
                                   albedo_reset.iloc[start_idx:gap_idx],
                                   color=props['color'], linestyle='-', 
                                   linewidth=1, alpha=0.5, zorder=1)
                        start_idx = gap_idx
                else:
                    # MODIS: scatter points
                    ax.scatter(valid_data['date'], valid_data[albedo_col], 
                              color=props['color'], marker=props['marker'], 
                              s=25, alpha=0.4, edgecolors='white', linewidth=0.5, zorder=2)
        
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        aws_count = len(glacier_data[glacier_data['method'] == 'AWS'])
        modis_count = len(glacier_data[glacier_data['method'] != 'AWS'])
        date_range = glacier_data['date'].agg(['min', 'max'])
        
        ax.text(0.02, 0.95, f'AWS: n={aws_count}\nMODIS: n={modis_count}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top', 
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        ax.text(0.98, 0.95, f'{date_range["min"].strftime("%Y-%m")} to {date_range["max"].strftime("%Y-%m")}', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    logger.info("Starting Multi-Glacier Temporal Analysis Generation")
    
    data_loader = DataLoader(CONFIG)
    pixel_selector = PixelSelector(CONFIG)
    data_processor = TemporalDataProcessor(CONFIG)
    visualizer = TemporalSeriesVisualizer(CONFIG)
    
    all_temporal_data = {}
    
    for glacier_id in ['athabasca', 'haig', 'coropuna']:
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {glacier_id.upper()} Glacier")
            logger.info(f"{'='*50}")
            
            modis_data, aws_data = data_loader.load_glacier_data(glacier_id)
            selected_modis = pixel_selector.select_best_pixels(modis_data, glacier_id)
            temporal_data = data_processor.process_temporal_data(selected_modis, aws_data, glacier_id)
            
            if not temporal_data.empty:
                all_temporal_data[glacier_id] = temporal_data
                logger.info(f"✅ {glacier_id} processing complete")
            else:
                logger.warning(f"⚠️  No temporal data for {glacier_id}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {glacier_id}: {e}")
            continue
    
    if all_temporal_data:
        logger.info(f"\n{'='*60}")
        logger.info("Creating Multi-Glacier Temporal Analysis Visualization")
        logger.info(f"{'='*60}")
        
        output_path = "multi_glacier_temporal_analysis.png"
        visualizer.create_temporal_analysis(all_temporal_data, output_path)
        plt.show()
        
        logger.info(f"\n✅ SUCCESS: Analysis generated and saved to {output_path}")
        logger.info(f"📊 Total glaciers processed: {len(all_temporal_data)}")
    else:
        logger.error("❌ No temporal data could be processed for any glacier")


if __name__ == "__main__":
    main()
    
    # Initialize components
    data_loader = DataLoader(CONFIG)
    pixel_selector = PixelSelector(CONFIG)
    data_processor = TemporalDataProcessor(CONFIG)
    visualizer = TemporalSeriesVisualizer(CONFIG)
    
    # Process each glacier
    all_temporal_data = {}
    
    for glacier_id in ['athabasca', 'haig', 'coropuna']:
        try:
            logger.info(f"\\n{'='*50}")
            logger.info(f"Processing {glacier_id.upper()} Glacier")
            logger.info(f"{'='*50}")
            
            # Load data
            modis_data, aws_data = data_loader.load_glacier_data(glacier_id)
            
            # Apply pixel selection
            selected_modis = pixel_selector.select_best_pixels(modis_data, glacier_id)
            
            # Process temporal data
            temporal_data = data_processor.process_temporal_data(selected_modis, aws_data, glacier_id)
            
            if not temporal_data.empty:
                all_temporal_data[glacier_id] = temporal_data
                
                # Log summary
                methods_available = temporal_data['method'].unique()
                date_range = temporal_data['date'].agg(['min', 'max'])
                complete_dates = len(temporal_data[temporal_data['method'] == 'AWS'])
                
                logger.info(f"  Methods: {', '.join(methods_available)}")
                logger.info(f"  Date range: {date_range['min'].strftime('%Y-%m-%d')} to {date_range['max'].strftime('%Y-%m-%d')}")
                logger.info(f"  Complete case dates: {complete_dates}")
                
                logger.info(f"✅ Successfully processed {glacier_id}")
            else:
                logger.warning(f"⚠️  No temporal data for {glacier_id}")
                
        except Exception as e:
            logger.error(f"❌ Error processing {glacier_id}: {e}")
            continue
    
    # Create visualization
    if all_temporal_data:
        logger.info(f"\\n{'='*60}")
        logger.info("Creating Multi-Glacier Temporal Analysis Visualization")
        logger.info(f"{'='*60}")
        
        # Generate output filename
        output_path = "multi_glacier_temporal_analysis.png"
        
        # Create the plot
        fig = visualizer.create_temporal_analysis(all_temporal_data, output_path)
        
        # Show the plot
        plt.show()
        
        logger.info(f"\\n✅ SUCCESS: Temporal analysis generated and saved to {output_path}")
        logger.info(f"📊 Total glaciers processed: {len(all_temporal_data)}")
        
    else:
        logger.error("❌ No temporal data could be processed for any glacier")


if __name__ == "__main__":
    main()