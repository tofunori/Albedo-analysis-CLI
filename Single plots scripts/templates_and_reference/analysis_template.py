#!/usr/bin/env python3
"""
[ANALYSIS NAME] Generator

[DETAILED DESCRIPTION OF WHAT THIS ANALYSIS DOES]
[EXPLAIN THE PURPOSE AND SCIENTIFIC CONTEXT]

Features:
- [FEATURE 1 - e.g., "3×3 correlation matrix across glaciers and methods"]
- [FEATURE 2 - e.g., "Statistical significance testing with p-values"]
- [FEATURE 3 - e.g., "Automated outlier detection and filtering"]
- [FEATURE 4 - e.g., "Publication-ready visualizations with error bars"]
- [FEATURE 5 - e.g., "Comprehensive summary statistics and documentation"]

Author: [YOUR NAME]
Date: [CURRENT DATE]
"""

# ============================================================================
# IMPORTS
# ============================================================================

# Standard library imports
import logging
from datetime import datetime
from pathlib import Path

# Scientific computing and data manipulation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# Statistical analysis
from scipy import stats

# Type hints for better code documentation
from typing import Any, Dict, List, Optional, Tuple

# Local utilities
from output_manager import OutputManager

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
    # Data file paths for all three glaciers
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
    
    # AWS station coordinates for distance calculations
    'aws_stations': {
        'athabasca': {'lat': 52.1949, 'lon': -117.2431, 'name': 'Athabasca AWS'},
        'haig': {'lat': 50.7186, 'lon': -115.3433, 'name': 'Haig AWS'},
        'coropuna': {'lat': -15.5181, 'lon': -72.6617, 'name': 'Coropuna AWS'}
    },
    
    # Color schemes for consistent visualization
    'colors': {
        # Glacier-specific colors
        'athabasca': '#1f77b4',    # Blue
        'haig': '#ff7f0e',         # Orange  
        'coropuna': '#2ca02c',     # Green
        
        # Method-specific colors
        'MOD09GA': '#9467bd',      # Purple (Terra)
        'MYD09GA': '#17becf',      # Cyan (Aqua)
        'MCD43A3': '#d62728',      # Red
        'MOD10A1': '#8c564b',      # Brown (Terra)
        'MYD10A1': '#e377c2',      # Pink (Aqua)
        
        # AWS reference
        'AWS': '#000000'           # Black
    },
    
    # MODIS methods to analyze
    'methods': ['MCD43A3', 'MOD09GA', 'MOD10A1'],
    
    # Method name standardization mapping
    'method_mapping': {
        'mcd43a3': 'MCD43A3', 'MCD43A3': 'MCD43A3',
        'mod09ga': 'MOD09GA', 'MOD09GA': 'MOD09GA',
        'myd09ga': 'MOD09GA', 'MYD09GA': 'MOD09GA',  # Aqua grouped with Terra
        'mod10a1': 'MOD10A1', 'MOD10A1': 'MOD10A1',
        'myd10a1': 'MOD10A1', 'MYD10A1': 'MOD10A1'   # Aqua grouped with Terra
    },
    
    # Analysis parameters
    'outlier_threshold': 2.5,  # Standard deviations for outlier filtering
    
    # Quality control filters
    'quality_filters': {
        'min_glacier_fraction': 0.1,   # Minimum glacier coverage in pixel
        'min_observations': 10         # Minimum number of valid observations
    },
    
    # Visualization settings
    'visualization': {
        'figsize': (12, 10),          # Figure size (width, height)
        'dpi': 300,                   # Resolution for saved plots
        'style': 'seaborn-v0_8'       # Matplotlib style
    },
    
    # Output configuration
    'output': {
        'analysis_name': '[ANALYSIS_NAME_LOWERCASE]',  # e.g., 'correlation_analysis'
        'base_dir': 'outputs',
        'plot_filename': '[PLOT_FILENAME].png',        # e.g., 'correlation_matrix.png'
        'summary_template': {
            'analysis_type': '[ANALYSIS TYPE TITLE]',  # e.g., 'Correlation Analysis'
            'description': '[DETAILED DESCRIPTION OF ANALYSIS PURPOSE AND METHODOLOGY]'
        }
    }
}

# ============================================================================
# DATA LOADING MODULE
# ============================================================================

class DataLoader:
    """Handles loading and preprocessing of MODIS and AWS data for all glaciers."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DataLoader with configuration.
        
        Args:
            config: Configuration dictionary containing data paths and settings
        """
        self.config = config
        
    def load_glacier_data(self, glacier_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess MODIS and AWS data for a specific glacier.
        
        Args:
            glacier_id: Identifier for glacier ('athabasca', 'haig', 'coropuna')
            
        Returns:
            Tuple of (modis_data, aws_data) as pandas DataFrames
            
        Raises:
            ValueError: If glacier_id is not recognized
            FileNotFoundError: If data files cannot be found
        """
        if glacier_id not in self.config['data_paths']:
            raise ValueError(f"Unknown glacier ID: {glacier_id}")
        
        logger.info(f"Loading data for {glacier_id} glacier...")
        
        paths = self.config['data_paths'][glacier_id]
        
        # Load MODIS data
        modis_data = self._load_modis_data(paths['modis'], glacier_id)
        
        # Load AWS data
        aws_data = self._load_aws_data(paths['aws'], glacier_id)
        
        logger.info(f"Loaded {len(modis_data):,} MODIS and {len(aws_data):,} AWS records for {glacier_id}")
        
        return modis_data, aws_data
    
    def _load_modis_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        """Load MODIS data with glacier-specific parsing.
        
        Args:
            file_path: Path to MODIS CSV file
            glacier_id: Glacier identifier for custom processing
            
        Returns:
            Processed MODIS data in long format
        """
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
        """Convert wide format MODIS data to long format.
        
        Args:
            data: Wide format MODIS data
            glacier_id: Glacier identifier
            
        Returns:
            Long format data with method column
        """
        long_format_rows = []
        
        # Define method mappings based on available columns
        method_columns = {}
        for col in data.columns:
            if 'MOD09GA' in col and 'albedo' in col:
                method_columns['MOD09GA'] = col
            elif 'MYD09GA' in col and 'albedo' in col:
                method_columns['MOD09GA'] = col  # Group Aqua with Terra
            elif 'MOD10A1' in col and 'albedo' in col:
                method_columns['MOD10A1'] = col
            elif 'MYD10A1' in col and 'albedo' in col:
                method_columns['MOD10A1'] = col  # Group Aqua with Terra
            elif 'MCD43A3' in col and 'albedo' in col:
                method_columns['MCD43A3'] = col
            elif col in ['MOD09GA', 'MYD09GA', 'MOD10A1', 'MYD10A1', 'MCD43A3']:
                standard_method = self.config['method_mapping'].get(col, col)
                method_columns[standard_method] = col
        
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
        """Load AWS data with glacier-specific parsing.
        
        Args:
            file_path: Path to AWS CSV file
            glacier_id: Glacier identifier for custom processing
            
        Returns:
            Processed AWS data with date and Albedo columns
        """
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

# ============================================================================
# PIXEL SELECTION MODULE
# ============================================================================

class PixelSelector:
    """Implements intelligent pixel selection based on distance to AWS stations."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize PixelSelector with configuration.
        
        Args:
            config: Configuration dictionary containing AWS stations and filters
        """
        self.config = config
        
    def select_best_pixels(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Select best pixels for analysis based on AWS distance and glacier fraction.
        
        Args:
            modis_data: MODIS data with pixel information
            glacier_id: Glacier identifier
            
        Returns:
            Filtered MODIS data with selected pixels only
        """
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
        
        # Pixel selection strategy (customize based on analysis needs)
        if glacier_id == 'athabasca':
            # For Athabasca (small dataset), use all quality pixels
            selected_pixel_ids = quality_pixels['pixel_id'].tolist()
            logger.info(f"Using all {len(selected_pixel_ids)} quality pixels for {glacier_id}")
        else:
            # Sort by glacier fraction (descending) then distance (ascending)
            quality_pixels = quality_pixels.sort_values([
                'avg_glacier_fraction', 'distance_to_aws'
            ], ascending=[False, True])
            
            # Select the best performing pixel(s) - customize number as needed
            num_pixels_to_select = 1  # Modify based on analysis requirements
            selected_pixels = quality_pixels.head(num_pixels_to_select)
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
        """Calculate distance using Haversine formula.
        
        Args:
            lat1, lon1: First point coordinates (arrays or scalars)
            lat2, lon2: Second point coordinates (scalars)
            
        Returns:
            Distance in kilometers
        """
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c

# ============================================================================
# DATA PROCESSING MODULE
# ============================================================================

class DataProcessor:
    """Handles AWS-MODIS data merging and statistical processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DataProcessor with configuration.
        
        Args:
            config: Configuration dictionary containing analysis parameters
        """
        self.config = config
        
    def merge_and_process(self, modis_data: pd.DataFrame, aws_data: pd.DataFrame, 
                         glacier_id: str) -> pd.DataFrame:
        """Merge AWS and MODIS data and calculate statistics for each method.
        
        Args:
            modis_data: Processed MODIS data
            aws_data: Processed AWS data
            glacier_id: Glacier identifier
            
        Returns:
            DataFrame with statistical results for each method
        """
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
            
            if len(merged) < 3:  # Need minimum data points for statistics
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
                       f"{len(aws_clean)} samples, r={stats['correlation']:.3f}")
        
        return pd.DataFrame(results)
    
    def _apply_outlier_filtering(self, aws_vals: np.ndarray, modis_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply 2.5σ outlier filtering to AWS-MODIS pairs.
        
        Args:
            aws_vals: AWS albedo values
            modis_vals: MODIS albedo values
            
        Returns:
            Tuple of filtered (aws_vals, modis_vals)
        """
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
        """Calculate comprehensive statistics between AWS and MODIS values.
        
        Args:
            aws_vals: AWS albedo values
            modis_vals: MODIS albedo values
            
        Returns:
            Dictionary with statistical metrics
        """
        if len(aws_vals) == 0:
            return {
                'correlation': np.nan, 'rmse': np.nan, 'mae': np.nan, 'bias': np.nan, 
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
            'correlation': correlation if not np.isnan(correlation) else 0.0,
            'rmse': rmse,
            'mae': mae,
            'bias': bias,
            'n_samples': len(aws_vals),
            'p_value': p_value
        }

# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

class AnalysisVisualizer:
    """Creates the main visualization for this analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize visualizer with configuration.
        
        Args:
            config: Configuration dictionary containing visualization settings
        """
        self.config = config
        
    def create_visualization(self, processed_data: List[pd.DataFrame], 
                           output_path: Optional[str] = None) -> plt.Figure:
        """Create the main visualization.
        
        Args:
            processed_data: List of processed data for each glacier
            output_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        logger.info("Creating visualization...")
        
        # Set matplotlib style
        try:
            plt.style.use(self.config['visualization']['style'])
        except:
            logger.warning("Could not set plotting style, using default")
        
        # Create figure - customize subplot layout as needed
        fig, axes = plt.subplots(2, 2, figsize=self.config['visualization']['figsize'])
        
        # Add main title
        fig.suptitle('[ANALYSIS TITLE]', fontsize=16, fontweight='bold')
        
        # TODO: Implement your specific visualization logic here
        # This is where you customize the plots based on your analysis
        
        # Example placeholder plots
        for i, ax in enumerate(axes.flat):
            ax.text(0.5, 0.5, f'Plot {i+1}\n[Customize this]', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, style='italic')
            ax.set_title(f'Subplot {i+1}')
        
        plt.tight_layout()
        
        # Save figure if path provided
        if output_path:
            fig.savefig(output_path, dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight', facecolor='white')
            logger.info(f"Visualization saved to: {output_path}")
        
        return fig
    
    def _create_subplot(self, ax, data, glacier_id, method):
        """Create individual subplot for specific glacier-method combination.
        
        Args:
            ax: matplotlib Axes object
            data: Data for this subplot
            glacier_id: Glacier identifier
            method: MODIS method
        """
        # TODO: Implement specific subplot creation logic
        # This is where you customize individual plots
        
        # Example placeholder
        ax.scatter([1, 2, 3], [1, 2, 3], label=f'{glacier_id} - {method}')
        ax.legend()

# ============================================================================
# SUMMARY AND DOCUMENTATION FUNCTIONS
# ============================================================================

def generate_summary_and_readme(output_manager: OutputManager, processed_data: List[pd.DataFrame]):
    """Generate summary file and README with analysis results.
    
    Args:
        output_manager: OutputManager instance for file operations
        processed_data: List of processed data for all glaciers
    """
    try:
        # Collect statistics for summary
        glacier_stats = {}
        overall_stats = {}
        
        # TODO: Customize this section based on your analysis results
        # Process your specific data structure and extract relevant statistics
        
        for data_df in processed_data:
            if not data_df.empty:
                glacier_id = data_df['glacier_id'].iloc[0]
                glacier_stats[glacier_id] = {
                    'methods_processed': len(data_df),
                    'total_observations': len(data_df),
                    # Add your specific statistics here
                }
        
        # Calculate overall statistics
        if processed_data:
            # TODO: Add your overall statistics calculations
            overall_stats = {
                'total_glaciers': len(processed_data),
                'total_methods': sum(len(df) for df in processed_data),
                # Add your specific overall metrics here
            }
        
        # Prepare summary data
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': CONFIG['output']['summary_template']['analysis_type'],
            'configuration': {
                'glaciers': list(CONFIG['data_paths'].keys()),
                'methods': CONFIG['methods'],
                'outlier_threshold': CONFIG['outlier_threshold'],
                'quality_filters': CONFIG['quality_filters']
            },
            'data_info': {
                'glaciers_processed': len(processed_data),
                'total_observations': sum(len(df) for df in processed_data),
                'methods_analyzed': CONFIG['methods']
            },
            'key_results': {
                # TODO: Add your key numerical results here
                'example_metric': 0.0  # Replace with actual metrics
            },
            'statistics': {
                'overall_metrics': overall_stats,
                'glacier_performance': glacier_stats
            }
        }
        
        # Save summary
        output_manager.save_summary(summary_data)
        
        # Generate README
        key_findings = [
            # TODO: Customize these findings based on your analysis
            f"Analyzed [ANALYSIS TYPE] for {len(CONFIG['methods'])} MODIS methods across {len(processed_data)} glaciers",
            f"Generated [VISUALIZATION TYPE] with [SPECIFIC FEATURES]",
            # Add more specific findings here
        ]
        
        # TODO: Add glacier-specific findings
        for glacier_id, stats in glacier_stats.items():
            key_findings.append(f"{glacier_id.title()}: [Add specific finding for this glacier]")
        
        output_manager.save_readme(
            analysis_description=CONFIG['output']['summary_template']['description'],
            key_findings=key_findings,
            additional_info={
                'Analysis Type': '[DETAILED ANALYSIS TYPE]',
                'Methodology': '[BRIEF METHODOLOGY DESCRIPTION]',
                'Quality Filters': f"Min glacier fraction: {CONFIG['quality_filters']['min_glacier_fraction']}, Min observations: {CONFIG['quality_filters']['min_observations']}",
                'Outlier Filtering': f"{CONFIG['outlier_threshold']}σ threshold applied",
                # Add more analysis-specific information
            }
        )
        
        logger.info("Summary and README generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating summary and README: {e}")

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def main():
    """Main execution function."""
    logger.info("Starting [ANALYSIS NAME] Generation")
    
    # Initialize OutputManager
    output_manager = OutputManager(
        CONFIG['output']['analysis_name'],
        CONFIG['output']['base_dir']
    )
    
    # Initialize components
    data_loader = DataLoader(CONFIG)
    pixel_selector = PixelSelector(CONFIG)
    data_processor = DataProcessor(CONFIG)
    visualizer = AnalysisVisualizer(CONFIG)
    
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
                logger.info(f"Successfully processed {glacier_id}: {len(processed)} records")
            else:
                logger.warning(f"No processed data for {glacier_id}")
                
        except Exception as e:
            logger.error(f"Error processing {glacier_id}: {e}")
            continue
    
    # Create visualization
    if all_processed_data:
        logger.info(f"\n{'='*50}")
        logger.info("Creating [ANALYSIS TYPE] Visualization")
        logger.info(f"{'='*50}")
        
        # Use OutputManager for plot path
        plot_path = output_manager.get_plot_path(CONFIG['output']['plot_filename'])
        
        # Create the plot
        fig = visualizer.create_visualization(all_processed_data, str(plot_path))
        output_manager.log_file_saved(plot_path, "plot")
        
        # Show the plot
        plt.show()
        
        # Generate summary and README
        generate_summary_and_readme(output_manager, all_processed_data)
        
        logger.info(f"\nSUCCESS: [Analysis type] generated and saved")
        logger.info(f"Total glaciers processed: {len(all_processed_data)}")
        
    else:
        logger.error("No data could be processed for any glacier")


if __name__ == "__main__":
    main()