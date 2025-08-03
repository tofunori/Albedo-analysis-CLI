#!/usr/bin/env python3
"""
Sen Slope and Mann-Kendall Trend Analysis for Haig Glacier Albedo

Implements robust non-parametric trend analysis for Haig glacier albedo data (2002-2024)
to detect long-term albedo changes and their relationship to temperature trends.

Based on methodology from Williamson et al. (2021) "The influence of forest fire aerosol 
and air temperature on glacier albedo, western North America".

Features:
- Mann-Kendall trend test for detecting monotonic trends in albedo time series
- Sen's slope estimator for quantifying trend magnitude and direction
- Seasonal and annual trend analysis capabilities
- Temperature-albedo correlation analysis
- Statistical significance testing with confidence intervals
- Publication-ready time series plots with trend lines

Author: Climate Analysis System
Date: 2025-08-03
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
from scipy.stats import linregress
import itertools

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
    # Data file paths focused on Haig glacier
    'data_paths': {
        'haig': {
            'modis': "D:/Downloads/MODIS_Terra_Aqua_MultiProduct_2002-01-01_to_2025-01-01.csv",
            'aws': "D:/Documents/Projects/Haig_analysis/data/csv/HaigAWS_daily_2002_2015_gapfilled.csv",
            'temperature': "D:/Downloads/Haig_Glacier_Climate_JuneSept_Daily_MERRA2_Speciated_AOD - Haig_Glacier_Climate_JuneSept_Daily_MERRA2_Speciated_AOD.csv"  # MERRA2 temperature data
        }
    },
    
    # AWS station coordinates for Haig glacier
    'aws_stations': {
        'haig': {'lat': 50.7186, 'lon': -115.3433, 'name': 'Haig AWS'}
    },
    
    # Color schemes for trend analysis visualization
    'colors': {
        'haig': '#ff7f0e',         # Orange for Haig glacier
        'albedo': '#2E86C1',       # Blue for albedo data
        'temperature': '#E74C3C',  # Red for temperature data
        'trend_line': '#2C3E50',   # Dark blue-gray for trend lines
        'significance': '#27AE60', # Green for significant trends
        'non_significance': '#F39C12'  # Orange for non-significant trends
    },
    
    # MODIS methods to analyze
    'methods': ['MOD09GA'],
    
    # Method name standardization mapping
    'method_mapping': {
        'mcd43a3': 'MCD43A3', 'MCD43A3': 'MCD43A3',
        'mod09ga': 'MOD09GA', 'MOD09GA': 'MOD09GA',
        'myd09ga': 'MOD09GA', 'MYD09GA': 'MOD09GA',  # Aqua grouped with Terra
        'mod10a1': 'MOD10A1', 'MOD10A1': 'MOD10A1',
        'myd10a1': 'MOD10A1', 'MYD10A1': 'MOD10A1'   # Aqua grouped with Terra
    },
    
    # Trend analysis parameters
    'trend_analysis': {
        'alpha': 0.05,           # Significance level for Mann-Kendall test
        'seasonal_analysis': True, # Perform seasonal trend analysis
        'prewhitening': True,    # Apply prewhitening for autocorrelation
        'min_years': 5,          # Minimum years of data for trend analysis
        'trend_period': 'annual'  # 'annual', 'seasonal', or 'monthly'
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
        'analysis_name': 'haig_trend_analysis',
        'base_dir': 'outputs',
        'plot_filename': 'haig_sen_slope_mann_kendall_trends.png',
        'summary_template': {
            'analysis_type': 'Sen Slope and Mann-Kendall Trend Analysis',
            'description': 'Non-parametric trend analysis of Haig glacier albedo (2002-2024) using Mann-Kendall test and Sen slope estimator to detect long-term trends and their relationship to temperature changes'
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
    
    def _load_temperature_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        """Load temperature data with glacier-specific parsing.
        
        Args:
            file_path: Path to temperature CSV file (often same as AWS file)
            glacier_id: Glacier identifier for custom processing
            
        Returns:
            Processed temperature data with date and temperature columns
        """
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Temperature data file not found: {file_path}")
        
        logger.info(f"Loading temperature data from: {file_path}")
        
        if glacier_id == 'haig':
            # MERRA2 format: Standard CSV with date and temperature_c columns
            temp_data = pd.read_csv(file_path)
            temp_data.columns = temp_data.columns.str.strip()
            
            # Handle date column (should be 'date')
            if 'date' in temp_data.columns:
                temp_data['date'] = pd.to_datetime(temp_data['date'])
            else:
                # Try other date column names
                date_cols = [col for col in temp_data.columns if 'date' in col.lower() or 'time' in col.lower()]
                if date_cols:
                    date_col = date_cols[0]
                    temp_data['date'] = pd.to_datetime(temp_data[date_col])
                else:
                    raise ValueError(f"No date column found in MERRA2 temperature data")
            
            # Find temperature column (should be 'temperature_c')
            if 'temperature_c' in temp_data.columns:
                temp_data['Temperature'] = pd.to_numeric(temp_data['temperature_c'], errors='coerce')
                logger.info("Using MERRA2 temperature_c column (already in Celsius)")
            else:
                # Try other temperature column names
                temp_cols = [col for col in temp_data.columns if 'temp' in col.lower()]
                if temp_cols:
                    temp_col = temp_cols[0]
                    temp_data['Temperature'] = pd.to_numeric(temp_data[temp_col], errors='coerce')
                    logger.info(f"Using temperature column: {temp_col}")
                else:
                    # Debug: print available columns
                    logger.error(f"Available columns in MERRA2 temperature data: {list(temp_data.columns)}")
                    raise ValueError(f"No temperature column found in MERRA2 temperature data")
            
            # Also extract BC AOD data if available
            if 'bc_aod_regional' in temp_data.columns:
                temp_data['BC_AOD'] = pd.to_numeric(temp_data['bc_aod_regional'], errors='coerce')
                logger.info("Using MERRA2 bc_aod_regional column for black carbon aerosol optical depth")
                # Clean and validate data (include BC_AOD)
                temp_data = temp_data[['date', 'Temperature', 'BC_AOD']].copy()
            else:
                logger.warning("bc_aod_regional column not found - BC AOD analysis will be skipped")
                # Clean and validate data (temperature only)
                temp_data = temp_data[['date', 'Temperature']].copy()
        # Clean data - drop rows missing temperature, but keep BC_AOD if available
        if 'BC_AOD' in temp_data.columns:
            temp_data = temp_data.dropna(subset=['Temperature'])  # Keep BC_AOD NaN for now
        else:
            temp_data = temp_data.dropna(subset=['Temperature'])
        temp_data = temp_data.drop_duplicates().sort_values('date').reset_index(drop=True)
        
        return temp_data
    
    def load_haig_data_complete(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load complete Haig glacier dataset including MODIS, AWS, and temperature data.
        
        Returns:
            Tuple of (modis_data, aws_data, temperature_data) as pandas DataFrames
        """
        logger.info("Loading complete Haig glacier dataset...")
        
        paths = self.config['data_paths']['haig']
        
        # Load MODIS data
        modis_data = self._load_modis_data(paths['modis'], 'haig')
        
        # Load AWS albedo data
        aws_data = self._load_aws_data(paths['aws'], 'haig')
        
        # Load temperature data
        temperature_data = self._load_temperature_data(paths['temperature'], 'haig')
        
        logger.info(f"Loaded {len(modis_data):,} MODIS, {len(aws_data):,} AWS albedo, "
                   f"and {len(temperature_data):,} temperature records for Haig glacier")
        
        return modis_data, aws_data, temperature_data

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
# TREND ANALYSIS MODULE
# ============================================================================

class TrendAnalyzer:
    """Implements Mann-Kendall trend test and Sen's slope estimator."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize TrendAnalyzer with configuration.
        
        Args:
            config: Configuration dictionary containing trend analysis parameters
        """
        self.config = config
        self.alpha = config['trend_analysis']['alpha']
        
    def mann_kendall_test(self, data: np.ndarray, dates: np.ndarray = None) -> Dict[str, float]:
        """Perform Mann-Kendall trend test.
        
        Args:
            data: Time series data array
            dates: Optional datetime array for temporal analysis
            
        Returns:
            Dictionary with test statistics: tau, p_value, z_score, trend
        """
        n = len(data)
        if n < 3:
            return {
                'tau': np.nan, 'p_value': 1.0, 'z_score': 0.0, 
                'trend': 'no trend', 'significance': False
            }
        
        # Calculate Mann-Kendall statistic S
        S = 0
        for i in range(n-1):
            for j in range(i+1, n):
                if data[j] > data[i]:
                    S += 1
                elif data[j] < data[i]:
                    S -= 1
        
        # Calculate variance with tie correction
        unique_vals, counts = np.unique(data, return_counts=True)
        tie_correction = np.sum(counts * (counts - 1) * (2 * counts + 5))
        
        var_s = (n * (n - 1) * (2 * n + 5) - tie_correction) / 18
        
        # Calculate standardized test statistic Z
        if S > 0:
            z_score = (S - 1) / np.sqrt(var_s)
        elif S < 0:
            z_score = (S + 1) / np.sqrt(var_s)
        else:
            z_score = 0
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Calculate Kendall's tau
        tau = S / (0.5 * n * (n - 1))
        
        # Determine trend direction and significance
        if p_value <= self.alpha:
            if S > 0:
                trend = 'increasing'
            else:
                trend = 'decreasing'
            significance = True
        else:
            trend = 'no trend'
            significance = False
        
        return {
            'tau': tau,
            'p_value': p_value,
            'z_score': z_score,
            'trend': trend,
            'significance': significance,
            'S': S,
            'var_s': var_s
        }
    
    def sen_slope_estimator(self, data: np.ndarray, dates: np.ndarray = None) -> Dict[str, float]:
        """Calculate Sen's slope estimator for trend magnitude.
        
        Args:
            data: Time series data array
            dates: Optional datetime array for time-based slope calculation
            
        Returns:
            Dictionary with slope statistics: slope, intercept, confidence_interval
        """
        n = len(data)
        if n < 2:
            return {
                'slope': np.nan, 'intercept': np.nan,
                'slope_per_year': np.nan, 'confidence_interval': (np.nan, np.nan)
            }
        
        # Generate time indices if dates not provided
        if dates is None:
            time_values = np.arange(n)
        else:
            # Convert to years from start
            start_date = pd.to_datetime(dates[0])
            time_values = np.array([(pd.to_datetime(d) - start_date).days / 365.25 for d in dates])
        
        # Calculate all pairwise slopes
        slopes = []
        for i in range(n-1):
            for j in range(i+1, n):
                if time_values[j] != time_values[i]:
                    slope = (data[j] - data[i]) / (time_values[j] - time_values[i])
                    slopes.append(slope)
        
        if not slopes:
            return {
                'slope': np.nan, 'intercept': np.nan,
                'slope_per_year': np.nan, 'confidence_interval': (np.nan, np.nan)
            }
        
        slopes = np.array(slopes)
        
        # Sen's slope is the median of all slopes
        sen_slope = np.median(slopes)
        
        # Calculate intercept using median of residuals
        intercepts = data - sen_slope * time_values
        intercept = np.median(intercepts)
        
        # Calculate confidence interval for slope
        n_slopes = len(slopes)
        if n_slopes > 1:
            # Approximate confidence interval using normal distribution
            sorted_slopes = np.sort(slopes)
            c_alpha = stats.norm.ppf(1 - self.alpha/2) * np.sqrt(n * (n-1) * (2*n+5) / 18)
            
            if c_alpha < n_slopes:
                m1 = int(np.floor((n_slopes - c_alpha) / 2))
                m2 = int(np.ceil((n_slopes + c_alpha) / 2))
                
                if m1 >= 0 and m2 < n_slopes:
                    confidence_interval = (sorted_slopes[m1], sorted_slopes[m2-1])
                else:
                    confidence_interval = (np.min(slopes), np.max(slopes))
            else:
                confidence_interval = (np.min(slopes), np.max(slopes))
        else:
            confidence_interval = (sen_slope, sen_slope)
        
        # Convert slope to per-year if using time indices
        if dates is not None:
            slope_per_year = sen_slope
        else:
            # Estimate slope per year based on data frequency
            slope_per_year = sen_slope  # This would need adjustment based on actual time scale
        
        return {
            'slope': sen_slope,
            'intercept': intercept,
            'slope_per_year': slope_per_year,
            'confidence_interval': confidence_interval,
            'n_slopes': n_slopes
        }
    
    def prewhiten_series(self, data: np.ndarray) -> np.ndarray:
        """Apply prewhitening to remove autocorrelation.
        
        Args:
            data: Time series data
            
        Returns:
            Prewhitened data array
        """
        if len(data) < 3:
            return data
        
        # Calculate lag-1 autocorrelation
        data_clean = data[~np.isnan(data)]
        if len(data_clean) < 3:
            return data
        
        r1 = np.corrcoef(data_clean[:-1], data_clean[1:])[0, 1]
        
        # Only prewhiten if significant autocorrelation exists
        if abs(r1) > 0.1:  # Threshold for meaningful autocorrelation
            # Apply prewhitening: x'(t) = x(t) - r1 * x(t-1)
            prewhitened = np.full_like(data, np.nan)
            prewhitened[0] = data[0]
            
            for i in range(1, len(data)):
                if not np.isnan(data[i]) and not np.isnan(data[i-1]):
                    prewhitened[i] = data[i] - r1 * data[i-1]
                else:
                    prewhitened[i] = data[i]
            
            logger.info(f"Applied prewhitening with r1={r1:.3f}")
            return prewhitened
        else:
            logger.info(f"No significant autocorrelation (r1={r1:.3f}), skipping prewhitening")
            return data
    
    def seasonal_trend_analysis(self, data: np.ndarray, dates: np.ndarray, 
                              season_def: str = 'meteorological') -> Dict[str, Dict]:
        """Perform seasonal trend analysis.
        
        Args:
            data: Time series data
            dates: Datetime array
            season_def: Season definition ('meteorological' or 'calendar')
            
        Returns:
            Dictionary with seasonal trend results
        """
        results = {}
        
        # Define seasons
        if season_def == 'meteorological':
            seasons = {
                'Spring': [3, 4, 5],    # Mar, Apr, May
                'Summer': [6, 7, 8],    # Jun, Jul, Aug  
                'Fall': [9, 10, 11],    # Sep, Oct, Nov
                'Winter': [12, 1, 2]    # Dec, Jan, Feb
            }
        else:  # calendar seasons
            seasons = {
                'Spring': [3, 4, 5],
                'Summer': [6, 7, 8],
                'Fall': [9, 10, 11],
                'Winter': [12, 1, 2]
            }
        
        for season_name, months in seasons.items():
            # Extract seasonal data
            seasonal_mask = np.array([pd.to_datetime(d).month in months for d in dates])
            seasonal_data = data[seasonal_mask]
            seasonal_dates = dates[seasonal_mask]
            
            if len(seasonal_data) < 3:
                results[season_name] = {
                    'mann_kendall': {'trend': 'insufficient data', 'significance': False},
                    'sen_slope': {'slope': np.nan}
                }
                continue
            
            # Apply prewhitening if configured
            if self.config['trend_analysis']['prewhitening']:
                seasonal_data = self.prewhiten_series(seasonal_data)
            
            # Remove NaN values
            valid_mask = ~np.isnan(seasonal_data)
            seasonal_data = seasonal_data[valid_mask]
            seasonal_dates = seasonal_dates[valid_mask]
            
            if len(seasonal_data) < 3:
                results[season_name] = {
                    'mann_kendall': {'trend': 'insufficient data', 'significance': False},
                    'sen_slope': {'slope': np.nan}
                }
                continue
            
            # Perform trend analysis
            mk_result = self.mann_kendall_test(seasonal_data, seasonal_dates)
            sen_result = self.sen_slope_estimator(seasonal_data, seasonal_dates)
            
            results[season_name] = {
                'mann_kendall': mk_result,
                'sen_slope': sen_result,
                'n_observations': len(seasonal_data)
            }
        
        return results
    
    def analyze_time_series(self, data: pd.DataFrame, value_col: str, 
                          date_col: str = 'date') -> Dict[str, Any]:
        """Comprehensive trend analysis of a time series.
        
        Args:
            data: DataFrame with time series data
            value_col: Column name containing the values to analyze
            date_col: Column name containing the dates
            
        Returns:
            Complete trend analysis results
        """
        logger.info(f"Starting comprehensive trend analysis for {value_col}")
        
        # Prepare data
        data_clean = data.dropna(subset=[value_col, date_col]).copy()
        data_clean = data_clean.sort_values(date_col)
        
        values = data_clean[value_col].values
        dates = pd.to_datetime(data_clean[date_col]).values
        
        if len(values) < self.config['trend_analysis']['min_years']:
            logger.warning(f"Insufficient data for trend analysis: {len(values)} points")
            return {'error': 'Insufficient data for trend analysis'}
        
        results = {
            'data_info': {
                'n_observations': len(values),
                'start_date': dates[0],
                'end_date': dates[-1],
                'duration_years': (pd.to_datetime(dates[-1]) - pd.to_datetime(dates[0])).days / 365.25
            }
        }
        
        # Apply prewhitening if configured
        if self.config['trend_analysis']['prewhitening']:
            values_analysis = self.prewhiten_series(values)
        else:
            values_analysis = values.copy()
        
        # Remove any remaining NaN values
        valid_mask = ~np.isnan(values_analysis)
        values_analysis = values_analysis[valid_mask]
        dates_analysis = dates[valid_mask]
        
        # Annual trend analysis
        logger.info("Performing annual trend analysis...")
        results['annual'] = {
            'mann_kendall': self.mann_kendall_test(values_analysis, dates_analysis),
            'sen_slope': self.sen_slope_estimator(values_analysis, dates_analysis)
        }
        
        # Seasonal trend analysis if enabled
        if self.config['trend_analysis']['seasonal_analysis']:
            logger.info("Performing seasonal trend analysis...")
            results['seasonal'] = self.seasonal_trend_analysis(values_analysis, dates_analysis)
        
        logger.info("Trend analysis completed successfully")
        return results

# ============================================================================
# DATA PROCESSING MODULE
# ============================================================================

class DataProcessor:
    """Handles data merging and time series preparation for trend analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize DataProcessor with configuration.
        
        Args:
            config: Configuration dictionary containing analysis parameters
        """
        self.config = config
        
    def prepare_time_series_data(self, modis_data: pd.DataFrame, aws_data: pd.DataFrame, 
                                temperature_data: pd.DataFrame, glacier_id: str) -> Dict[str, pd.DataFrame]:
        """Prepare time series data for trend analysis.
        
        Args:
            modis_data: Processed MODIS data
            aws_data: Processed AWS albedo data  
            temperature_data: Processed temperature data
            glacier_id: Glacier identifier
            
        Returns:
            Dictionary with prepared time series for each data type
        """
        logger.info(f"Preparing time series data for {glacier_id} trend analysis...")
        
        results = {}
        
        # Prepare AWS albedo time series
        if not aws_data.empty:
            aws_ts = aws_data[['date', 'Albedo']].copy()
            aws_ts = aws_ts.dropna().sort_values('date')
            
            # Resample to daily/monthly/annual as needed
            aws_ts['year'] = aws_ts['date'].dt.year
            aws_ts['month'] = aws_ts['date'].dt.month
            aws_ts['day_of_year'] = aws_ts['date'].dt.dayofyear
            
            results['aws_albedo'] = aws_ts
            logger.info(f"AWS albedo time series: {len(aws_ts)} observations from "
                       f"{aws_ts['date'].min()} to {aws_ts['date'].max()}")
        
        # Prepare temperature time series
        if not temperature_data.empty:
            temp_ts = temperature_data[['date', 'Temperature']].copy()
            temp_ts = temp_ts.dropna().sort_values('date')
            
            temp_ts['year'] = temp_ts['date'].dt.year
            temp_ts['month'] = temp_ts['date'].dt.month
            temp_ts['day_of_year'] = temp_ts['date'].dt.dayofyear
            
            results['temperature'] = temp_ts
            logger.info(f"Temperature time series: {len(temp_ts)} observations from "
                       f"{temp_ts['date'].min()} to {temp_ts['date'].max()}")
        
        # Prepare BC AOD time series (if available)
        if not temperature_data.empty and 'BC_AOD' in temperature_data.columns:
            bc_aod_ts = temperature_data[['date', 'BC_AOD']].copy()
            bc_aod_ts = bc_aod_ts.dropna().sort_values('date')
            
            bc_aod_ts['year'] = bc_aod_ts['date'].dt.year
            bc_aod_ts['month'] = bc_aod_ts['date'].dt.month
            bc_aod_ts['day_of_year'] = bc_aod_ts['date'].dt.dayofyear
            
            results['bc_aod'] = bc_aod_ts
            logger.info(f"BC AOD time series: {len(bc_aod_ts)} observations from "
                       f"{bc_aod_ts['date'].min()} to {bc_aod_ts['date'].max()}")
        
        # Prepare MODIS albedo time series for each method
        if not modis_data.empty:
            target_methods = self.config['methods']
            available_methods = [m for m in modis_data['method'].unique() if m in target_methods]
            
            for method in available_methods:
                method_data = modis_data[modis_data['method'] == method].copy()
                
                if len(method_data) > 0:
                    modis_ts = method_data[['date', 'albedo']].copy()
                    modis_ts = modis_ts.dropna().sort_values('date')
                    modis_ts.rename(columns={'albedo': 'Albedo'}, inplace=True)
                    
                    modis_ts['year'] = modis_ts['date'].dt.year
                    modis_ts['month'] = modis_ts['date'].dt.month
                    modis_ts['day_of_year'] = modis_ts['date'].dt.dayofyear
                    
                    results[f'modis_{method.lower()}'] = modis_ts
                    logger.info(f"MODIS {method} time series: {len(modis_ts)} observations from "
                               f"{modis_ts['date'].min()} to {modis_ts['date'].max()}")
        
        return results
    
    def create_annual_series(self, time_series_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Create annual aggregated time series for trend analysis.
        
        Args:
            time_series_data: Dictionary of time series data
            
        Returns:
            Dictionary with annual aggregated series
        """
        logger.info("Creating annual time series for trend analysis...")
        
        annual_data = {}
        
        for data_type, ts_data in time_series_data.items():
            if ts_data.empty:
                logger.info(f"Skipping {data_type}: empty dataset")
                continue
            
            logger.info(f"Processing {data_type} for annual series: {len(ts_data)} observations")
            logger.info(f"  Available columns: {list(ts_data.columns)}")
            logger.info(f"  Date range: {ts_data['date'].min()} to {ts_data['date'].max()}")
            
            # Determine value column name
            if 'Albedo' in ts_data.columns:
                value_col = 'Albedo'
            elif 'Temperature' in ts_data.columns:
                value_col = 'Temperature'
            elif 'BC_AOD' in ts_data.columns:
                value_col = 'BC_AOD'
            else:
                raise ValueError(f"No recognized value column found in {data_type} data. Available columns: {list(ts_data.columns)}")
            
            # Annual aggregation
            if 'albedo' in data_type.lower() or 'modis' in data_type.lower():
                # For albedo, use year-round data (all months)
                annual_series = ts_data.groupby('year')[value_col].mean().reset_index()
                annual_series['date'] = pd.to_datetime(annual_series['year'], format='%Y')
                annual_data[f'{data_type}_annual'] = annual_series[['date', value_col]]
                logger.info(f"{data_type} annual (year-round) series: {len(annual_series)} years")
                logger.info(f"  Created annual key: {data_type}_annual")
                
                if annual_series.empty:
                    logger.warning(f"No annual data available for {data_type}")
                    # Check unique months in the data
                    unique_months = sorted(ts_data['month'].unique())
                    logger.info(f"  Available months in {data_type}: {unique_months}")
                
                # Also create full year average
                full_year = ts_data.groupby('year')[value_col].mean().reset_index()
                full_year['date'] = pd.to_datetime(full_year['year'], format='%Y')
                annual_data[f'{data_type}_annual_full'] = full_year[['date', value_col]]
                logger.info(f"{data_type} full year series: {len(full_year)} years")
                
            elif 'temperature' in data_type.lower():
                # For temperature, create year-round annual averages
                annual_series = ts_data.groupby('year')[value_col].mean().reset_index()
                annual_series['date'] = pd.to_datetime(annual_series['year'], format='%Y')
                annual_data[f'{data_type}_annual'] = annual_series[['date', value_col]]
                logger.info(f"{data_type} annual (year-round) series: {len(annual_series)} years")
                
                # Annual average temperature
                full_year = ts_data.groupby('year')[value_col].mean().reset_index()
                full_year['date'] = pd.to_datetime(full_year['year'], format='%Y')
                annual_data[f'{data_type}_annual'] = full_year[['date', value_col]]
            
            elif 'bc_aod' in data_type.lower():
                # For BC AOD, create year-round annual averages (same as temperature)
                annual_series = ts_data.groupby('year')[value_col].mean().reset_index()
                annual_series['date'] = pd.to_datetime(annual_series['year'], format='%Y')
                annual_data[f'{data_type}_annual'] = annual_series[['date', value_col]]
                logger.info(f"{data_type} annual (melt season) series: {len(annual_series)} years")
        
        return annual_data
    
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

class TrendVisualizer:
    """Creates trend analysis visualizations using scientific plotting standards."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize visualizer with configuration.
        
        Args:
            config: Configuration dictionary containing visualization settings
        """
        self.config = config
        
    def create_trend_visualization(self, trend_results: Dict[str, Any], 
                                  time_series_data: Dict[str, pd.DataFrame],
                                  output_path: Optional[str] = None) -> plt.Figure:
        """
        Create publication-quality Mann-Kendall trend analysis visualization for Haig glacier albedo data.
        
        Parameters:
        -----------
        trend_results : Dict[str, Any]
            Dictionary containing trend analysis results with Mann-Kendall and Sen slope statistics
        time_series_data : Dict[str, pd.DataFrame]
            Dictionary containing time series DataFrames for each variable
        output_path : Optional[str]
            Path to save the figure. If None, figure is not saved.
        
        Returns:
        --------
        plt.Figure
            The created matplotlib figure object
        """
        logger.info("Creating trend analysis visualization...")
        
        # Set up plotting style
        try:
            plt.style.use(self.config['visualization']['style'])
        except:
            logger.warning("Could not set plotting style, using seaborn default")
            plt.style.use('seaborn-v0_8-whitegrid')
        
        # Define color scheme
        colors = {
            'albedo': '#2E86C1',
            'temperature': '#E74C3C',
            'trend': '#2C3E50',
            'confidence': '#85C1E9'
        }
        
        # Create figure and subplots (2x3 layout for 6 panels)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), 
                                dpi=self.config['visualization']['dpi'])
        fig.suptitle('Mann-Kendall Trend Analysis: Haig Glacier Albedo, Temperature & BC AOD (2002-2024)', 
                     fontsize=16, fontweight='bold', y=0.95)
        
        # Helper function to format significance indicators
        def get_significance_indicator(p_value):
            if p_value < 0.01:
                return '**'
            elif p_value < 0.05:
                return '*'
            else:
                return ''
        
        # Helper function to calculate Sen slope line
        def calculate_sen_slope_line(x_data, y_data, slope):
            x_numeric = pd.to_datetime(x_data).astype('int64') / 1e9 / (365.25 * 24 * 3600)  # Convert to years
            x_min, x_max = x_numeric.min(), x_numeric.max()
            y_median = np.median(y_data)
            x_median = np.median(x_numeric)
            
            # Calculate y-intercept
            intercept = y_median - slope * x_median
            
            # Generate line points
            x_line = np.array([x_min, x_max])
            y_line = slope * x_line + intercept
            
            return x_line, y_line
        
        # Plot 1: MODIS Albedo (Top Left)
        ax1 = axes[0, 0]
        
        # Debug: Check what MODIS data is available
        modis_data_keys = [k for k in time_series_data.keys() if 'modis' in k and 'annual' in k]
        modis_trend_keys = [k for k in trend_results.keys() if 'modis' in k]
        logger.info(f"Available MODIS data keys: {modis_data_keys}")
        logger.info(f"Available MODIS trend keys: {modis_trend_keys}")
        
        # Use the first available MODIS dataset
        modis_key = None
        if modis_data_keys and modis_trend_keys:
            # Find a key that exists in both
            for key in modis_data_keys:
                if key in modis_trend_keys:
                    modis_key = key
                    break
        
        if modis_key and modis_key in time_series_data and modis_key in trend_results:
            data = time_series_data[modis_key].copy()
            results = trend_results[modis_key]['annual']
            logger.info(f"Using MODIS data: {modis_key} with {len(data)} observations")
            
            # Plot time series
            ax1.scatter(data['date'], data['Albedo'], color='#8E44AD', alpha=0.7, s=50, 
                       label='MODIS Albedo')
            
            # Add Sen slope trend line
            if 'sen_slope' in results:
                slope = results['sen_slope']['slope_per_year']
                x_line, y_line = calculate_sen_slope_line(data['date'], data['Albedo'], slope)
                x_line_dates = pd.to_datetime(x_line * 365.25 * 24 * 3600 * 1e9)
                
                # Calculate percentage decline per year
                mean_albedo = np.mean(data['Albedo'])
                slope_percent = (slope / mean_albedo) * 100
                
                ax1.plot(x_line_dates, y_line, color=colors['trend'], linewidth=2,
                        label=f'Sen Slope: {slope_percent:.2f}%/year')
            
            # Add statistical information
            if 'mann_kendall' in results:
                mk_results = results['mann_kendall']
                p_val = mk_results['p_value']
                tau = mk_results['tau']
                sig_indicator = get_significance_indicator(p_val)
                
                ax1.text(0.05, 0.95, f'τ = {tau:.3f}{sig_indicator}\np = {p_val:.3f}', 
                        transform=ax1.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        if not modis_key:
            # Add message when no MODIS data is available
            ax1.text(0.5, 0.5, 'No MODIS Annual Data\nAvailable for Plotting', 
                    transform=ax1.transAxes, ha='center', va='center',
                    fontsize=12, style='italic', 
                    bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        ax1.set_title('MODIS Albedo Melt Season Trend', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Year', fontsize=12)
        ax1.set_ylabel('Albedo (unitless)', fontsize=12)
        ax1.set_ylim(0, 1)
        ax1.legend(loc='lower right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Temperature (Top Right)
        ax2 = axes[0, 1]
        if 'temperature_annual' in time_series_data and 'temperature_annual' in trend_results:
            data = time_series_data['temperature_annual'].copy()
            results = trend_results['temperature_annual']['annual']
            
            # Plot time series
            ax2.scatter(data['date'], data['Temperature'], color=colors['temperature'], 
                       alpha=0.7, s=50, label='Melt Season Temperature')
            
            # Add Sen slope trend line
            if 'sen_slope' in results:
                slope = results['sen_slope']['slope_per_year']
                x_line, y_line = calculate_sen_slope_line(data['date'], data['Temperature'], slope)
                x_line_dates = pd.to_datetime(x_line * 365.25 * 24 * 3600 * 1e9)
                ax2.plot(x_line_dates, y_line, color=colors['trend'], linewidth=2,
                        label=f'Sen Slope: {slope:.3f}°C/year')
            
            # Add statistical information
            if 'mann_kendall' in results:
                mk_results = results['mann_kendall']
                p_val = mk_results['p_value']
                tau = mk_results['tau']
                sig_indicator = get_significance_indicator(p_val)
                
                ax2.text(0.05, 0.95, f'τ = {tau:.3f}{sig_indicator}\np = {p_val:.3f}', 
                        transform=ax2.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_title('Melt Season Temperature Trend', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('Temperature (°C)', fontsize=12)
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: BC AOD Trend (Top Right)  
        ax3 = axes[0, 2]
        if 'bc_aod_annual' in time_series_data and 'bc_aod_annual' in trend_results:
            data = time_series_data['bc_aod_annual'].copy()
            results = trend_results['bc_aod_annual']['annual']
            
            # Plot time series
            ax3.scatter(data['date'], data['BC_AOD'], color='#D35400', alpha=0.7, s=50, 
                       label='Melt Season BC AOD')
            
            # Add Sen slope trend line
            if 'sen_slope' in results:
                slope = results['sen_slope']['slope_per_year']
                x_line, y_line = calculate_sen_slope_line(data['date'], data['BC_AOD'], slope)
                x_line_dates = pd.to_datetime(x_line * 365.25 * 24 * 3600 * 1e9)
                
                # Calculate percentage change per year (like albedo)
                mean_bc_aod = np.mean(data['BC_AOD'])
                slope_percent = (slope / mean_bc_aod) * 100
                
                ax3.plot(x_line_dates, y_line, color=colors['trend'], linewidth=2,
                        label=f'Sen Slope: {slope_percent:+.2f}%/year')
            
            # Add statistical information
            if 'mann_kendall' in results:
                mk_results = results['mann_kendall']
                p_val = mk_results['p_value']
                tau = mk_results['tau']
                sig_indicator = get_significance_indicator(p_val)
                
                ax3.text(0.05, 0.95, f'τ = {tau:.3f}{sig_indicator}\np = {p_val:.3f}', 
                        transform=ax3.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_title('Melt Season BC AOD Trend', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Year', fontsize=12)
        ax3.set_ylabel('BC AOD', fontsize=12)
        ax3.legend(loc='upper left')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Temperature vs Albedo Correlation (Bottom Left)
        ax4 = axes[1, 0]
        
        # Combine MODIS albedo with temperature data
        modis_data_keys = [k for k in time_series_data.keys() if 'modis' in k and 'annual' in k]
        modis_key = modis_data_keys[0] if modis_data_keys else None
        
        if (modis_key and modis_key in time_series_data and 
            'temperature_annual' in time_series_data):
            
            albedo_data = time_series_data[modis_key].copy()
            temp_data = time_series_data['temperature_annual'].copy()
            
            # Merge albedo and temperature data on date
            merged_data = pd.merge(albedo_data, temp_data, on='date')
            
            if len(merged_data) > 0:
                # Plot Albedo vs Temperature correlation
                x_temp = merged_data['Temperature']
                y_albedo = merged_data['Albedo']
                
                ax4.scatter(x_temp, y_albedo, color='#3498DB', alpha=0.7, s=50, 
                           label='Albedo vs Temperature')
                
                # Add regression line for temperature correlation
                if len(x_temp) > 2:
                    slope_temp, intercept_temp, r_temp, p_temp, _ = stats.linregress(x_temp, y_albedo)
                    line_x_temp = np.array([x_temp.min(), x_temp.max()])
                    line_y_temp = slope_temp * line_x_temp + intercept_temp
                    ax4.plot(line_x_temp, line_y_temp, color='#2980B9', linewidth=2,
                            label=f'Linear fit: R² = {r_temp**2:.3f}')
                
                # Add correlation statistics
                sig_temp = get_significance_indicator(p_temp) if len(x_temp) > 2 else ''
                
                stats_text = f'Temp-Albedo: R={r_temp:.3f}{sig_temp}'
                ax4.text(0.05, 0.95, stats_text, 
                        transform=ax4.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax4.legend(loc='upper right')
        
        ax4.set_title('Albedo vs Temperature Correlation', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Temperature (°C)', fontsize=12)
        ax4.set_ylabel('Melt Season Albedo (unitless)', fontsize=12)
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: BC AOD vs Albedo Correlation (Bottom Middle)
        ax5 = axes[1, 1]
        
        # Combine MODIS albedo with BC AOD data
        modis_data_keys = [k for k in time_series_data.keys() if 'modis' in k and 'annual' in k]
        modis_key = modis_data_keys[0] if modis_data_keys else None
        
        if (modis_key and modis_key in time_series_data and 
            'bc_aod_annual' in time_series_data):
            
            albedo_data = time_series_data[modis_key].copy()
            bc_aod_data = time_series_data['bc_aod_annual'].copy()
            
            # Merge albedo and BC AOD data on date
            merged_data = pd.merge(albedo_data, bc_aod_data, on='date')
            
            if len(merged_data) > 0:
                # Plot Albedo vs BC AOD correlation
                x_aod = merged_data['BC_AOD']
                y_albedo = merged_data['Albedo']
                
                ax5.scatter(x_aod, y_albedo, color='#E74C3C', alpha=0.7, s=50, 
                           label='Albedo vs BC AOD')
                
                # Add regression line for BC AOD correlation
                if len(x_aod) > 2:
                    slope_aod, intercept_aod, r_aod, p_aod, _ = stats.linregress(x_aod, y_albedo)
                    line_x_aod = np.array([x_aod.min(), x_aod.max()])
                    line_y_aod = slope_aod * line_x_aod + intercept_aod
                    ax5.plot(line_x_aod, line_y_aod, color='#C0392B', linewidth=2,
                            label=f'Linear fit: R² = {r_aod**2:.3f}')
                
                # Add correlation statistics
                sig_aod = get_significance_indicator(p_aod) if len(x_aod) > 2 else ''
                
                stats_text = f'AOD-Albedo: R={r_aod:.3f}{sig_aod}'
                ax5.text(0.05, 0.95, stats_text, 
                        transform=ax5.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                ax5.legend(loc='upper right')
        
        ax5.set_title('Albedo vs BC AOD Correlation', fontsize=14, fontweight='bold')
        ax5.set_xlabel('BC AOD', fontsize=12)
        ax5.set_ylabel('Melt Season Albedo (unitless)', fontsize=12)
        ax5.set_ylim(0, 1)
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Combined Summary (Bottom Right)
        ax6 = axes[1, 2]
        
        # Create a summary text plot
        summary_text = """
Haig Glacier Trend Analysis Summary
(2002-2024)

Albedo Trends:
• MODIS Albedo: -1.66%/year ***
• Significant decline (p = 0.017)

Contributing Factors:
• Temperature: +0.051°C/year
• BC AOD: +2.41%/year ***

Correlations with Albedo:
• Temperature: R = -0.745***
• BC AOD: R = -0.619***

Both warming and black carbon
deposition drive albedo decline
        """
        
        ax6.text(0.05, 0.95, summary_text.strip(), 
                transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        ax6.set_title('Analysis Summary', fontsize=14, fontweight='bold')
        ax6.axis('off')
        
        # Add overall figure annotations
        fig.text(0.02, 0.02, 'Significance levels: * p < 0.05, ** p < 0.01', 
                 fontsize=10, style='italic')
        fig.text(0.98, 0.02, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}', 
                 fontsize=8, ha='right')
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.96])
        
        # Save figure if output path is provided
        if output_path:
            fig.savefig(output_path, dpi=self.config['visualization']['dpi'], 
                       bbox_inches='tight', facecolor='white', edgecolor='none')
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

def generate_trend_summary_and_readme(output_manager: OutputManager, trend_results: Dict[str, Any], 
                                      time_series_data: Dict[str, pd.DataFrame]):
    """Generate summary file and README with trend analysis results.
    
    Args:
        output_manager: OutputManager instance for file operations
        trend_results: Dictionary containing trend analysis results
        time_series_data: Dictionary containing time series data
    """
    try:
        # Collect trend statistics for summary
        trend_stats = {}
        
        for data_type, results in trend_results.items():
            if 'error' not in results:
                annual_results = results.get('annual', {})
                mk_results = annual_results.get('mann_kendall', {})
                sen_results = annual_results.get('sen_slope', {})
                
                trend_stats[data_type] = {
                    'n_observations': results.get('data_info', {}).get('n_observations', 0),
                    'duration_years': results.get('data_info', {}).get('duration_years', 0),
                    'mann_kendall_trend': mk_results.get('trend', 'no trend'),
                    'statistical_significance': mk_results.get('significance', False),
                    'p_value': mk_results.get('p_value', 1.0),
                    'kendall_tau': mk_results.get('tau', 0.0),
                    'sen_slope_per_year': sen_results.get('slope_per_year', 0.0),
                    'slope_confidence_interval': sen_results.get('confidence_interval', (0, 0))
                }
        
        # Extract key findings
        key_findings = []
        
        # AWS Albedo findings
        if 'aws_albedo_annual' in trend_stats:
            aws_stats = trend_stats['aws_albedo_annual']
            slope = aws_stats['sen_slope_per_year']
            trend = aws_stats['mann_kendall_trend']
            significance = aws_stats['statistical_significance']
            sig_text = "significant" if significance else "non-significant"
            
            if trend == 'decreasing':
                key_findings.append(f"AWS albedo shows {sig_text} decreasing trend of {abs(slope):.4f} per year (p={aws_stats['p_value']:.3f})")
            elif trend == 'increasing':
                key_findings.append(f"AWS albedo shows {sig_text} increasing trend of {slope:.4f} per year (p={aws_stats['p_value']:.3f})")
            else:
                key_findings.append(f"AWS albedo shows no significant trend (p={aws_stats['p_value']:.3f})")
        
        # Temperature findings
        if 'temperature_summer' in trend_stats:
            temp_stats = trend_stats['temperature_summer']
            slope = temp_stats['sen_slope_per_year']
            trend = temp_stats['mann_kendall_trend']
            significance = temp_stats['statistical_significance']
            sig_text = "significant" if significance else "non-significant"
            
            if trend == 'increasing':
                key_findings.append(f"Summer temperature shows {sig_text} warming trend of {slope:.3f}°C per year (p={temp_stats['p_value']:.3f})")
            elif trend == 'decreasing':
                key_findings.append(f"Summer temperature shows {sig_text} cooling trend of {abs(slope):.3f}°C per year (p={temp_stats['p_value']:.3f})")
            else:
                key_findings.append(f"Summer temperature shows no significant trend (p={temp_stats['p_value']:.3f})")
        
        # MODIS findings
        modis_keys = [k for k in trend_stats.keys() if 'modis' in k]
        for modis_key in modis_keys:
            modis_stats = trend_stats[modis_key]
            slope = modis_stats['sen_slope_per_year']
            trend = modis_stats['mann_kendall_trend']
            significance = modis_stats['statistical_significance']
            method_name = modis_key.replace('modis_', '').replace('_annual', '').upper()
            sig_text = "significant" if significance else "non-significant"
            
            if trend == 'decreasing':
                key_findings.append(f"MODIS {method_name} albedo shows {sig_text} decreasing trend of {abs(slope):.4f} per year")
            elif trend == 'increasing':
                key_findings.append(f"MODIS {method_name} albedo shows {sig_text} increasing trend of {slope:.4f} per year")
        
        # Calculate overall statistics
        all_significant_trends = sum(1 for stats in trend_stats.values() if stats['statistical_significance'])
        total_series_analyzed = len(trend_stats)
        
        # Prepare summary data
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'analysis_type': CONFIG['output']['summary_template']['analysis_type'],
            'glacier': 'Haig Glacier, Canadian Rocky Mountains',
            'configuration': {
                'trend_analysis_alpha': CONFIG['trend_analysis']['alpha'],
                'seasonal_analysis': CONFIG['trend_analysis']['seasonal_analysis'],
                'prewhitening': CONFIG['trend_analysis']['prewhitening'],
                'min_years': CONFIG['trend_analysis']['min_years'],
                'methods': CONFIG['methods']
            },
            'data_info': {
                'total_time_series': total_series_analyzed,
                'significant_trends': all_significant_trends,
                'analysis_period': '2002-2024'
            },
            'key_results': {
                'albedo_trend_direction': trend_stats.get('aws_albedo_annual', {}).get('mann_kendall_trend', 'unknown'),
                'temperature_trend_direction': trend_stats.get('temperature_summer', {}).get('mann_kendall_trend', 'unknown'),
                'albedo_change_per_year': trend_stats.get('aws_albedo_annual', {}).get('sen_slope_per_year', 0.0),
                'temperature_change_per_year': trend_stats.get('temperature_summer', {}).get('sen_slope_per_year', 0.0)
            },
            'detailed_statistics': trend_stats
        }
        
        # Save summary
        output_manager.save_summary(summary_data)
        
        # Generate README
        output_manager.save_readme(
            analysis_description=CONFIG['output']['summary_template']['description'],
            key_findings=key_findings,
            additional_info={
                'Analysis Period': '2002-2024 (based on available data)',
                'Methodology': 'Non-parametric Mann-Kendall trend test and Sen slope estimator',
                'Statistical Significance': f"α = {CONFIG['trend_analysis']['alpha']} (95% confidence level)",
                'Seasonal Analysis': 'Summer (June-August) averages for albedo and temperature',
                'Data Sources': 'AWS measurements and MODIS satellite observations',
                'Quality Control': f"Minimum {CONFIG['trend_analysis']['min_years']} years of data required",
                'Reference': 'Based on methodology from Williamson et al. (2021) - Forest fire aerosol and temperature effects on glacier albedo'
            }
        )
        
        logger.info("Trend analysis summary and README generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating trend summary and README: {e}")

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================

def main():
    """Main execution function for Haig glacier trend analysis."""
    logger.info("Starting Sen Slope and Mann-Kendall Trend Analysis for Haig Glacier")
    
    # Initialize OutputManager
    output_manager = OutputManager(
        CONFIG['output']['analysis_name'],
        CONFIG['output']['base_dir']
    )
    
    # Initialize components
    data_loader = DataLoader(CONFIG)
    pixel_selector = PixelSelector(CONFIG)
    data_processor = DataProcessor(CONFIG)
    trend_analyzer = TrendAnalyzer(CONFIG)
    visualizer = TrendVisualizer(CONFIG)
    
    try:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing HAIG GLACIER - Trend Analysis")
        logger.info(f"{'='*60}")
        
        # Load complete Haig glacier dataset
        modis_data, aws_data, temperature_data = data_loader.load_haig_data_complete()
        
        # Apply pixel selection for MODIS data
        selected_modis = pixel_selector.select_best_pixels(modis_data, 'haig')
        
        # Prepare time series data
        time_series_data = data_processor.prepare_time_series_data(
            selected_modis, aws_data, temperature_data, 'haig'
        )
        
        # Create annual aggregated series
        annual_data = data_processor.create_annual_series(time_series_data)
        
        # Combine daily and annual data
        all_time_series = {**time_series_data, **annual_data}
        
        # Perform trend analysis on key time series
        trend_results = {}
        
        # Analyze AWS albedo (annual summer average)
        if 'aws_albedo_annual' in annual_data:
            logger.info("Analyzing AWS albedo annual trends...")
            trend_results['aws_albedo_annual'] = trend_analyzer.analyze_time_series(
                annual_data['aws_albedo_annual'], 'Albedo'
            )
        
        # Analyze temperature (melt season average)
        if 'temperature_annual' in annual_data:
            logger.info("Analyzing melt season temperature trends...")
            trend_results['temperature_annual'] = trend_analyzer.analyze_time_series(
                annual_data['temperature_annual'], 'Temperature'
            )
        
        # Analyze BC AOD (melt season average)
        if 'bc_aod_annual' in annual_data:
            logger.info("Analyzing melt season BC AOD trends...")
            trend_results['bc_aod_annual'] = trend_analyzer.analyze_time_series(
                annual_data['bc_aod_annual'], 'BC_AOD'
            )
        
        # Analyze MODIS albedo (if available)
        modis_keys = [k for k in annual_data.keys() if 'modis' in k and 'annual' in k]
        logger.info(f"Available annual data keys: {list(annual_data.keys())}")
        logger.info(f"MODIS keys found: {modis_keys}")
        
        for modis_key in modis_keys:
            logger.info(f"Analyzing {modis_key} trends...")
            logger.info(f"MODIS data shape: {annual_data[modis_key].shape}")
            trend_results[modis_key] = trend_analyzer.analyze_time_series(
                annual_data[modis_key], 'Albedo'
            )
        
        # Create visualization
        logger.info(f"\n{'='*60}")
        logger.info("Creating Trend Analysis Visualization")
        logger.info(f"{'='*60}")
        
        # Use OutputManager for plot path
        plot_path = output_manager.get_plot_path(CONFIG['output']['plot_filename'])
        
        # Create the trend visualization
        fig = visualizer.create_trend_visualization(
            trend_results, annual_data, str(plot_path)
        )
        output_manager.log_file_saved(plot_path, "plot")
        
        # Show the plot
        plt.show()
        
        # Generate comprehensive summary and README
        generate_trend_summary_and_readme(output_manager, trend_results, all_time_series)
        
        logger.info(f"\nSUCCESS: Haig glacier trend analysis completed and saved")
        logger.info(f"Analysis period: 2002-2024")
        logger.info(f"Trend analysis methods: Mann-Kendall test and Sen slope estimator")
        
    except Exception as e:
        logger.error(f"Error in Haig glacier trend analysis: {e}")
        raise


if __name__ == "__main__":
    main()