#!/usr/bin/env python3
"""
Multi-Glacier Comparative Analysis Engine

This module provides comprehensive comparative analysis capabilities across
all glaciers in the framework, enabling insights into:
- Method performance across different environments
- Regional differences in MODIS accuracy
- Environmental factors affecting albedo retrieval
- Cross-glacier statistical comparisons
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import glob
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiGlacierComparativeAnalysis:
    """
    Main class for comparative analysis across multiple glaciers.
    
    This class aggregates results from individual glacier analyses and provides
    comparative visualizations and statistical insights across:
    - Athabasca Glacier (Canadian Rockies)
    - Haig Glacier (Canadian Rockies) 
    - Coropuna Glacier (Peruvian Andes)
    
    The class supports two main data aggregation modes:
    1. Standard mode: Uses pre-processed pivot results from individual analyses
    2. Pixel selection mode: Loads raw data and applies optimal pixel selection
    
    Attributes:
        outputs_dir (Path): Directory for storing analysis outputs
        config_path (str): Path to glacier configuration file
        glacier_metadata (Dict): Metadata for all configured glaciers
        aggregated_data (DataFrame): Combined analysis results from all glaciers
    
    Example:
        >>> analyzer = MultiGlacierComparativeAnalysis()
        >>> data = analyzer.aggregate_glacier_data()
        >>> summary = analyzer.get_summary_statistics()
    """
    
    def __init__(self, outputs_dir: str = "outputs", config_path: str = "config/glacier_sites.yaml"):
        """
        Initialize the comparative analysis engine.
        
        Args:
            outputs_dir (str): Directory path for storing analysis outputs.
                             Creates timestamped subdirectories for each run.
            config_path (str): Path to YAML configuration file containing
                             glacier metadata and analysis parameters.
        
        Note:
            The outputs directory will be created if it doesn't exist.
            Configuration file must contain 'glaciers' section with glacier definitions.
        """
        self.outputs_dir = Path(outputs_dir)
        self.config_path = config_path
        self.glacier_metadata = self._load_glacier_metadata()
        self.aggregated_data = None
        
    def _load_glacier_metadata(self) -> Dict[str, Dict[str, Any]]:
        """
        Load glacier metadata from configuration file.
        
        Reads the YAML configuration file and extracts metadata for each glacier
        including coordinates, elevation, region, and analysis parameters.
        
        Returns:
            Dict[str, Dict[str, Any]]: Nested dictionary with glacier_id as keys
                                     and metadata dictionaries as values.
                                     Each metadata dict contains:
                                     - name: Human-readable glacier name
                                     - region: Geographic region
                                     - latitude/longitude: Coordinates
                                     - elevation: Estimated elevation (m)
                                     - data_type: Analysis type flag
        
        Raises:
            Exception: If configuration file cannot be loaded or parsed
        """
        try:
            import sys
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from utils.config.helpers import load_config
            
            glacier_config = load_config(self.config_path)
            metadata = {}
            
            for glacier_id, config in glacier_config['glaciers'].items():
                metadata[glacier_id] = {
                    'name': config['name'],
                    'region': config.get('region', 'Unknown'),
                    'latitude': config.get('coordinates', {}).get('lat', 0),
                    'longitude': config.get('coordinates', {}).get('lon', 0),
                    'elevation': self._get_elevation_estimate(glacier_id, config),
                    'data_type': config.get('data_type', 'standard')
                }
            
            logger.info(f"Loaded metadata for {len(metadata)} glaciers")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load glacier metadata: {e}")
            return {}
    
    def _get_elevation_estimate(self, glacier_id: str, config: Dict[str, Any]) -> float:
        """Estimate glacier elevation from AWS station or default values."""
        # Try to get elevation from AWS station
        aws_stations = config.get('aws_stations', {})
        if aws_stations:
            for station in aws_stations.values():
                if 'elevation' in station:
                    return station['elevation']
        
        # Default elevations based on known glaciers
        elevation_defaults = {
            'athabasca': 2200,
            'haig': 2800, 
            'coropuna': 5400
        }
        
        return elevation_defaults.get(glacier_id, 3000)  # Default middle elevation
    
    def discover_latest_results(self) -> Dict[str, Dict[str, str]]:
        """
        Discover the latest analysis results for each glacier.
        
        Returns:
            Dictionary mapping glacier_id to file paths for latest results
        """
        glacier_results = {}
        
        if not self.outputs_dir.exists():
            logger.warning(f"Outputs directory {self.outputs_dir} does not exist")
            return glacier_results
        
        # Look for directories matching glacier analysis patterns (pivot or comprehensive)
        for glacier_id in self.glacier_metadata.keys():
            # Try multiple patterns in order of preference
            patterns = [f"{glacier_id}_pivot_*", f"{glacier_id}_comprehensive_*", f"{glacier_id}_enhanced_*", f"{glacier_id}_best_pixel_*"]
            
            latest_dir = None
            analysis_type = None
            
            for pattern in patterns:
                matching_dirs = list(self.outputs_dir.glob(pattern))
                if matching_dirs:
                    # Get the most recent directory (by name/timestamp)
                    latest_dir = max(matching_dirs, key=lambda x: x.name)
                    analysis_type = pattern.split('_')[1].replace('*', '')  # Extract analysis type
                    break
            
            if latest_dir:
                # Check for required result files with flexible naming
                results_dir = latest_dir / "results"
                if results_dir.exists():
                    # Try different file naming patterns
                    method_comparison_file = None
                    outlier_analysis_file = None
                    merged_data_file = None
                    
                    # Check for files with analysis type suffix
                    for suffix in [f"_{analysis_type}", ""]:
                        if not method_comparison_file:
                            candidate = results_dir / f"{glacier_id}{suffix}_method_comparison.csv"
                            if candidate.exists():
                                method_comparison_file = candidate
                        
                        if not outlier_analysis_file:
                            candidate = results_dir / f"{glacier_id}{suffix}_outlier_analysis.csv"
                            if candidate.exists():
                                outlier_analysis_file = candidate
                        
                        if not merged_data_file:
                            candidate = results_dir / f"{glacier_id}{suffix}_merged_data.csv"
                            if candidate.exists():
                                merged_data_file = candidate
                    
                    if method_comparison_file and method_comparison_file.exists():
                        glacier_results[glacier_id] = {
                            'directory': str(latest_dir),
                            'method_comparison': str(method_comparison_file),
                            'outlier_analysis': str(outlier_analysis_file) if outlier_analysis_file and outlier_analysis_file.exists() else None,
                            'merged_data': str(merged_data_file) if merged_data_file and merged_data_file.exists() else None,
                            'timestamp': self._extract_timestamp(latest_dir.name),
                            'analysis_type': analysis_type
                        }
                        logger.info(f"Found {analysis_type} results for {glacier_id}: {latest_dir.name}")
                    else:
                        logger.warning(f"No method comparison file found for {glacier_id} in {latest_dir.name}")
                else:
                    logger.warning(f"No results directory found in {latest_dir.name}")
            else:
                logger.warning(f"No analysis results found for {glacier_id}")
        
        logger.info(f"Discovered results for {len(glacier_results)} glaciers")
        return glacier_results
    
    def _extract_timestamp(self, dirname: str) -> str:
        """Extract timestamp from directory name."""
        match = re.search(r'(\d{8}_\d{6})$', dirname)
        return match.group(1) if match else "unknown"
    
    def aggregate_glacier_data(self) -> pd.DataFrame:
        """
        Aggregate outlier-filtered method comparison data from all glaciers into a unified dataset.
        
        This method discovers the latest analysis results for each glacier and combines
        them into a single DataFrame for comparative analysis. It prioritizes outlier-filtered
        statistics when available to ensure robust comparisons.
        
        Process:
        1. Discovers latest analysis directories for each glacier (glacier_pivot_YYYYMMDD_HHMMSS)
        2. Loads outlier_analysis.csv if available, otherwise falls back to method_comparison.csv
        3. Filters for 'without_outliers' condition to use cleaned statistics
        4. Adds glacier metadata (coordinates, elevation, region) to each row
        5. Standardizes column names and data types across glaciers
        
        Returns:
            pd.DataFrame: Unified dataset with columns:
                - glacier_id: Glacier identifier (athabasca, haig, coropuna)
                - glacier_name: Human-readable name
                - region: Geographic region
                - method: MODIS product (MOD09GA, MYD09GA, mod10a1, myd10a1, mcd43a3)
                - n_samples: Number of AWS-MODIS paired observations
                - r: Pearson correlation coefficient
                - p: Statistical significance p-value
                - r_squared: Coefficient of determination
                - rmse: Root Mean Square Error
                - mae: Mean Absolute Error  
                - bias: Mean bias (MODIS - AWS)
                - elevation, latitude, longitude: Glacier coordinates
                - analysis_timestamp: When analysis was performed
                
        Note:
            Returns empty DataFrame if no analysis results are found.
            Prefers outlier-filtered statistics for better statistical robustness.
        """
        glacier_results = self.discover_latest_results()
        
        if not glacier_results:
            logger.error("No glacier results found for comparative analysis")
            logger.info("Expected directory pattern: {glacier_id}_pivot_YYYYMMDD_HHMMSS in outputs/")
            logger.info("Ensure individual glacier analyses have been run first")
            return pd.DataFrame()
        
        all_data = []
        
        for glacier_id, result_files in glacier_results.items():
            try:
                # Prefer outlier analysis data if available, fallback to method comparison
                if result_files['outlier_analysis'] and Path(result_files['outlier_analysis']).exists():
                    # Load outlier analysis data and filter for "without_outliers" condition
                    outlier_df = pd.read_csv(result_files['outlier_analysis'])
                    
                    # Filter for without_outliers condition
                    filtered_df = outlier_df[outlier_df['condition'] == 'without_outliers'].copy()
                    
                    if filtered_df.empty:
                        logger.warning(f"No without_outliers data found for {glacier_id}, using raw data")
                        logger.debug(f"Available conditions in outlier file: {outlier_df['condition'].unique()}")
                        method_df = pd.read_csv(result_files['method_comparison'], index_col=0)
                    else:
                        # Set method as index to match expected format
                        method_df = filtered_df.set_index('method')
                        logger.info(f"Using outlier-filtered data for {glacier_id} ({len(filtered_df)} methods)")
                        logger.debug(f"Outlier filtering improved statistics for: {list(filtered_df.index)}")
                else:
                    # Fallback to method comparison data
                    logger.warning(f"No outlier analysis file found for {glacier_id}, using raw data")
                    method_df = pd.read_csv(result_files['method_comparison'], index_col=0)
                
                # Add metadata to each row
                metadata = self.glacier_metadata[glacier_id]
                
                for method in method_df.index:
                    row_data = method_df.loc[method].to_dict()
                    row_data.update({
                        'glacier_id': glacier_id,
                        'glacier_name': metadata['name'],
                        'region': metadata['region'],
                        'latitude': metadata['latitude'],
                        'longitude': metadata['longitude'],
                        'elevation': metadata['elevation'],
                        'method': method,
                        'data_type': metadata['data_type'],
                        'analysis_timestamp': result_files['timestamp']
                    })
                    all_data.append(row_data)
                
                logger.info(f"Aggregated data for {glacier_id}: {len(method_df)} methods")
                
            except Exception as e:
                logger.error(f"Failed to load data for {glacier_id}: {e}")
                continue
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Ensure numeric columns are properly typed
            # Handle different column names between outlier_analysis.csv (n) and method_comparison.csv (n_samples)
            if 'n' in df.columns and 'n_samples' not in df.columns:
                df['n_samples'] = df['n']
            
            numeric_columns = ['n_samples', 'n', 'r', 'p', 'r_squared', 'rmse', 'mae', 'bias', 
                             'latitude', 'longitude', 'elevation']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by glacier and method for consistent ordering
            df = df.sort_values(['glacier_id', 'method']).reset_index(drop=True)
            
            logger.info(f"Created unified dataset with {len(df)} method-glacier combinations")
            self.aggregated_data = df
            return df
        
        else:
            logger.error("No data could be aggregated")
            return pd.DataFrame()
    
    def aggregate_glacier_data_with_pixel_selection(self) -> pd.DataFrame:
        """
        Aggregate glacier data using selected pixels from the actual source data files.
        
        This is the advanced analysis mode that bypasses pre-processed pivot results
        and instead loads raw MODIS/AWS data directly, applying intelligent pixel
        selection to optimize analysis quality.
        
        Pixel Selection Algorithm:
        - Composite scoring: 60% distance weight + 40% glacier fraction weight
        - Selects pixels closest to AWS stations with highest glacier coverage
        - For small datasets (≤2 pixels), uses all available pixels
        
        Process:
        1. Loads raw MODIS data from source CSV files for each glacier
        2. Loads AWS data with glacier-specific parsing (handles different formats)
        3. Applies pixel selection algorithm to find optimal pixels
        4. Performs temporal matching between AWS and MODIS observations
        5. Calculates statistical metrics for each method-glacier combination
        6. Returns unified dataset with same structure as standard aggregation
        
        Data Sources:
        - Athabasca: D:/Documents/Projects/athabasca_analysis/data/csv/
        - Haig: D:/Documents/Projects/Haig_analysis/data/csv/
        - Coropuna: D:/Documents/Projects/Coropuna_glacier/data/csv/
        
        Returns:
            pd.DataFrame: Same structure as aggregate_glacier_data() but using
                         selected pixels only. Contains correlation, RMSE, MAE,
                         bias statistics for each method-glacier combination.
                         
        Note:
            This method is computationally intensive as it processes raw data
            but provides the most accurate results by using optimal pixel selection.
            Selected pixels are typically <1km from AWS stations with >90% glacier coverage.
        """
        logger.info("Aggregating glacier data with pixel selection from source files...")
        
        # Import required modules
        try:
            import sys
            import os
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
            from analysis.spatial.glacier_mapping_simple import MultiGlacierMapperSimple
            from utils.config.helpers import load_config
        except ImportError as e:
            logger.error(f"Failed to import required modules: {e}")
            return pd.DataFrame()
        
        all_data = []
        mapper = MultiGlacierMapperSimple()
        config = load_config('config/config.yaml')
        
        for glacier_id in ['athabasca', 'haig', 'coropuna']:
            if glacier_id not in self.glacier_metadata:
                logger.warning(f"No metadata found for {glacier_id}")
                continue
            
            try:
                logger.info(f"Processing {glacier_id} with selected pixels from source data...")
                
                # Get glacier configuration
                glacier_config = self.glacier_metadata[glacier_id]
                
                # Determine the correct data path based on glacier
                if glacier_id == 'coropuna':
                    # For Coropuna, use the direct path to the real data
                    data_path = "D:/Documents/Projects/Coropuna_glacier/data/csv/coropuna_glacier_2014-01-01_to_2025-01-01.csv"
                    aws_path = "D:/Documents/Projects/Coropuna_glacier/data/csv/COROPUNA_simple.csv"
                elif glacier_id == 'haig':
                    # For Haig, use the configured path
                    data_path = "D:/Documents/Projects/Haig_analysis/data/csv/Haig_MODIS_Pixel_Analysis_MultiProduct_2002_to_2016_fraction.csv"
                    aws_path = "D:/Documents/Projects/Haig_analysis/data/csv/HaigAWS_daily_2002_2015_gapfilled.csv"
                elif glacier_id == 'athabasca':
                    # For Athabasca, use the configured path
                    data_path = "D:/Documents/Projects/athabasca_analysis/data/csv/Athabasca_Terra_Aqua_MultiProduct_2014-01-01_to_2021-01-01.csv"
                    aws_path = "D:/Documents/Projects/athabasca_analysis/data/csv/iceAWS_Atha_albedo_daily_20152020_filled_clean.csv"
                else:
                    logger.warning(f"Unknown glacier {glacier_id}")
                    continue
                
                # Check if files exist
                if not os.path.exists(data_path):
                    logger.warning(f"MODIS data file not found: {data_path}")
                    continue
                if not os.path.exists(aws_path):
                    logger.warning(f"AWS data file not found: {aws_path}")
                    continue
                
                # Load the MODIS data
                logger.info(f"Loading MODIS data from: {data_path}")
                modis_df = pd.read_csv(data_path)
                logger.info(f"Loaded {len(modis_df):,} MODIS observations")
                
                # Load AWS data with glacier-specific parameters
                logger.info(f"Loading AWS data from: {aws_path}")
                
                try:
                    if glacier_id == 'haig':
                        # Haig AWS format has header rows and uses semicolons with European decimal format
                        aws_df = pd.read_csv(aws_path, sep=';', skiprows=6, decimal=',')
                        # Clean up column names by stripping whitespace
                        aws_df.columns = aws_df.columns.str.strip()
                    else:
                        # Standard CSV format for other glaciers
                        aws_df = pd.read_csv(aws_path)
                    
                    logger.info(f"Loaded {len(aws_df):,} AWS observations")
                    
                except Exception as e:
                    logger.error(f"Failed to load AWS data for {glacier_id}: {e}")
                    continue
                
                # Get selected pixels using the mapper
                selected_pixels = mapper.load_original_modis_data(glacier_id, analysis_mode=True)
                if selected_pixels is None or selected_pixels.empty:
                    logger.warning(f"No selected pixels found for {glacier_id}")
                    continue
                
                # For small datasets like Athabasca, use all available pixels
                unique_pixels = modis_df['pixel_id'].unique()
                if len(unique_pixels) <= 2:
                    selected_pixel_ids = unique_pixels.tolist()
                    logger.info(f"Using all {len(selected_pixel_ids)} pixels for {glacier_id} (small dataset)")
                else:
                    # Get the selected pixel ID
                    selected_pixel_id = int(selected_pixels['pixel_id'].iloc[0])
                    
                    # Check if selected pixel exists in the actual data
                    if selected_pixel_id in unique_pixels:
                        selected_pixel_ids = [selected_pixel_id]
                        pixel_count = len(modis_df[modis_df['pixel_id'] == selected_pixel_id])
                        logger.info(f"Using selected pixel {selected_pixel_id} with {pixel_count:,} observations")
                    else:
                        # Find the pixel with most observations as fallback
                        pixel_counts = modis_df['pixel_id'].value_counts()
                        best_pixel = pixel_counts.index[0]
                        selected_pixel_ids = [best_pixel]
                        logger.warning(f"Selected pixel {selected_pixel_id} not found in data!")
                        logger.info(f"Using best available pixel {best_pixel} with {pixel_counts.iloc[0]:,} observations")
                
                # Filter MODIS data by selected pixels
                filtered_modis = modis_df[modis_df['pixel_id'].isin(selected_pixel_ids)].copy()
                logger.info(f"Filtered to {len(filtered_modis):,} MODIS observations for selected pixels")
                
                if filtered_modis.empty:
                    logger.warning(f"No MODIS data remaining after pixel filtering for {glacier_id}")
                    continue
                
                # Prepare data for AWS-MODIS matching
                # Convert MODIS date to datetime
                filtered_modis['date'] = pd.to_datetime(filtered_modis['date'])
                
                # Prepare AWS data based on glacier-specific format
                if glacier_id == 'coropuna':
                    # Coropuna AWS format: Timestamp, Albedo
                    aws_df['date'] = pd.to_datetime(aws_df['Timestamp'])
                    aws_date_col = 'date'
                    aws_albedo_col = 'Albedo'
                elif glacier_id == 'haig':
                    # Haig AWS format - data already loaded with proper parameters
                    # Create date from Year and Day columns (Day of Year)
                    # Handle potential missing values
                    aws_df = aws_df.dropna(subset=['Year', 'Day'])
                    aws_df['Year'] = aws_df['Year'].astype(int)
                    aws_df['Day'] = aws_df['Day'].astype(int)
                    
                    # Convert to datetime using Day of Year
                    aws_df['date'] = pd.to_datetime(aws_df['Year'].astype(str) + '-01-01') + pd.to_timedelta(aws_df['Day'] - 1, unit='D')
                    aws_date_col = 'date'
                    aws_albedo_col = 'albedo'  # Column name after stripping spaces
                    
                    logger.info(f"Successfully parsed Haig AWS data: {len(aws_df)} records")
                elif glacier_id == 'athabasca':
                    # Athabasca AWS format: Time, Albedo
                    aws_df['date'] = pd.to_datetime(aws_df['Time'])
                    aws_date_col = 'date'
                    aws_albedo_col = 'Albedo'
                
                # Merge MODIS and AWS data on date
                merged_data = filtered_modis.merge(
                    aws_df[[aws_date_col, aws_albedo_col]].rename(columns={aws_date_col: 'date', aws_albedo_col: 'aws_albedo'}),
                    on='date',
                    how='inner'
                )
                
                logger.info(f"Merged to {len(merged_data):,} AWS-MODIS pairs for {glacier_id}")
                
                if merged_data.empty:
                    logger.warning(f"No AWS-MODIS pairs found for {glacier_id}")
                    continue
                
                # Process each method separately 
                metadata = self.glacier_metadata[glacier_id]
                
                if glacier_id == 'coropuna':
                    # Coropuna has method column - process each method separately
                    available_methods = merged_data['method'].unique()
                    
                    for method in available_methods:
                        method_subset = merged_data[merged_data['method'] == method]
                        if len(method_subset) < 3:  # Need minimum data points
                            continue
                        
                        # Get AWS and MODIS values for this method
                        aws_values = method_subset['aws_albedo'].values
                        modis_values = method_subset['albedo'].values
                        
                        # Remove NaN values
                        valid_mask = ~(np.isnan(aws_values) | np.isnan(modis_values))
                        if np.sum(valid_mask) < 3:
                            continue
                        
                        valid_aws = aws_values[valid_mask]
                        valid_modis = modis_values[valid_mask]
                        
                        # Calculate statistics
                        correlation = np.corrcoef(valid_aws, valid_modis)[0, 1]
                        n_samples = len(valid_aws)
                        
                        # Calculate RMSE, MAE, Bias
                        rmse = np.sqrt(np.mean((valid_aws - valid_modis) ** 2))
                        mae = np.mean(np.abs(valid_aws - valid_modis))
                        bias = np.mean(valid_modis - valid_aws)
                        
                        # Calculate p-value
                        if n_samples > 2 and not np.isnan(correlation):
                            t_stat = correlation * np.sqrt((n_samples - 2) / (1 - correlation**2))
                            from scipy import stats
                            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 2))
                        else:
                            p_value = 1.0
                        
                        row_data = {
                            'glacier_id': glacier_id,
                            'glacier_name': metadata['name'],
                            'region': metadata['region'],
                            'latitude': metadata['latitude'],
                            'longitude': metadata['longitude'],
                            'elevation': metadata['elevation'],
                            'method': method,
                            'data_type': metadata['data_type'],
                            'n_samples': n_samples,
                            'r': correlation if not np.isnan(correlation) else 0.0,
                            'p': p_value,
                            'r_squared': correlation**2 if not np.isnan(correlation) else 0.0,
                            'rmse': rmse,
                            'mae': mae,
                            'bias': bias,
                            'analysis_timestamp': 'selected_pixels_analysis'
                        }
                        all_data.append(row_data)
                        
                        logger.info(f"Processed method {method} for {glacier_id}: {n_samples} samples, r={correlation:.3f}")
                
                else:
                    # Other glaciers - check for method columns or use method grouping
                    method_columns = [col for col in merged_data.columns if 'modis_' in col.lower() or col in ['MOD09GA', 'MYD09GA', 'mod10a1', 'myd10a1', 'mcd43a3']]
                    
                    if method_columns:
                        # Process each method column
                        aws_values = merged_data['aws_albedo'].values
                        
                        for method_col in method_columns:
                            modis_values = merged_data[method_col].values
                            
                            # Remove NaN values
                            valid_mask = ~(np.isnan(aws_values) | np.isnan(modis_values))
                            if np.sum(valid_mask) < 3:
                                continue
                            
                            valid_aws = aws_values[valid_mask]
                            valid_modis = modis_values[valid_mask]
                            
                            # Calculate statistics
                            correlation = np.corrcoef(valid_aws, valid_modis)[0, 1]
                            n_samples = len(valid_aws)
                            
                            # Calculate RMSE, MAE, Bias
                            rmse = np.sqrt(np.mean((valid_aws - valid_modis) ** 2))
                            mae = np.mean(np.abs(valid_aws - valid_modis))
                            bias = np.mean(valid_modis - valid_aws)
                            
                            # Calculate p-value
                            if n_samples > 2 and not np.isnan(correlation):
                                t_stat = correlation * np.sqrt((n_samples - 2) / (1 - correlation**2))
                                from scipy import stats
                                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 2))
                            else:
                                p_value = 1.0
                            
                            row_data = {
                                'glacier_id': glacier_id,
                                'glacier_name': metadata['name'],
                                'region': metadata['region'],
                                'latitude': metadata['latitude'],
                                'longitude': metadata['longitude'],
                                'elevation': metadata['elevation'],
                                'method': method_col,
                                'data_type': metadata['data_type'],
                                'n_samples': n_samples,
                                'r': correlation if not np.isnan(correlation) else 0.0,
                                'p': p_value,
                                'r_squared': correlation**2 if not np.isnan(correlation) else 0.0,
                                'rmse': rmse,
                                'mae': mae,
                                'bias': bias,
                                'analysis_timestamp': 'selected_pixels_analysis'
                            }
                            all_data.append(row_data)
                            
                            logger.info(f"Processed method {method_col} for {glacier_id}: {n_samples} samples, r={correlation:.3f}")
                    
                    elif 'method' in merged_data.columns:
                        # Use method grouping like Coropuna
                        available_methods = merged_data['method'].unique()
                        
                        for method in available_methods:
                            method_subset = merged_data[merged_data['method'] == method]
                            if len(method_subset) < 3:
                                continue
                            
                            aws_values = method_subset['aws_albedo'].values
                            modis_values = method_subset['albedo'].values
                            
                            # Remove NaN values
                            valid_mask = ~(np.isnan(aws_values) | np.isnan(modis_values))
                            if np.sum(valid_mask) < 3:
                                continue
                            
                            valid_aws = aws_values[valid_mask]
                            valid_modis = modis_values[valid_mask]
                            
                            # Calculate statistics
                            correlation = np.corrcoef(valid_aws, valid_modis)[0, 1]
                            n_samples = len(valid_aws)
                            
                            # Calculate RMSE, MAE, Bias
                            rmse = np.sqrt(np.mean((valid_aws - valid_modis) ** 2))
                            mae = np.mean(np.abs(valid_aws - valid_modis))
                            bias = np.mean(valid_modis - valid_aws)
                            
                            # Calculate p-value
                            if n_samples > 2 and not np.isnan(correlation):
                                t_stat = correlation * np.sqrt((n_samples - 2) / (1 - correlation**2))
                                from scipy import stats
                                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 2))
                            else:
                                p_value = 1.0
                            
                            row_data = {
                                'glacier_id': glacier_id,
                                'glacier_name': metadata['name'],
                                'region': metadata['region'],
                                'latitude': metadata['latitude'],
                                'longitude': metadata['longitude'],
                                'elevation': metadata['elevation'],
                                'method': method,
                                'data_type': metadata['data_type'],
                                'n_samples': n_samples,
                                'r': correlation if not np.isnan(correlation) else 0.0,
                                'p': p_value,
                                'r_squared': correlation**2 if not np.isnan(correlation) else 0.0,
                                'rmse': rmse,
                                'mae': mae,
                                'bias': bias,
                                'analysis_timestamp': 'selected_pixels_analysis'
                            }
                            all_data.append(row_data)
                            
                            logger.info(f"Processed method {method} for {glacier_id}: {n_samples} samples, r={correlation:.3f}")
                
                logger.info(f"Successfully processed {glacier_id} with selected pixels from source data")
                
            except Exception as e:
                logger.error(f"Failed to process {glacier_id} with selected pixels: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['n_samples', 'r', 'p', 'r_squared', 'rmse', 'mae', 'bias', 
                             'latitude', 'longitude', 'elevation']
            
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by glacier and method for consistent ordering
            df = df.sort_values(['glacier_id', 'method']).reset_index(drop=True)
            
            logger.info(f"Created pixel-selected dataset with {len(df)} method-glacier combinations")
            self.aggregated_data = df
            return df
        
        else:
            logger.error("No data could be aggregated with pixel selection")
            return pd.DataFrame()
    
    def aggregate_merged_data_for_scatterplots(self) -> pd.DataFrame:
        """
        Aggregate merged data from all glaciers for AWS vs MODIS scatterplots.
        
        Returns:
            Unified DataFrame with AWS and MODIS values for all glaciers
        """
        glacier_results = self.discover_latest_results()
        
        if not glacier_results:
            logger.error("No glacier results found for scatterplot analysis")
            return pd.DataFrame()
        
        all_data = []
        
        for glacier_id, result_files in glacier_results.items():
            try:
                # Check if merged data file exists
                if not result_files['merged_data'] or not Path(result_files['merged_data']).exists():
                    logger.warning(f"No merged data file found for {glacier_id}")
                    continue
                
                # Load merged data
                merged_df = pd.read_csv(result_files['merged_data'])
                
                # Add glacier metadata
                metadata = self.glacier_metadata[glacier_id]
                merged_df['glacier_id'] = glacier_id
                merged_df['glacier_name'] = metadata['name']
                merged_df['region'] = metadata['region']
                merged_df['latitude'] = metadata['latitude']
                merged_df['longitude'] = metadata['longitude']
                merged_df['elevation'] = metadata['elevation']
                merged_df['analysis_timestamp'] = result_files['timestamp']
                
                all_data.append(merged_df)
                logger.info(f"Loaded merged data for {glacier_id}: {len(merged_df)} records")
                
            except Exception as e:
                logger.error(f"Failed to load merged data for {glacier_id}: {e}")
                continue
        
        if all_data:
            # Combine all glacier data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Ensure numeric columns are properly typed
            numeric_columns = ['MCD43A3', 'MOD09GA', 'MOD10A1', 'AWS', 
                             'latitude', 'longitude', 'elevation']
            
            for col in numeric_columns:
                if col in combined_df.columns:
                    combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            
            # Remove rows where all MODIS methods or AWS are NaN
            combined_df = combined_df.dropna(subset=['AWS'])
            
            logger.info(f"Created unified merged dataset with {len(combined_df)} observations")
            return combined_df
        
        else:
            logger.error("No merged data could be aggregated")
            return pd.DataFrame()
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics for the comparative analysis."""
        if self.aggregated_data is None or self.aggregated_data.empty:
            logger.warning("No aggregated data available for summary statistics")
            return {}
        
        df = self.aggregated_data
        
        summary = {
            'total_glaciers': df['glacier_id'].nunique(),
            'total_methods': df['method'].nunique(),
            'total_observations': df['n_samples'].sum(),
            'date_range': {
                'earliest_analysis': df['analysis_timestamp'].min(),
                'latest_analysis': df['analysis_timestamp'].max()
            },
            'geographic_range': {
                'min_latitude': df['latitude'].min(),
                'max_latitude': df['latitude'].max(),
                'min_elevation': df['elevation'].min(),
                'max_elevation': df['elevation'].max()
            },
            'performance_ranges': {
                'correlation_range': [df['r'].min(), df['r'].max()],
                'rmse_range': [df['rmse'].min(), df['rmse'].max()],
                'bias_range': [df['bias'].min(), df['bias'].max()]
            },
            'glaciers_by_region': df.groupby('region')['glacier_id'].nunique().to_dict(),
            'methods_by_glacier': df.groupby('glacier_id')['method'].nunique().to_dict(),
            'best_performing_method': {
                'by_correlation': df.loc[df['r'].idxmax(), ['glacier_id', 'method', 'r']].to_dict(),
                'by_rmse': df.loc[df['rmse'].idxmin(), ['glacier_id', 'method', 'rmse']].to_dict()
            }
        }
        
        return summary
    
    def create_output_directory(self) -> Path:
        """Create timestamped output directory for comparative analysis."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = self.outputs_dir / f"comparative_analysis_{timestamp}"
        
        # Create subdirectories
        (output_dir / "plots").mkdir(parents=True, exist_ok=True)
        (output_dir / "results").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory: {output_dir}")
        return output_dir
    
    def export_aggregated_data(self, output_dir: Path) -> None:
        """Export the aggregated data to CSV."""
        if self.aggregated_data is None or self.aggregated_data.empty:
            logger.warning("No aggregated data to export")
            return
        
        output_file = output_dir / "results" / "comparative_summary.csv"
        self.aggregated_data.to_csv(output_file, index=False)
        logger.info(f"Exported aggregated data to: {output_file}")
        
        # Also export summary statistics
        summary = self.get_summary_statistics()
        summary_file = output_dir / "results" / "analysis_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Multi-Glacier Comparative Analysis Summary\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total Glaciers: {summary['total_glaciers']}\n")
            f.write(f"Total Methods: {summary['total_methods']}\n")
            f.write(f"Total Observations: {summary['total_observations']}\n\n")
            
            f.write("Geographic Range:\n")
            f.write(f"  Latitude: {summary['geographic_range']['min_latitude']:.2f} to {summary['geographic_range']['max_latitude']:.2f}\n")
            f.write(f"  Elevation: {summary['geographic_range']['min_elevation']:.0f}m to {summary['geographic_range']['max_elevation']:.0f}m\n\n")
            
            f.write("Performance Ranges:\n")
            f.write(f"  Correlation: {summary['performance_ranges']['correlation_range'][0]:.3f} to {summary['performance_ranges']['correlation_range'][1]:.3f}\n")
            f.write(f"  RMSE: {summary['performance_ranges']['rmse_range'][0]:.3f} to {summary['performance_ranges']['rmse_range'][1]:.3f}\n")
            f.write(f"  Bias: {summary['performance_ranges']['bias_range'][0]:.3f} to {summary['performance_ranges']['bias_range'][1]:.3f}\n\n")
            
            f.write("Best Performing Methods:\n")
            best_corr = summary['best_performing_method']['by_correlation']
            best_rmse = summary['best_performing_method']['by_rmse']
            f.write(f"  Highest Correlation: {best_corr['method']} on {best_corr['glacier_id']} (r={best_corr['r']:.3f})\n")
            f.write(f"  Lowest RMSE: {best_rmse['method']} on {best_rmse['glacier_id']} (RMSE={best_rmse['rmse']:.3f})\n")
        
        logger.info(f"Exported summary statistics to: {summary_file}")


def main():
    """Main function for testing the comparative analysis engine."""
    analyzer = MultiGlacierComparativeAnalysis()
    
    # Test data discovery
    results = analyzer.discover_latest_results()
    print(f"Found results for {len(results)} glaciers:")
    for glacier_id, info in results.items():
        print(f"  {glacier_id}: {info['timestamp']}")
    
    # Test data aggregation
    df = analyzer.aggregate_glacier_data()
    if not df.empty:
        print(f"\nAggregated data shape: {df.shape}")
        print(f"Glaciers: {df['glacier_id'].unique()}")
        print(f"Methods: {df['method'].unique()}")
        
        # Test output directory creation and export
        output_dir = analyzer.create_output_directory()
        analyzer.export_aggregated_data(output_dir)
        
        print(f"\nResults exported to: {output_dir}")


if __name__ == "__main__":
    main()