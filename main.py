#!/usr/bin/env python3
"""
Main script for MODIS Albedo Analysis Framework

This script processes one glacier at a time, comparing MODIS albedo products
(MOD09GA, MOD10A1, MCD43A3) against AWS albedo measurements.

Usage:
    python main.py --glacier glacier_1 --config config/config.yaml
    python main.py --glacier glacier_2 --config config/config.yaml
    python main.py --all-glaciers --config config/config.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import geopandas as gpd

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.helpers import load_config, setup_logging, ensure_directory_exists, get_timestamp
from src.data.pivot_based_loaders import create_pivot_based_loader, PivotBasedProcessor
from src.data.data_processor import DataProcessor
from src.analysis.albedo_calculator import AlbedoCalculator
# Skip StatisticalAnalyzer due to sklearn dependency issues - use simplified stats
# from src.analysis.statistical_analysis import StatisticalAnalyzer
from src.mapping.spatial_utils import SpatialProcessor
from src.mapping.glacier_masks import GlacierMaskProcessor
from src.visualization.plots import PlotGenerator
# Skip MapGenerator due to cartopy dependency issues
# from src.visualization.maps import MapGenerator


class AlbedoAnalysisPipeline:
    """Main pipeline for albedo analysis of a single glacier."""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration."""
        self.config = load_config(config_path)
        setup_logging(self.config)
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors  
        self.data_processor = DataProcessor(self.config)
        self.pivot_processor = PivotBasedProcessor(self.config)
        self.albedo_calculator = AlbedoCalculator(self.config)
        # Skip StatisticalAnalyzer due to sklearn dependency issues
        # self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.spatial_processor = SpatialProcessor(self.config)
        self.mask_processor = GlacierMaskProcessor(self.config)
        try:
            self.plot_generator = PlotGenerator(self.config)
        except ImportError as e:
            logging.warning(f"PlotGenerator failed to initialize: {e}")
            self.plot_generator = None
        # Skip MapGenerator due to cartopy dependency issues
        self.map_generator = None
        
        # Create output directories
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
    
    def process_glacier(self, glacier_id: str) -> Dict[str, Any]:
        """Process a single glacier and return results."""
        self.logger.info(f"Starting analysis for glacier: {glacier_id}")
        
        try:
            # Load glacier configuration
            glacier_config = self._load_glacier_config(glacier_id)
            
            # Create glacier-specific output directories
            glacier_output_dir = self._create_glacier_output_dir(glacier_id)
            
            # Step 1: Load and validate data
            self.logger.info("Loading data...")
            data_dict = self._load_all_data(glacier_config)
            
            # Step 2: Spatial processing
            self.logger.info("Processing spatial data...")
            spatial_results = self._process_spatial_data(data_dict, glacier_config)
            
            # Step 3: Temporal alignment
            self.logger.info("Aligning temporal data...")
            aligned_data = self._align_temporal_data(data_dict)
            
            # Step 4: Statistical analysis
            self.logger.info("Performing statistical analysis...")
            statistics = self._perform_statistical_analysis(aligned_data)
            
            # Step 5: Generate visualizations
            self.logger.info("Generating plots...")
            self._generate_plots(aligned_data, statistics, glacier_id, glacier_output_dir)
            
            # Step 6: Generate maps
            self.logger.info("Generating maps...")
            self._generate_maps(spatial_results, glacier_config, glacier_id, glacier_output_dir)
            
            # Step 7: Export results
            self.logger.info("Exporting results...")
            results = self._export_results(statistics, glacier_id, glacier_output_dir)
            
            self.logger.info(f"Analysis completed for glacier: {glacier_id}")
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
        glacier_dir = os.path.join(self.config['output']['base_path'], f"{glacier_id}_{timestamp}")
        
        subdirs = ['results', 'plots', 'maps']
        for subdir in subdirs:
            ensure_directory_exists(os.path.join(glacier_dir, subdir))
        
        return glacier_dir
    
    def _load_all_data(self, glacier_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load all data for the glacier using appropriate method."""
        data_dict = {}
        
        # Check if this is an Athabasca-type glacier (use pivot-based approach)
        if glacier_config.get('data_type') == 'athabasca_multiproduct':
            return self._load_pivot_based_data(glacier_config)
        
        # Original approach for other glaciers
        # Load MODIS data
        modis_products = self.config['analysis']['albedo']['modis_products']
        data_dict['modis'] = {}
        
        for product in modis_products:
            if product in glacier_config['data_files']['modis']:
                file_path = os.path.join(
                    self.config['data']['modis_path'],
                    glacier_config['data_files']['modis'][product]
                )
                
                if os.path.exists(file_path):
                    try:
                        loader = create_pivot_based_loader(product, self.config, glacier_config)
                        data_dict['modis'][product] = loader.load_data(file_path)
                        self.logger.info(f"Loaded {product}: {len(data_dict['modis'][product])} records")
                    except ValueError:
                        # Fallback to enhanced loader for non-pivot methods
                        from src.data.enhanced_loaders import create_enhanced_loader
                        loader = create_enhanced_loader(product, self.config, glacier_config)
                        data_dict['modis'][product] = loader.load_data(file_path)
                        self.logger.info(f"Loaded {product}: {len(data_dict['modis'][product])} records")
                else:
                    self.logger.warning(f"MODIS file not found: {file_path}")
        
        # Load AWS data
        aws_file_path = os.path.join(
            self.config['data']['aws_path'],
            glacier_config['data_files']['aws']
        )
        
        if os.path.exists(aws_file_path):
            try:
                aws_loader = create_pivot_based_loader('AWS', self.config, None)
                data_dict['aws'] = aws_loader.load_data(aws_file_path)
            except ValueError:
                # Fallback to enhanced loader
                from src.data.enhanced_loaders import create_enhanced_loader
                aws_loader = create_enhanced_loader('AWS', self.config, glacier_config)
                data_dict['aws'] = aws_loader.load_data(aws_file_path)
            
            self.logger.info(f"Loaded AWS data: {len(data_dict['aws'])} records")
        else:
            self.logger.warning(f"AWS file not found: {aws_file_path}")
            data_dict['aws'] = pd.DataFrame()
        
        # Load glacier mask
        mask_file_path = os.path.join(
            self.config['data']['glacier_masks_path'],
            glacier_config['data_files']['mask']
        )
        
        if os.path.exists(mask_file_path):
            data_dict['glacier_mask'] = self.mask_processor.load_and_validate_mask(mask_file_path)
            self.logger.info(f"Loaded glacier mask: {len(data_dict['glacier_mask'])} features")
        else:
            self.logger.warning(f"Glacier mask not found: {mask_file_path}")
            data_dict['glacier_mask'] = gpd.GeoDataFrame()
        
        return data_dict
    
    def _load_pivot_based_data(self, glacier_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load data using pivot-based approach for Athabasca-type glaciers."""
        # Find the MultiProduct_with_AWS.csv file
        modis_path = self.config['data']['modis_path']
        multiproduct_file = None
        
        for filename in os.listdir(modis_path):
            if 'MultiProduct_with_AWS.csv' in filename and glacier_config['name'].lower().split()[0].lower() in filename.lower():
                multiproduct_file = os.path.join(modis_path, filename)
                break
        
        if not multiproduct_file:
            raise FileNotFoundError(f"MultiProduct_with_AWS.csv file not found for {glacier_config['name']}")
        
        # Load MODIS data
        modis_loader = create_pivot_based_loader("athabasca_multiproduct", self.config, glacier_config)
        modis_data = modis_loader.load_data(multiproduct_file)
        
        # Load AWS data (use separate file)
        aws_loader = create_pivot_based_loader("AWS", self.config, None)
        if 'aws' in glacier_config.get('data_files', {}):
            aws_file_path = os.path.join(self.config['data']['aws_path'], glacier_config['data_files']['aws'])
            aws_data = aws_loader.load_data(aws_file_path)
        else:
            aws_data = pd.DataFrame()
        
        # Apply Terra/Aqua merge and create pivot table
        modis_merged = self.pivot_processor.apply_terra_aqua_merge(modis_data)
        merged_data = self.pivot_processor.create_pivot_and_merge(modis_merged, aws_data)
        
        # Return in the format expected by the rest of the pipeline
        return {
            'merged_data': merged_data,
            'modis_raw': modis_data,
            'aws_raw': aws_data,
            'glacier_mask': gpd.GeoDataFrame()  # Placeholder
        }
    
    def _process_spatial_data(self, data_dict: Dict[str, Any], 
                            glacier_config: Dict[str, Any]) -> Dict[str, Any]:
        """Process spatial aspects of the data."""
        spatial_results = {}
        
        # Mask MODIS data to glacier extent
        if not data_dict['glacier_mask'].empty:
            spatial_results['modis_masked'] = {}
            
            for product, modis_data in data_dict['modis'].items():
                if not modis_data.empty:
                    masked_data = self.mask_processor.mask_modis_data(
                        modis_data, data_dict['glacier_mask']
                    )
                    spatial_results['modis_masked'][product] = masked_data
        
        # Create AWS buffer zones
        aws_coordinates = {}
        if 'aws_stations' in glacier_config:
            aws_coordinates = {
                station_id: {
                    'lat': station_info.get('lat'),
                    'lon': station_info.get('lon'),
                    'elevation': station_info.get('elevation')
                }
                for station_id, station_info in glacier_config['aws_stations'].items()
            }
        
        spatial_results['aws_coordinates'] = aws_coordinates
        
        if aws_coordinates:
            spatial_results['aws_buffers'] = self.spatial_processor.create_aws_buffer_zones(
                aws_coordinates
            )
        
        return spatial_results
    
    def _align_temporal_data(self, data_dict: Dict[str, Any]) -> Dict[str, pd.Series]:
        """Align temporal data for comparison."""
        aligned_data = {}
        
        # Handle pivot-based data differently
        if 'merged_data' in data_dict:
            # Pivot-based approach - data is already aligned
            merged_data = data_dict['merged_data']
            modis_methods = [col for col in merged_data.columns if col in ['MCD43A3', 'MOD09GA', 'MOD10A1']]
            
            for method in modis_methods:
                if method in merged_data.columns and 'AWS' in merged_data.columns:
                    valid_data = merged_data[[method, 'AWS']].dropna()
                    if len(valid_data) > 0:
                        aligned_data[f'{method}_aws'] = valid_data['AWS']
                        aligned_data[method] = valid_data[method]
                        self.logger.info(f"Aligned {method}: {len(valid_data)} common dates")
            
            return aligned_data
        
        # Original approach for other glaciers
        # AWS data as reference
        if not data_dict['aws'].empty:
            aws_daily = data_dict['aws'].set_index('date')['albedo']
            
            # Store method-specific aligned pairs instead of trying to align all methods to one AWS reference
            for product, modis_data in data_dict['modis'].items():
                if modis_data.empty:
                    continue
                    
                # Convert to daily means
                modis_daily = modis_data.groupby('date')['albedo'].mean()
                
                # Find common dates with AWS
                common_dates = aws_daily.index.intersection(modis_daily.index)
                
                if len(common_dates) > 0:
                    # Store both AWS and MODIS data for these specific common dates
                    aligned_data[f'{product}_aws'] = aws_daily.loc[common_dates]
                    aligned_data[product] = modis_daily.loc[common_dates]
                    
                    self.logger.info(f"Aligned {product}: {len(common_dates)} common dates")
                else:
                    self.logger.warning(f"No common dates found for {product}")
        
        return aligned_data
    
    def _perform_statistical_analysis(self, aligned_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Perform statistical analysis using dependency-free approach."""
        statistics = {}
        
        # Get available MODIS products (those without '_aws' suffix)
        modis_products = [k for k in aligned_data.keys() if not k.endswith('_aws')]
        
        if not modis_products:
            self.logger.warning("No MODIS data available for statistical analysis")
            return statistics
        
        # Calculate metrics for each method
        statistics['method_comparison'] = {}
        
        for product in modis_products:
            aws_key = f'{product}_aws'
            
            if product in aligned_data and aws_key in aligned_data:
                aws_data = aligned_data[aws_key]
                modis_data = aligned_data[product]
                
                if len(modis_data) > 0 and len(aws_data) > 0:
                    metrics = self._calculate_basic_metrics(aws_data, modis_data)
                    statistics['method_comparison'][product] = metrics
                    
                    self.logger.info(f"\n{product} Statistics:")
                    self.logger.info(f"  RMSE: {metrics['rmse']:.4f}")
                    self.logger.info(f"  Bias: {metrics['bias']:.4f}")
                    self.logger.info(f"  R²: {metrics['r2']:.4f}")
                    self.logger.info(f"  Correlation: {metrics['correlation']:.4f}")
                    self.logger.info(f"  n samples: {metrics['n_samples']}")
        
        # Create method ranking
        if len(statistics['method_comparison']) > 1:
            ranking_data = []
            for method, metrics in statistics['method_comparison'].items():
                ranking_data.append({
                    'method': method,
                    'rmse': metrics['rmse'],
                    'bias': abs(metrics['bias']),
                    'correlation': metrics['correlation'],
                    'n_samples': metrics['n_samples']
                })
            
            ranking_df = pd.DataFrame(ranking_data)
            ranking_df['rmse_rank'] = ranking_df['rmse'].rank()
            ranking_df['bias_rank'] = ranking_df['bias'].rank() 
            ranking_df['corr_rank'] = ranking_df['correlation'].rank(ascending=False)
            ranking_df['overall_rank'] = (ranking_df['rmse_rank'] + ranking_df['bias_rank'] + ranking_df['corr_rank']) / 3
            
            statistics['ranking'] = ranking_df.sort_values('overall_rank')
            
            self.logger.info(f"\nMethod Ranking:")
            for i, row in statistics['ranking'].iterrows():
                self.logger.info(f"  {int(row['overall_rank'])}. {row['method']} (RMSE: {row['rmse']:.4f})")
        
        return statistics
    
    def _calculate_basic_metrics(self, observed, predicted):
        """Calculate basic statistical metrics without sklearn."""
        # Remove NaN values
        mask = ~(pd.isna(observed) | pd.isna(predicted))
        obs = observed[mask]
        pred = predicted[mask]
        
        if len(obs) == 0:
            return {'n_samples': 0, 'rmse': np.nan, 'bias': np.nan, 'r2': np.nan, 'correlation': np.nan}
        
        # Calculate metrics without sklearn
        rmse = np.sqrt(np.mean((obs - pred) ** 2))
        bias = np.mean(pred - obs)
        correlation = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 else np.nan
        
        # Calculate R² manually
        ss_res = np.sum((obs - pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan
        
        return {
            'n_samples': len(obs),
            'rmse': rmse,
            'bias': bias,
            'r2': r2,
            'correlation': correlation,
            'mean_obs': np.mean(obs),
            'mean_pred': np.mean(pred),
            'std_obs': np.std(obs),
            'std_pred': np.std(pred)
        }
    
    def _generate_plots(self, aligned_data: Dict[str, pd.Series], 
                       statistics: Dict[str, Any],
                       glacier_id: str, 
                       output_dir: str):
        """Generate all plots for the glacier."""
        plots_dir = os.path.join(output_dir, 'plots')
        
        if not self.plot_generator:
            self.logger.warning("PlotGenerator not available - skipping visualization")
            return
        
        # Get available MODIS products (those without '_aws' suffix)
        modis_products = [k for k in aligned_data.keys() if not k.endswith('_aws')]
        
        if not modis_products:
            self.logger.warning("No MODIS data available for plotting")
            return
        
        try:
            # Individual scatterplots
            for product in modis_products:
                aws_key = f'{product}_aws'
                
                if product in aligned_data and aws_key in aligned_data:
                    aws_data = aligned_data[aws_key]
                    modis_data = aligned_data[product]
                    
                    if len(modis_data) > 0 and len(aws_data) > 0:
                        output_path = os.path.join(plots_dir, f"{glacier_id}_{product}_scatterplot.png")
                        
                        self.plot_generator.create_scatterplot(
                            aws_data, modis_data,
                            x_label="AWS Albedo",
                            y_label=f"{product} Albedo",
                            title=f"{glacier_id.title()}: {product} vs AWS",
                            method_name=product,
                            show_stats=True,
                            add_regression_line=True,
                            add_identity_line=True,
                            output_path=output_path
                        )
                        
                        self.logger.info(f"Created scatterplot: {output_path}")
            
            # Additional plots can be added here following the same pattern
            self.logger.info("Plot generation completed")
            
        except Exception as e:
            self.logger.warning(f"Plot generation failed: {e}")
            self.logger.info("Continuing analysis without visualizations...")
    
    def _generate_maps(self, spatial_results: Dict[str, Any],
                      glacier_config: Dict[str, Any],
                      glacier_id: str,
                      output_dir: str):
        """Generate all maps for the glacier."""
        maps_dir = os.path.join(output_dir, 'maps')
        
        glacier_mask = spatial_results.get('glacier_mask', gpd.GeoDataFrame())
        aws_coordinates = spatial_results.get('aws_coordinates', {})
        
        # Overview map
        if not glacier_mask.empty:
            output_path = os.path.join(maps_dir, f"{glacier_id}_overview_map.png")
            # Note: This would need actual glacier mask from data_dict
            # self.map_generator.create_glacier_overview_map(
            #     glacier_mask, aws_coordinates,
            #     title=f"{glacier_id}: Overview Map",
            #     output_path=output_path
            # )
        
        # Validation setup map
        modis_data = spatial_results.get('modis_masked', {})
        if modis_data:
            # Combine all MODIS data for visualization
            all_modis_points = pd.DataFrame()
            for product, data in modis_data.items():
                if not data.empty and 'lat' in data.columns and 'lon' in data.columns:
                    data_copy = data[['lat', 'lon', 'albedo']].copy()
                    data_copy['product'] = product
                    all_modis_points = pd.concat([all_modis_points, data_copy], ignore_index=True)
            
            if not all_modis_points.empty:
                # Convert to GeoDataFrame
                modis_gdf = gpd.GeoDataFrame(
                    all_modis_points,
                    geometry=gpd.points_from_xy(all_modis_points.lon, all_modis_points.lat),
                    crs='EPSG:4326'
                )
                
                output_path = os.path.join(maps_dir, f"{glacier_id}_validation_map.png")
                # Note: This would need the glacier mask from data_dict
                # self.map_generator.create_validation_map(
                #     modis_gdf, aws_coordinates,
                #     title=f"{glacier_id}: Validation Setup",
                #     output_path=output_path
                # )
    
    def _export_results(self, statistics: Dict[str, Any], 
                       glacier_id: str,
                       output_dir: str) -> Dict[str, Any]:
        """Export results to files."""
        results_dir = os.path.join(output_dir, 'results')
        
        # Export comparison metrics
        if 'method_comparison' in statistics:
            comparison_df = pd.DataFrame(statistics['method_comparison']).T
            comparison_file = os.path.join(results_dir, f"{glacier_id}_method_comparison.csv")
            comparison_df.to_csv(comparison_file)
            
        # Export ranking if available
        if 'ranking' in statistics:
            ranking_file = os.path.join(results_dir, f"{glacier_id}_method_ranking.csv")
            statistics['ranking'].to_csv(ranking_file, index=False)
        
        # Export significance tests
        if 'significance_tests' in statistics:
            sig_df = pd.DataFrame(statistics['significance_tests']).T
            sig_file = os.path.join(results_dir, f"{glacier_id}_significance_tests.csv")
            sig_df.to_csv(sig_file)
        
        # Create summary
        summary = {
            'glacier_id': glacier_id,
            'analysis_timestamp': get_timestamp(),
            'output_directory': output_dir,
            'statistics': statistics
        }
        
        return summary


def main():
    """Main function to run the analysis."""
    parser = argparse.ArgumentParser(description='MODIS Albedo Analysis Framework')
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
        pipeline = AlbedoAnalysisPipeline(args.config)
        
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
                    'analysis_timestamp': result['analysis_timestamp'],
                    'output_directory': result['output_directory']
                }
                for result in all_results
            ])
            summary_df.to_csv(args.output_summary, index=False)
            print(f"Summary exported to: {args.output_summary}")
        
        print(f"Analysis completed for {len(all_results)} glacier(s)")
        
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()