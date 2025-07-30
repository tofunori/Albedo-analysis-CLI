#!/usr/bin/env python3
"""
Multi-Glacier Mapping Module for Comparative Analysis

This module provides comprehensive mapping capabilities for the multi-glacier
comparative analysis framework, including individual glacier maps, overview maps,
and spatial correlation visualizations.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.mapping.glacier_masks import GlacierMaskProcessor
from src.utils.helpers import load_config

# Set up logging
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8')


class MultiGlacierMapper:
    """
    Comprehensive mapping suite for multi-glacier comparative analysis.
    
    Provides functionality for:
    - Individual glacier maps with masks, AWS stations, and MODIS coverage
    - Multi-glacier overview maps
    - Spatial correlation analysis maps
    - Integration with comparative analysis framework
    """
    
    def __init__(self, config_path: str = "config/glacier_sites.yaml", 
                 figsize_base: Tuple[int, int] = (12, 8), dpi: int = 300):
        """Initialize the multi-glacier mapper."""
        self.config_path = config_path
        self.figsize_base = figsize_base
        self.dpi = dpi
        
        # Load glacier configuration
        self.glacier_config = load_config(config_path)
        self.glaciers = self.glacier_config['glaciers']
        
        # Initialize mask processor
        self.mask_processor = GlacierMaskProcessor(self.glacier_config)
        
        # Color scheme for consistent visualization
        self.colors = {
            'athabasca': '#1f77b4',   # Blue
            'haig': '#ff7f0e',        # Orange  
            'coropuna': '#2ca02c',    # Green
            'aws': '#d62728',         # Red for AWS stations
            'modis': '#9467bd',       # Purple for MODIS pixels
            'mask': '#17becf'         # Cyan for glacier masks
        }
        
        # Regional colors
        self.region_colors = {
            'Canadian Rockies': '#2E8B57',  # Sea Green
            'Peruvian Andes': '#DAA520'     # Goldenrod
        }
        
        logger.info("Multi-Glacier Mapper initialized")
    
    def load_glacier_mask(self, glacier_id: str) -> Optional[gpd.GeoDataFrame]:
        """Load glacier mask for a specific glacier."""
        try:
            glacier_config = self.glaciers[glacier_id]
            mask_file = glacier_config['data_files']['mask']
            
            # Handle different mask file specifications
            if mask_file.startswith('data/'):
                mask_path = Path(mask_file)
            else:
                mask_path = Path('data') / 'glacier_masks' / glacier_id / mask_file
            
            if mask_path.exists():
                mask_gdf = self.mask_processor.load_and_validate_mask(str(mask_path))
                logger.info(f"Loaded mask for {glacier_id}: {len(mask_gdf)} features")
                return mask_gdf
            else:
                logger.warning(f"Mask file not found for {glacier_id}: {mask_path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading mask for {glacier_id}: {e}")
            return None
    
    def get_aws_coordinates(self, glacier_id: str) -> Dict[str, Dict[str, float]]:
        """Get AWS station coordinates for a glacier."""
        try:
            glacier_config = self.glaciers[glacier_id]
            aws_stations = glacier_config.get('aws_stations', {})
            
            coordinates = {}
            for station_id, station_config in aws_stations.items():
                coordinates[station_id] = {
                    'lat': station_config.get('lat'),
                    'lon': station_config.get('lon'),
                    'elevation': station_config.get('elevation'),
                    'name': station_config.get('name', station_id)
                }
            
            return coordinates
            
        except Exception as e:
            logger.error(f"Error getting AWS coordinates for {glacier_id}: {e}")
            return {}
    
    def load_modis_sample_data(self, glacier_id: str, sample_size: int = 1000) -> Optional[pd.DataFrame]:
        """Load a sample of MODIS data for visualization."""
        try:
            from src.analysis.comparative_analysis import MultiGlacierComparativeAnalysis
            
            # Get merged data file path
            analyzer = MultiGlacierComparativeAnalysis()
            glacier_results = analyzer.discover_latest_results()
            
            if glacier_id not in glacier_results:
                logger.warning(f"No analysis results found for {glacier_id}")
                return None
            
            merged_file = glacier_results[glacier_id].get('merged_data')
            if not merged_file or not Path(merged_file).exists():
                logger.warning(f"No merged data file found for {glacier_id}")
                return None
            
            # Load and sample the data
            modis_data = pd.read_csv(merged_file)
            
            # Sample data for visualization (too many points can be slow)
            if len(modis_data) > sample_size:
                modis_data = modis_data.sample(n=sample_size, random_state=42)
            
            # Ensure we have required columns
            required_cols = ['MCD43A3', 'MOD09GA', 'MOD10A1']
            available_cols = [col for col in required_cols if col in modis_data.columns]
            
            if available_cols:
                # Add a representative albedo column (using first available method)
                modis_data['albedo'] = modis_data[available_cols[0]]
                return modis_data
            else:
                logger.warning(f"No MODIS albedo columns found in {glacier_id} data")
                return None
                
        except Exception as e:
            logger.error(f"Error loading MODIS data for {glacier_id}: {e}")
            return None
    
    def create_individual_glacier_map(self, glacier_id: str, output_dir: Path) -> Optional[plt.Figure]:
        """Create detailed map for an individual glacier."""
        logger.info(f"Creating individual map for {glacier_id}")
        
        # Load data
        mask_gdf = self.load_glacier_mask(glacier_id)
        aws_coords = self.get_aws_coordinates(glacier_id)
        modis_data = self.load_modis_sample_data(glacier_id)
        
        if mask_gdf is None:
            logger.warning(f"Cannot create map for {glacier_id} - no mask data")
            return None
        
        # Create figure with regular matplotlib (no cartopy for now)
        fig, ax = plt.subplots(1, 1, figsize=self.figsize_base)
        
        # Set background color
        ax.set_facecolor('lightgray')
        
        # Plot glacier mask
        mask_gdf.plot(ax=ax, color=self.colors['mask'], alpha=0.7,
                     edgecolor='darkblue', linewidth=2, 
                     label='Glacier extent')
        
        # Plot MODIS pixels if available
        if modis_data is not None and not modis_data.empty:
            # Check for coordinate columns
            lat_col = 'lat' if 'lat' in modis_data.columns else 'latitude'
            lon_col = 'lon' if 'lon' in modis_data.columns else 'longitude'
            
            if lat_col in modis_data.columns and lon_col in modis_data.columns:
                # Color by albedo if available
                if 'albedo' in modis_data.columns:
                    scatter = ax.scatter(modis_data[lon_col], modis_data[lat_col],
                                       c=modis_data['albedo'], cmap='RdYlBu_r',
                                       s=15, alpha=0.6, transform=ccrs.PlateCarree(),
                                       label='MODIS pixels', vmin=0, vmax=1)
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
                    cbar.set_label('Albedo', rotation=270, labelpad=15)
                else:
                    ax.scatter(modis_data[lon_col], modis_data[lat_col],
                             c=self.colors['modis'], s=15, alpha=0.6,
                             transform=ccrs.PlateCarree(), label='MODIS pixels')
        
        # Plot AWS stations
        if aws_coords:
            for station_id, coords in aws_coords.items():
                if coords['lat'] is not None and coords['lon'] is not None:
                    ax.scatter(coords['lon'], coords['lat'], 
                             c=self.colors['aws'], s=150, marker='^',
                             label='AWS station', zorder=5, 
                             transform=ccrs.PlateCarree(),
                             edgecolors='white', linewidth=2)
                    
                    # Add station label
                    ax.annotate(coords['name'], (coords['lon'], coords['lat']),
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=10, ha='left', fontweight='bold',
                               transform=ccrs.PlateCarree(),
                               bbox=dict(boxstyle='round,pad=0.3', 
                                        facecolor='white', alpha=0.8))
        
        # Set map extent
        bounds = mask_gdf.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.2
        ax.set_extent([bounds[0] - buffer, bounds[2] + buffer,
                       bounds[1] - buffer, bounds[3] + buffer],
                      crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        # Customize plot
        glacier_name = self.glaciers[glacier_id]['name']
        region = self.glaciers[glacier_id]['region']
        ax.set_title(f'{glacier_name}\n{region}', fontsize=14, fontweight='bold', pad=20)
        
        # Add legend (remove duplicates)
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / "plots" / f"map_individual_{glacier_id}.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved individual map for {glacier_id}: {output_file}")
        
        return fig
    
    def create_multi_glacier_overview_map(self, output_dir: Path) -> Optional[plt.Figure]:
        """Create overview map showing all glaciers."""
        logger.info("Creating multi-glacier overview map")
        
        # Load all glacier data
        all_masks = {}
        all_aws_coords = {}
        
        for glacier_id in self.glaciers.keys():
            mask_gdf = self.load_glacier_mask(glacier_id)
            if mask_gdf is not None:
                all_masks[glacier_id] = mask_gdf
            
            aws_coords = self.get_aws_coordinates(glacier_id)
            if aws_coords:
                all_aws_coords[glacier_id] = aws_coords
        
        if not all_masks:
            logger.error("No glacier masks available for overview map")
            return None
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(16, 10),
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add map features for global view
        ax.add_feature(cfeature.COASTLINE, alpha=0.8, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, alpha=0.6, linewidth=0.6)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.4)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.4)
        ax.add_feature(cfeature.LAKES, color='lightblue', alpha=0.6)
        
        # Plot each glacier
        legend_elements = []
        
        for glacier_id, mask_gdf in all_masks.items():
            glacier_name = self.glaciers[glacier_id]['name']
            color = self.colors.get(glacier_id, 'blue')
            
            # Plot glacier mask
            mask_gdf.plot(ax=ax, color=color, alpha=0.8,
                         edgecolor='black', linewidth=2,
                         transform=ccrs.PlateCarree())
            
            # Add to legend
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color,
                                               edgecolor='black', alpha=0.8,
                                               label=glacier_name))
            
            # Plot AWS stations
            if glacier_id in all_aws_coords:
                for station_id, coords in all_aws_coords[glacier_id].items():
                    if coords['lat'] is not None and coords['lon'] is not None:
                        ax.scatter(coords['lon'], coords['lat'], 
                                 c='red', s=100, marker='^',
                                 zorder=5, transform=ccrs.PlateCarree(),
                                 edgecolors='white', linewidth=1)
        
        # Add AWS legend element
        legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                         markerfacecolor='red', markersize=10,
                                         label='AWS stations'))
        
        # Set global extent
        all_bounds = []
        for mask_gdf in all_masks.values():
            all_bounds.append(mask_gdf.total_bounds)
        
        if all_bounds:
            min_lon = min(bounds[0] for bounds in all_bounds)
            min_lat = min(bounds[1] for bounds in all_bounds)
            max_lon = max(bounds[2] for bounds in all_bounds)
            max_lat = max(bounds[3] for bounds in all_bounds)
            
            # Add buffer
            lon_buffer = (max_lon - min_lon) * 0.3
            lat_buffer = (max_lat - min_lat) * 0.3
            
            ax.set_extent([min_lon - lon_buffer, max_lon + lon_buffer,
                           min_lat - lat_buffer, max_lat + lat_buffer],
                          crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        # Customize plot
        ax.set_title('Multi-Glacier Comparative Analysis Overview\nMODIS Albedo Validation Sites', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Add legend
        ax.legend(handles=legend_elements, loc='upper left', 
                 bbox_to_anchor=(0.02, 0.98), frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / "plots" / "map_multi_glacier_overview.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        logger.info(f"Saved multi-glacier overview map: {output_file}")
        
        return fig
    
    def create_spatial_correlation_maps(self, glacier_id: str, output_dir: Path) -> List[plt.Figure]:
        """Create spatial correlation maps for each MODIS method."""
        logger.info(f"Creating spatial correlation maps for {glacier_id}")
        
        figures = []
        
        # Load data
        mask_gdf = self.load_glacier_mask(glacier_id)
        aws_coords = self.get_aws_coordinates(glacier_id)
        modis_data = self.load_modis_sample_data(glacier_id, sample_size=2000)  # More points for correlation analysis
        
        if mask_gdf is None or modis_data is None:
            logger.warning(f"Cannot create correlation maps for {glacier_id} - missing data")
            return figures
        
        # MODIS methods to analyze
        methods = ['MCD43A3', 'MOD09GA', 'MOD10A1']
        available_methods = [m for m in methods if m in modis_data.columns]
        
        if not available_methods:
            logger.warning(f"No MODIS methods available for {glacier_id}")
            return figures
        
        # Check for coordinate and AWS columns
        lat_col = 'lat' if 'lat' in modis_data.columns else 'latitude'
        lon_col = 'lon' if 'lon' in modis_data.columns else 'longitude'
        aws_col = 'AWS'
        
        if lat_col not in modis_data.columns or lon_col not in modis_data.columns:
            logger.warning(f"Missing coordinate columns for {glacier_id}")
            return figures
        
        if aws_col not in modis_data.columns:
            logger.warning(f"Missing AWS column for {glacier_id}")
            return figures
        
        # Create correlation maps for each method
        for method in available_methods:
            # Filter valid data (both AWS and MODIS available)
            valid_data = modis_data.dropna(subset=[aws_col, method])
            
            if len(valid_data) < 10:  # Need minimum data for meaningful correlation
                logger.warning(f"Insufficient data for {glacier_id} {method} correlation map")
                continue
            
            # Calculate correlation coefficient for each point (local correlation within moving window)
            # For visualization, we'll show the residuals (MODIS - AWS)
            valid_data = valid_data.copy()
            valid_data['residuals'] = valid_data[method] - valid_data[aws_col]
            valid_data['abs_residuals'] = np.abs(valid_data['residuals'])
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=self.figsize_base,
                                  subplot_kw={'projection': ccrs.PlateCarree()})
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, alpha=0.8, linewidth=0.5)
            ax.add_feature(cfeature.BORDERS, alpha=0.5, linewidth=0.5)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
            
            # Plot glacier mask
            mask_gdf.plot(ax=ax, facecolor='none', edgecolor='darkblue',
                         linewidth=2, alpha=0.8, transform=ccrs.PlateCarree(),
                         label='Glacier extent')
            
            # Plot residuals with symmetric color scale
            abs_max = valid_data['residuals'].abs().quantile(0.95)
            norm = Normalize(vmin=-abs_max, vmax=abs_max)
            
            scatter = ax.scatter(valid_data[lon_col], valid_data[lat_col],
                               c=valid_data['residuals'], cmap='RdBu_r', norm=norm,
                               s=20, alpha=0.7, transform=ccrs.PlateCarree(),
                               edgecolors='black', linewidth=0.1)
            
            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
            cbar.set_label('MODIS - AWS Albedo', rotation=270, labelpad=15)
            
            # Plot AWS stations
            if aws_coords:
                for station_id, coords in aws_coords.items():
                    if coords['lat'] is not None and coords['lon'] is not None:
                        ax.scatter(coords['lon'], coords['lat'], 
                                 c='red', s=150, marker='^',
                                 label='AWS station', zorder=5, 
                                 transform=ccrs.PlateCarree(),
                                 edgecolors='white', linewidth=2)
            
            # Set map extent
            bounds = mask_gdf.total_bounds
            buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
            ax.set_extent([bounds[0] - buffer, bounds[2] + buffer,
                           bounds[1] - buffer, bounds[3] + buffer],
                          crs=ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            
            # Calculate statistics for title
            correlation = np.corrcoef(valid_data[aws_col], valid_data[method])[0, 1]
            rmse = np.sqrt(np.mean(valid_data['residuals']**2))
            bias = np.mean(valid_data['residuals'])
            
            # Customize plot
            glacier_name = self.glaciers[glacier_id]['name']
            ax.set_title(f'{glacier_name} - {method}\nSpatial Residuals (MODIS - AWS)\n'
                        f'R = {correlation:.3f}, RMSE = {rmse:.3f}, Bias = {bias:.3f}',
                        fontsize=12, fontweight='bold')
            
            # Add legend (remove duplicates)
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if by_label:
                ax.legend(by_label.values(), by_label.keys(), loc='upper right')
            
            plt.tight_layout()
            
            # Save plot
            output_file = output_dir / "plots" / f"map_correlation_{glacier_id}_{method}.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved correlation map: {output_file}")
            
            figures.append(fig)
        
        return figures
    
    def generate_all_maps(self, output_dir: Path) -> None:
        """Generate all mapping visualizations."""
        logger.info("Starting generation of all mapping visualizations...")
        
        try:
            # Create individual glacier maps
            for glacier_id in self.glaciers.keys():
                fig = self.create_individual_glacier_map(glacier_id, output_dir)
                if fig:
                    plt.close(fig)  # Free memory
            
            # Create multi-glacier overview map
            overview_fig = self.create_multi_glacier_overview_map(output_dir)
            if overview_fig:
                plt.close(overview_fig)
            
            # Create spatial correlation maps
            for glacier_id in self.glaciers.keys():
                correlation_figs = self.create_spatial_correlation_maps(glacier_id, output_dir)
                for fig in correlation_figs:
                    plt.close(fig)  # Free memory
            
            logger.info("Successfully generated all mapping visualizations")
            
        except Exception as e:
            logger.error(f"Error generating maps: {e}")
            raise


def main():
    """Test the multi-glacier mapping suite."""
    mapper = MultiGlacierMapper()
    
    # Test loading masks
    for glacier_id in mapper.glaciers.keys():
        mask = mapper.load_glacier_mask(glacier_id)
        print(f"{glacier_id}: {'✓' if mask is not None else '✗'} mask loaded")
        
        aws_coords = mapper.get_aws_coordinates(glacier_id)
        print(f"{glacier_id}: {len(aws_coords)} AWS stations")


if __name__ == "__main__":
    main()