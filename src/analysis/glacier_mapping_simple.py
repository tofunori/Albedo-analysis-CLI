#!/usr/bin/env python3
"""
Simplified Multi-Glacier Mapping Module (No Cartopy Required)

This module provides basic mapping capabilities for the multi-glacier
comparative analysis framework without requiring cartopy.
"""

import os
import sys
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
import warnings
warnings.filterwarnings('ignore')
import math

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.utils.helpers import load_config

# Set up logging
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8')


class MultiGlacierMapperSimple:
    """
    Simplified mapping suite for multi-glacier comparative analysis.
    
    Works without cartopy dependency, using basic matplotlib plotting.
    """
    
    def __init__(self, config_path: str = "config/glacier_sites.yaml", 
                 figsize_base: Tuple[int, int] = (12, 8), dpi: int = 300):
        """Initialize the multi-glacier mapper."""
        self.config_path = config_path
        self.figsize_base = figsize_base
        self.dpi = dpi
        
        # Load glacier configuration
        try:
            self.glacier_config = load_config(config_path)
            self.glaciers = self.glacier_config['glaciers']
            logger.info(f"Loaded configuration for {len(self.glaciers)} glaciers")
        except Exception as e:
            logger.error(f"Failed to load glacier configuration: {e}")
            self.glaciers = {}
        
        # Color scheme for consistent visualization
        self.colors = {
            'athabasca': '#1f77b4',   # Blue
            'haig': '#ff7f0e',        # Orange  
            'coropuna': '#2ca02c',    # Green
            'aws': '#d62728',         # Red for AWS stations
            'modis': '#9467bd',       # Purple for MODIS pixels
            'mask': '#17becf'         # Cyan for glacier masks
        }
        
        logger.info("Simple Multi-Glacier Mapper initialized")
    
    @staticmethod
    def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth using the Haversine formula.
        
        Args:
            lat1, lon1: Latitude and longitude of first point in decimal degrees
            lat2, lon2: Latitude and longitude of second point in decimal degrees
            
        Returns:
            Distance in kilometers
        """
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        
        return c * r
    
    def select_best_pixel_for_analysis(self, glacier_id: str, modis_data: pd.DataFrame, 
                                     aws_coords: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Select the best performing pixel closest to AWS station for analysis.
        
        For Athabasca (≤2 pixels): Returns all pixels
        For Haig and Coropuna (many pixels): Returns single best pixel based on:
        - Distance to AWS station (closer is better)
        - Glacier fraction (higher is better, more glacier coverage)
        - Composite score combining both factors
        
        Args:
            glacier_id: Glacier identifier
            modis_data: DataFrame with unique pixel locations and data
            aws_coords: AWS station coordinates
            
        Returns:
            DataFrame with selected pixel(s) for analysis
        """
        try:
            # For Athabasca or small pixel counts, return all pixels
            if glacier_id == 'athabasca' or len(modis_data) <= 2:
                logger.info(f"{glacier_id}: Using all {len(modis_data)} pixels for analysis (small dataset)")
                return modis_data
            
            if not aws_coords:
                logger.warning(f"{glacier_id}: No AWS coordinates available, using all pixels")
                return modis_data
            
            # Get AWS coordinates (use first available station)
            aws_station = list(aws_coords.values())[0]
            aws_lat = aws_station['lat']
            aws_lon = aws_station['lon']
            
            if aws_lat is None or aws_lon is None:
                logger.warning(f"{glacier_id}: Invalid AWS coordinates, using all pixels")
                return modis_data
            
            # Calculate distance from each pixel to AWS station
            modis_data = modis_data.copy()
            modis_data['distance_to_aws_km'] = modis_data.apply(
                lambda row: self.calculate_haversine_distance(
                    row['latitude'], row['longitude'], aws_lat, aws_lon
                ), axis=1
            )
            
            # Define performance metrics
            # 1. Distance score (closer is better, normalized)
            max_distance = modis_data['distance_to_aws_km'].max()
            min_distance = modis_data['distance_to_aws_km'].min()
            if max_distance > min_distance:
                modis_data['distance_score'] = 1 - (modis_data['distance_to_aws_km'] - min_distance) / (max_distance - min_distance)
            else:
                modis_data['distance_score'] = 1.0
            
            # 2. Glacier fraction score (higher is better)
            glacier_fraction_score = 0.5  # Default if no glacier_fraction column
            if 'glacier_fraction' in modis_data.columns:
                max_fraction = modis_data['glacier_fraction'].max()
                if max_fraction > 0:
                    modis_data['glacier_fraction_score'] = modis_data['glacier_fraction'] / max_fraction
                else:
                    modis_data['glacier_fraction_score'] = 0.5
                glacier_fraction_weight = 0.4
            else:
                modis_data['glacier_fraction_score'] = 0.5
                glacier_fraction_weight = 0.0
            
            # 3. Composite score: 60% distance + 40% glacier fraction
            distance_weight = 0.6
            modis_data['composite_score'] = (
                distance_weight * modis_data['distance_score'] + 
                glacier_fraction_weight * modis_data['glacier_fraction_score'] +
                (1 - distance_weight - glacier_fraction_weight) * 0.5  # Base score for other factors
            )
            
            # Select pixel with highest composite score
            best_pixel_idx = modis_data['composite_score'].idxmax()
            best_pixel = modis_data.loc[[best_pixel_idx]]
            
            # Log selection details
            distance = best_pixel['distance_to_aws_km'].iloc[0]
            glacier_frac = best_pixel.get('glacier_fraction', [None]).iloc[0]
            pixel_id = best_pixel.get('pixel_id', ['unknown']).iloc[0]
            
            logger.info(f"{glacier_id}: Selected best pixel for analysis:")
            logger.info(f"  - Pixel ID: {pixel_id}")
            logger.info(f"  - Distance to AWS: {distance:.2f} km")
            if glacier_frac is not None:
                logger.info(f"  - Glacier fraction: {glacier_frac:.3f}")
            logger.info(f"  - Selected from {len(modis_data)} available pixels")
            
            # Return only the essential columns (clean up temporary scoring columns)
            essential_cols = ['pixel_id', 'longitude', 'latitude', 'albedo']
            if 'glacier_fraction' in modis_data.columns:
                essential_cols.append('glacier_fraction')
            
            return best_pixel[essential_cols]
            
        except Exception as e:
            logger.error(f"Error selecting best pixel for {glacier_id}: {e}")
            logger.info(f"Falling back to using all pixels for {glacier_id}")
            return modis_data
    
    def load_glacier_mask(self, glacier_id: str) -> Optional[gpd.GeoDataFrame]:
        """Load glacier mask for a specific glacier."""
        try:
            if glacier_id not in self.glaciers:
                logger.warning(f"Glacier {glacier_id} not found in configuration")
                return None
                
            glacier_config = self.glaciers[glacier_id]
            mask_file = glacier_config['data_files']['mask']
            
            # Handle different mask file specifications
            if mask_file.startswith('data/'):
                mask_path = Path(mask_file)
            else:
                mask_path = Path('data') / 'glacier_masks' / glacier_id / mask_file
            
            if not mask_path.exists():
                logger.warning(f"Mask file not found for {glacier_id}: {mask_path}")
                return None
            
            # Load mask file (shapefile or GeoTIFF)
            if mask_path.suffix.lower() in ['.tif', '.tiff']:
                # Handle GeoTIFF raster mask
                import rasterio
                from rasterio.features import shapes
                
                with rasterio.open(str(mask_path)) as src:
                    # Read raster data
                    raster_data = src.read(1)
                    transform = src.transform
                    crs = src.crs
                    
                    # Create binary mask (assuming non-zero values are glacier)
                    binary_mask = raster_data > 0
                    
                    # Convert to polygons
                    polygons = []
                    for geom, value in shapes(binary_mask.astype(np.uint8), transform=transform):
                        if value == 1:
                            from shapely.geometry import shape
                            polygons.append(shape(geom))
                    
                    # Create GeoDataFrame
                    mask_gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
                    
                    # Dissolve to create single glacier outline
                    if len(mask_gdf) > 0:
                        mask_gdf = mask_gdf.dissolve().reset_index(drop=True)
                    
                    logger.info(f"Converted GeoTIFF mask to vector for {glacier_id}: {len(polygons)} features")
            else:
                # Load shapefile
                mask_gdf = gpd.read_file(str(mask_path))
            
            # Ensure WGS84 CRS
            if mask_gdf.crs is None:
                mask_gdf.crs = 'EPSG:4326'
            elif mask_gdf.crs.to_epsg() != 4326:
                mask_gdf = mask_gdf.to_crs('EPSG:4326')
            
            # Clean invalid geometries
            mask_gdf = mask_gdf[mask_gdf.geometry.is_valid]
            
            if not mask_gdf.empty:
                logger.info(f"Loaded mask for {glacier_id}: {len(mask_gdf)} features")
                return mask_gdf
            else:
                logger.warning(f"No valid geometries found in mask for {glacier_id}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading mask for {glacier_id}: {e}")
            return None
    
    def get_aws_coordinates(self, glacier_id: str) -> Dict[str, Dict[str, float]]:
        """Get AWS station coordinates for a glacier."""
        try:
            if glacier_id not in self.glaciers:
                return {}
                
            glacier_config = self.glaciers[glacier_id]
            
            # For Athabasca, try to load from Point_Custom.shp first
            if glacier_id == 'athabasca' and 'aws_points' in glacier_config['data_files']:
                try:
                    aws_points_file = glacier_config['data_files']['aws_points']
                    if aws_points_file.startswith('data/'):
                        points_path = Path(aws_points_file)
                    else:
                        points_path = Path('data') / 'glacier_masks' / glacier_id / aws_points_file
                    
                    if points_path.exists():
                        points_gdf = gpd.read_file(str(points_path))
                        if not points_gdf.empty:
                            # Use first point
                            point_geom = points_gdf.geometry.iloc[0]
                            return {
                                'athabasca_aws': {
                                    'lat': point_geom.y,
                                    'lon': point_geom.x,
                                    'elevation': 2200,
                                    'name': 'Athabasca AWS (from shapefile)'
                                }
                            }
                except Exception as e:
                    logger.warning(f"Could not load AWS points from shapefile for {glacier_id}: {e}")
            
            # Fallback to configuration coordinates
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
    
    def load_original_modis_data(self, glacier_id: str, analysis_mode: bool = False) -> Optional[pd.DataFrame]:
        """
        Load original MODIS data with coordinates.
        
        Args:
            glacier_id: Glacier identifier
            analysis_mode: If True, returns only best pixel for analysis (Haig/Coropuna)
                          If False, returns all pixels for mapping visualization
        
        Returns:
            DataFrame with MODIS pixel data
        """
        try:
            if glacier_id not in self.glaciers:
                logger.warning(f"Glacier {glacier_id} not found in configuration")
                return None
                
            glacier_config = self.glaciers[glacier_id]
            
            # Get MODIS data file path (use first method available)
            modis_files = glacier_config.get('data_files', {}).get('modis', {})
            if not modis_files:
                logger.warning(f"No MODIS data files configured for {glacier_id}")
                return None
            
            # Get first available MODIS file
            first_method = list(modis_files.keys())[0]
            modis_file = modis_files[first_method]
            
            # Handle different file path specifications
            if modis_file.startswith('data/'):
                modis_path = Path(modis_file)
            else:
                modis_path = Path('data') / 'modis' / modis_file
            
            if not modis_path.exists():
                logger.warning(f"MODIS data file not found for {glacier_id}: {modis_path}")
                return None
            
            # Load the original MODIS data with coordinates
            logger.info(f"Loading original MODIS data for {glacier_id} from {modis_path}")
            modis_data = pd.read_csv(modis_path)
            
            # Check for required columns
            required_cols = ['longitude', 'latitude', 'albedo']
            missing_cols = [col for col in required_cols if col not in modis_data.columns]
            
            if missing_cols:
                logger.warning(f"Missing columns in {glacier_id} MODIS data: {missing_cols}")
                return None
            
            # Filter for valid data points
            valid_data = modis_data.dropna(subset=['longitude', 'latitude', 'albedo'])
            
            # Group by pixel_id to get unique pixel locations (not every observation)
            if 'pixel_id' in valid_data.columns:
                # Take mean values for each unique pixel location
                unique_pixels = valid_data.groupby('pixel_id').agg({
                    'longitude': 'first',
                    'latitude': 'first', 
                    'albedo': 'mean',
                    'glacier_fraction': 'mean' if 'glacier_fraction' in valid_data.columns else 'first'
                }).reset_index()
                
                logger.info(f"Loaded {len(unique_pixels)} unique MODIS pixel locations for {glacier_id} (from {len(valid_data)} observations)")
                
                # Apply pixel selection for analysis mode
                if analysis_mode:
                    aws_coords = self.get_aws_coordinates(glacier_id)
                    selected_pixels = self.select_best_pixel_for_analysis(glacier_id, unique_pixels, aws_coords)
                    return selected_pixels
                else:
                    return unique_pixels
            else:
                logger.warning(f"No pixel_id column found for {glacier_id}, using all data points")
                logger.info(f"Loaded {len(valid_data)} MODIS data points for {glacier_id}")
                
                # Apply pixel selection for analysis mode
                if analysis_mode:
                    aws_coords = self.get_aws_coordinates(glacier_id)
                    selected_pixels = self.select_best_pixel_for_analysis(glacier_id, valid_data, aws_coords)
                    return selected_pixels
                else:
                    return valid_data
                
        except Exception as e:
            logger.error(f"Error loading original MODIS data for {glacier_id}: {e}")
            return None
    
    def create_individual_glacier_map(self, glacier_id: str, output_dir: Path) -> Optional[plt.Figure]:
        """Create detailed map for an individual glacier."""
        logger.info(f"Creating individual map for {glacier_id}")
        
        try:
            # Load data
            mask_gdf = self.load_glacier_mask(glacier_id)
            aws_coords = self.get_aws_coordinates(glacier_id)
            modis_data = self.load_original_modis_data(glacier_id)
            
            if mask_gdf is None:
                logger.warning(f"Cannot create map for {glacier_id} - no mask data")
                return None
            
            if modis_data is None or modis_data.empty:
                logger.warning(f"Cannot create map for {glacier_id} - no MODIS pixel data")
                return None
            
            # Create figure with larger size for detailed view
            fig, ax = plt.subplots(1, 1, figsize=(14, 10))
            
            # Set background color (light gray like the example)
            ax.set_facecolor('lightgray')
            
            # Plot glacier mask with blue outline like the example
            mask_gdf.plot(ax=ax, facecolor='lightblue', alpha=0.6,
                         edgecolor='blue', linewidth=2, 
                         label='Glacier Mask')
            
            # Plot ALL MODIS pixels with glacier fraction color map
            if 'glacier_fraction' in modis_data.columns:
                # Use glacier fraction for coloring (like the example)
                scatter = ax.scatter(modis_data['longitude'], modis_data['latitude'],
                                   c=modis_data['glacier_fraction'], cmap='viridis',
                                   s=35, alpha=0.8, 
                                   label=f'Pixel Locations (n={len(modis_data)})', 
                                   vmin=0, vmax=1, zorder=3)
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
                cbar.set_label('Glacier Fraction', rotation=270, labelpad=20, fontsize=12)
            else:
                # Fallback to albedo coloring
                scatter = ax.scatter(modis_data['longitude'], modis_data['latitude'],
                                   c=modis_data['albedo'], cmap='viridis',
                                   s=35, alpha=0.8, 
                                   label=f'Pixel Locations (n={len(modis_data)})', 
                                   vmin=0, vmax=1, zorder=3)
                # Add colorbar
                cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
                cbar.set_label('Albedo', rotation=270, labelpad=20, fontsize=12)
            
            # Plot AWS station with red star like the example
            if aws_coords:
                for station_id, coords in aws_coords.items():
                    if coords['lat'] is not None and coords['lon'] is not None:
                        ax.scatter(coords['lon'], coords['lat'], 
                                 c='red', s=200, marker='*',
                                 label='AWS Station', zorder=5,
                                 edgecolors='darkred', linewidth=2)
                        
                        # Add coordinates annotation in a box (like the example)
                        lon_dir = 'W' if coords['lon'] < 0 else 'E'
                        lat_dir = 'S' if coords['lat'] < 0 else 'N'
                        lon_val = abs(coords['lon'])
                        lat_val = abs(coords['lat'])
                        coord_text = f"AWS Station\n{lon_val:.0f}deg{lon_dir} {lat_val:.0f}deg{lat_dir}"
                        ax.annotate(coord_text, (coords['lon'], coords['lat']),
                                   xytext=(20, 20), textcoords='offset points',
                                   fontsize=10, ha='left', fontweight='bold',
                                   bbox=dict(boxstyle='round,pad=0.3', 
                                            facecolor='white', alpha=0.9, edgecolor='red'))
            
            # Set comprehensive title like the example
            glacier_name = self.glaciers[glacier_id]['name']
            pixel_count = len(modis_data)
            
            title = f"{glacier_name} Comprehensive Analysis Map\n"
            title += f"Pixel Locations, MODIS Data, Glacier Mask & AWS Station"
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Set axis labels
            ax.set_xlabel('Longitude (°)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Latitude (°)', fontsize=12, fontweight='bold')
            
            # Add grid with subtle styling
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Set equal aspect ratio for proper geographic representation
            ax.set_aspect('equal', adjustable='box')
            
            # Create comprehensive legend like the example
            legend_elements = []
            legend_elements.append(plt.scatter([], [], c='gray', s=20, alpha=0.8, label='Pixel Locations'))
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', 
                                               edgecolor='blue', alpha=0.6, label='Glacier Mask'))
            legend_elements.append(plt.scatter([], [], c='red', s=100, marker='*', 
                                             edgecolors='darkred', label='AWS Station'))
            
            ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
                     fancybox=True, shadow=True, fontsize=11)
            
            plt.tight_layout()
            
            # Save plot with high DPI for publication quality
            output_file = output_dir / "plots" / f"map_individual_{glacier_id}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved individual map for {glacier_id}: {output_file}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating individual map for {glacier_id}: {e}")
            return None
    
    def create_multi_glacier_overview_map(self, output_dir: Path) -> Optional[plt.Figure]:
        """Create overview map showing all glaciers."""
        logger.info("Creating multi-glacier overview map")
        
        try:
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
            fig, ax = plt.subplots(1, 1, figsize=(16, 10))
            
            # Set background color for ocean
            ax.set_facecolor('lightblue')
            
            # Plot each glacier
            legend_elements = []
            
            for glacier_id, mask_gdf in all_masks.items():
                glacier_name = self.glaciers[glacier_id]['name']
                color = self.colors.get(glacier_id, 'blue')
                
                # Plot glacier mask
                mask_gdf.plot(ax=ax, color=color, alpha=0.8,
                             edgecolor='black', linewidth=2)
                
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
                                     zorder=5, edgecolors='white', linewidth=1)
            
            # Add AWS legend element
            legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                             markerfacecolor='red', markersize=10,
                                             label='AWS stations'))
            
            # Set axis labels and title
            ax.set_xlabel('Longitude (°)', fontsize=12)
            ax.set_ylabel('Latitude (°)', fontsize=12)
            ax.set_title('Multi-Glacier Comparative Analysis Overview\nMODIS Albedo Validation Sites', 
                        fontsize=16, fontweight='bold', pad=20)
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Add legend
            ax.legend(handles=legend_elements, loc='upper left', 
                     bbox_to_anchor=(0.02, 0.98), frameon=True, fancybox=True, shadow=True)
            
            plt.tight_layout()
            
            # Save plot
            output_file = output_dir / "plots" / "map_multi_glacier_overview.png"
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Saved multi-glacier overview map: {output_file}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating overview map: {e}")
            return None
    
    def create_basic_correlation_map(self, glacier_id: str, method: str, output_dir: Path) -> Optional[plt.Figure]:
        """Create a basic spatial correlation map for one glacier and method."""
        logger.info(f"Creating correlation map for {glacier_id} {method}")
        
        try:
            # Load data
            mask_gdf = self.load_glacier_mask(glacier_id)
            aws_coords = self.get_aws_coordinates(glacier_id)
            modis_data = self.load_all_modis_data(glacier_id)
            
            if mask_gdf is None or modis_data is None:
                logger.warning(f"Cannot create correlation map for {glacier_id} {method} - missing data")
                return None
            
            # Check for required columns
            lat_col = 'lat' if 'lat' in modis_data.columns else 'latitude'
            lon_col = 'lon' if 'lon' in modis_data.columns else 'longitude'
            aws_col = 'AWS'
            
            if (lat_col not in modis_data.columns or lon_col not in modis_data.columns or 
                aws_col not in modis_data.columns or method not in modis_data.columns):
                logger.warning(f"Missing required columns for {glacier_id} {method}")
                return None
            
            # Filter valid data (both AWS and MODIS available)
            valid_data = modis_data.dropna(subset=[aws_col, method])
            
            if len(valid_data) < 10:
                logger.warning(f"Insufficient data for {glacier_id} {method} correlation map")
                return None
            
            # Calculate residuals (MODIS - AWS)
            valid_data = valid_data.copy()
            valid_data['residuals'] = valid_data[method] - valid_data[aws_col]
            
            # Create figure
            fig, ax = plt.subplots(1, 1, figsize=self.figsize_base)
            
            # Set background color
            ax.set_facecolor('lightgray')
            
            # Plot glacier mask
            mask_gdf.plot(ax=ax, facecolor='none', edgecolor='darkblue',
                         linewidth=2, alpha=0.8, label='Glacier extent')
            
            # Plot residuals with symmetric color scale
            abs_max = valid_data['residuals'].abs().quantile(0.95)
            norm = Normalize(vmin=-abs_max, vmax=abs_max)
            
            scatter = ax.scatter(valid_data[lon_col], valid_data[lat_col],
                               c=valid_data['residuals'], cmap='RdBu_r', norm=norm,
                               s=20, alpha=0.7, edgecolors='black', linewidth=0.1)
            
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
                                 edgecolors='white', linewidth=2)
            
            # Calculate statistics for title
            correlation = np.corrcoef(valid_data[aws_col], valid_data[method])[0, 1]
            rmse = np.sqrt(np.mean(valid_data['residuals']**2))
            bias = np.mean(valid_data['residuals'])
            
            # Set labels and title
            ax.set_xlabel('Longitude (°)', fontsize=12)
            ax.set_ylabel('Latitude (°)', fontsize=12)
            
            glacier_name = self.glaciers[glacier_id]['name']
            ax.set_title(f'{glacier_name} - {method}\nSpatial Residuals (MODIS - AWS)\n'
                        f'R = {correlation:.3f}, RMSE = {rmse:.3f}, Bias = {bias:.3f}',
                        fontsize=12, fontweight='bold')
            
            # Add grid and aspect ratio
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            
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
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation map for {glacier_id} {method}: {e}")
            return None
    
    def create_combined_glacier_maps(self, output_dir: Path) -> Optional[plt.Figure]:
        """Create combined layout showing all three glaciers with exact same design as individual maps."""
        logger.info("Creating combined 3-glacier layout map")
        
        try:
            # Create figure with 1x3 subplot layout 
            fig, axes = plt.subplots(1, 3, figsize=(42, 10))
            
            # Define glacier order for consistent layout
            glacier_order = ['athabasca', 'haig', 'coropuna']
            
            for idx, glacier_id in enumerate(glacier_order):
                if glacier_id not in self.glaciers:
                    continue
                    
                ax = axes[idx]
                logger.info(f"Creating subplot for {glacier_id}")
                
                # Load data (same as individual maps)
                mask_gdf = self.load_glacier_mask(glacier_id)
                aws_coords = self.get_aws_coordinates(glacier_id)
                modis_data = self.load_original_modis_data(glacier_id)
                
                if mask_gdf is None or modis_data is None or modis_data.empty:
                    logger.warning(f"Cannot create subplot for {glacier_id} - missing data")
                    ax.text(0.5, 0.5, f"No data available\nfor {glacier_id}", 
                           ha='center', va='center', transform=ax.transAxes)
                    continue
                
                # Set background color (exact same as individual maps)
                ax.set_facecolor('lightgray')
                
                # Plot glacier mask with blue outline (exact same styling)
                mask_gdf.plot(ax=ax, facecolor='lightblue', alpha=0.6,
                             edgecolor='blue', linewidth=2, 
                             label='Glacier Mask')
                
                # Plot MODIS pixels with exact same styling
                if 'glacier_fraction' in modis_data.columns:
                    # Use glacier fraction for coloring
                    scatter = ax.scatter(modis_data['longitude'], modis_data['latitude'],
                                       c=modis_data['glacier_fraction'], cmap='viridis',
                                       s=35, alpha=0.8, 
                                       label=f'Pixel Locations (n={len(modis_data)})', 
                                       vmin=0, vmax=1, zorder=3)
                    # Add colorbar for each subplot
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
                    cbar.set_label('Glacier Fraction', rotation=270, labelpad=20, fontsize=10)
                else:
                    # Fallback to albedo coloring
                    scatter = ax.scatter(modis_data['longitude'], modis_data['latitude'],
                                       c=modis_data['albedo'], cmap='viridis',
                                       s=35, alpha=0.8, 
                                       label=f'Pixel Locations (n={len(modis_data)})', 
                                       vmin=0, vmax=1, zorder=3)
                    # Add colorbar for each subplot
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
                    cbar.set_label('Albedo', rotation=270, labelpad=20, fontsize=10)
                
                # Plot AWS station with exact same styling
                if aws_coords:
                    for station_id, coords in aws_coords.items():
                        if coords['lat'] is not None and coords['lon'] is not None:
                            ax.scatter(coords['lon'], coords['lat'], 
                                     c='red', s=200, marker='*',
                                     label='AWS Station', zorder=5,
                                     edgecolors='darkred', linewidth=2)
                            
                            # Add coordinates annotation in a box (exact same styling)
                            lon_dir = 'W' if coords['lon'] < 0 else 'E'
                            lat_dir = 'S' if coords['lat'] < 0 else 'N'
                            lon_val = abs(coords['lon'])
                            lat_val = abs(coords['lat'])
                            coord_text = f"AWS Station\\n{lon_val:.0f}deg{lon_dir} {lat_val:.0f}deg{lat_dir}"
                            ax.annotate(coord_text, (coords['lon'], coords['lat']),
                                       xytext=(20, 20), textcoords='offset points',
                                       fontsize=8, ha='left', fontweight='bold',
                                       bbox=dict(boxstyle='round,pad=0.3', 
                                                facecolor='white', alpha=0.9, edgecolor='red'))
                
                # Set title for each subplot (exact same format)
                glacier_name = self.glaciers[glacier_id]['name']
                pixel_count = len(modis_data)
                
                title = f"{glacier_name} Analysis Map\\n"
                title += f"Pixel Locations, MODIS Data, Glacier Mask & AWS Station"
                
                ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
                
                # Set axis labels (exact same styling)
                ax.set_xlabel('Longitude (°)', fontsize=10, fontweight='bold')
                ax.set_ylabel('Latitude (°)', fontsize=10, fontweight='bold')
                
                # Add grid with exact same styling
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                
                # Set equal aspect ratio for proper geographic representation
                ax.set_aspect('equal', adjustable='box')
                
                # Create legend for each subplot (exact same as individual)
                legend_elements = []
                legend_elements.append(plt.scatter([], [], c='gray', s=20, alpha=0.8, label='Pixel Locations'))
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='lightblue', 
                                                   edgecolor='blue', alpha=0.6, label='Glacier Mask'))
                legend_elements.append(plt.scatter([], [], c='red', s=100, marker='*', 
                                                 edgecolors='darkred', label='AWS Station'))
                
                ax.legend(handles=legend_elements, loc='upper right', frameon=True, 
                         fancybox=True, shadow=True, fontsize=9)
            
            # Add overall title for the combined figure
            fig.suptitle('Multi-Glacier Comprehensive Analysis Maps\\n' +
                        'Pixel Locations, MODIS Data, Glacier Masks & AWS Stations', 
                        fontsize=16, fontweight='bold', y=0.95)
            
            plt.tight_layout()
            plt.subplots_adjust(top=0.88)  # Make room for main title
            
            # Save plot with high DPI for publication quality
            output_file = output_dir / "plots" / "map_combined_all_glaciers.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
            logger.info(f"Saved combined glacier map: {output_file}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating combined glacier map: {e}")
            return None
    
    def generate_all_maps(self, output_dir: Path) -> None:
        """Generate individual glacier maps and combined layout map."""
        logger.info("Starting generation of all glacier maps...")
        
        try:
            # Ensure plots directory exists
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            map_count = 0
            
            # Create individual glacier maps
            for glacier_id in self.glaciers.keys():
                logger.info(f"Creating individual map for {glacier_id}...")
                fig = self.create_individual_glacier_map(glacier_id, output_dir)
                if fig:
                    plt.close(fig)  # Free memory
                    map_count += 1
                    logger.info(f"[OK] Created individual map for {glacier_id}")
                else:
                    logger.warning(f"[FAIL] Failed to create individual map for {glacier_id}")
            
            # Create combined layout map
            logger.info("Creating combined 3-glacier layout map...")
            combined_fig = self.create_combined_glacier_maps(output_dir)
            if combined_fig:
                plt.close(combined_fig)  # Free memory
                map_count += 1
                logger.info("[OK] Created combined glacier layout map")
            else:
                logger.warning("[FAIL] Failed to create combined glacier layout map")
            
            logger.info(f"Successfully generated {map_count} total maps ({map_count-1} individual + 1 combined)")
            
        except Exception as e:
            logger.error(f"Error generating maps: {e}")
            raise


def main():
    """Test the simplified multi-glacier mapping suite."""
    mapper = MultiGlacierMapperSimple()
    
    # Test loading masks
    for glacier_id in mapper.glaciers.keys():
        mask = mapper.load_glacier_mask(glacier_id)
        print(f"{glacier_id}: {'✓' if mask is not None else '✗'} mask loaded")
        
        aws_coords = mapper.get_aws_coordinates(glacier_id)
        print(f"{glacier_id}: {len(aws_coords)} AWS stations")


if __name__ == "__main__":
    main()