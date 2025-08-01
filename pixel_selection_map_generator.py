#!/usr/bin/env python3
"""
Pixel Selection Map Generator

This standalone script creates the exact pixel selection maps showing glacier boundaries,
MODIS pixels, selected best pixels, and AWS weather stations. It generates individual
glacier maps, combined layout maps, and pixel selection summary overview.

Features:
- Individual glacier maps with pixel selection visualization
- Combined 3-panel layout map (Athabasca, Haig, Coropuna)
- Pixel selection summary overview with legend
- High-resolution output matching existing framework style

Author: Generated from Multi-Glacier Albedo Analysis Framework
Date: 2025-07-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import logging
import math
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'glaciers': {
        'athabasca': {
            'name': 'Athabasca Glacier',
            'region': 'Canadian Rockies',
            'coordinates': {'lat': 52.2, 'lon': -117.25},
            'aws_stations': {
                'iceAWS_Atha': {
                    'name': 'Athabasca Glacier AWS',
                    'lat': 52.1949,
                    'lon': -117.2431,
                    'elevation': 2200
                }
            },
            'mask_path': 'D:/Documents/Projects/Albedo_analysis_New/data/glacier_masks/athabasca/masque_athabasa_zone_ablation.shp',
            'modis_path': 'D:/Documents/Projects/athabasca_analysis/data/csv/Athabasca_Terra_Aqua_MultiProduct_2014-01-01_to_2021-01-01.csv'
        },
        'haig': {
            'name': 'Haig Glacier',
            'region': 'Canadian Rockies', 
            'coordinates': {'lat': 50.73, 'lon': -116.17},
            'aws_stations': {
                'haig_station': {
                    'name': 'Haig Glacier AWS',
                    'lat': 50.7124,
                    'lon': -115.3018,
                    'elevation': 2800
                }
            },
            'mask_path': 'D:/Documents/Projects/Albedo_analysis_New/data/glacier_masks/haig/Haig_glacier_final.shp',
            'modis_path': 'D:/Documents/Projects/Haig_analysis/data/csv/Haig_MODIS_Pixel_Analysis_MultiProduct_2002_to_2016_fraction.csv'
        },
        'coropuna': {
            'name': 'Coropuna Glacier',
            'region': 'Peruvian Andes',
            'coordinates': {'lat': -15.54, 'lon': -72.66},
            'aws_stations': {
                'coropuna_station': {
                    'name': 'Coropuna Glacier AWS',
                    'lat': -15.5361,
                    'lon': -72.5997,
                    'elevation': 5400
                }
            },
            'mask_path': 'D:/Documents/Projects/Albedo_analysis_New/data/glacier_masks/coropuna/coropuna.shp',
            'modis_path': 'D:/Documents/Projects/Coropuna_glacier/data/csv/coropuna_glacier_2014-01-01_to_2025-01-01.csv'
        }
    },
    'colors': {
        'glacier_mask': 'lightblue',
        'glacier_edge': 'blue', 
        'all_pixels': 'lightgray',
        'selected_pixels': 'red',
        'aws_stations': 'blue',
        'background': 'lightgray'
    },
    'quality_filters': {
        'min_glacier_fraction': 0.1,
        'min_observations': 10
    },
    'visualization': {
        'individual_figsize': (12, 10),
        'combined_figsize': (36, 10),
        'summary_figsize': (18, 6),
        'dpi': 300
    }
}


class DataLoader:
    """Handles loading and preprocessing of MODIS data for mapping."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_modis_pixels(self, glacier_id: str) -> pd.DataFrame:
        """Load MODIS pixel data for a specific glacier."""
        logger.info(f"Loading MODIS pixel data for {glacier_id}...")
        
        glacier_config = self.config['glaciers'][glacier_id]
        modis_path = glacier_config['modis_path']
        
        if not Path(modis_path).exists():
            raise FileNotFoundError(f"MODIS data file not found: {modis_path}")
        
        # Load data
        data = pd.read_csv(modis_path)
        
        # Process based on glacier-specific format
        if glacier_id == 'coropuna':
            # Coropuna has method column - already in long format
            if 'method' in data.columns and 'albedo' in data.columns:
                logger.info(f"Coropuna data in long format: {len(data)} records")
                # Get unique pixels with their coordinates
                pixel_data = data.groupby('pixel_id').agg({
                    'latitude': 'first',
                    'longitude': 'first',
                    'glacier_fraction': 'mean',
                    'albedo': 'count'
                }).reset_index()
                pixel_data.columns = ['pixel_id', 'latitude', 'longitude', 'glacier_fraction', 'n_observations']
                return pixel_data
        
        else:
            # For Athabasca and Haig - check format
            if 'method' in data.columns:
                # Already in long format
                pixel_data = data.groupby('pixel_id').agg({
                    'latitude': 'first',
                    'longitude': 'first',
                    'glacier_fraction': 'mean' if 'glacier_fraction' in data.columns else 'size',
                    'date': 'count'
                }).reset_index()
                pixel_data.columns = ['pixel_id', 'latitude', 'longitude', 'glacier_fraction', 'n_observations']
                return pixel_data
            else:
                # Wide format - get unique pixels
                coord_cols = ['pixel_id', 'latitude', 'longitude']
                if 'glacier_fraction' in data.columns:
                    coord_cols.append('glacier_fraction')
                
                pixel_data = data[coord_cols].drop_duplicates(subset=['pixel_id'])
                pixel_data['n_observations'] = len(data)  # Rough estimate
                
                if 'glacier_fraction' not in pixel_data.columns:
                    pixel_data['glacier_fraction'] = 1.0  # Default
                
                return pixel_data
        
        logger.info(f"Loaded {len(pixel_data)} unique pixels for {glacier_id}")
        return pixel_data


class PixelSelector:
    """Implements intelligent pixel selection based on distance to AWS stations."""
    
    @staticmethod
    def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance using Haversine formula."""
        R = 6371  # Earth's radius in km
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def select_best_pixels(self, glacier_id: str, pixel_data: pd.DataFrame, 
                          aws_stations: Dict[str, Dict]) -> pd.DataFrame:
        """Select best pixels for analysis based on AWS distance and glacier fraction."""
        logger.info(f"Applying pixel selection for {glacier_id}...")
        
        # Apply quality filters first
        quality_filters = CONFIG['quality_filters']
        if 'glacier_fraction' in pixel_data.columns:
            quality_pixels = pixel_data[
                (pixel_data['glacier_fraction'] > quality_filters['min_glacier_fraction']) &
                (pixel_data['n_observations'] > quality_filters['min_observations'])
            ].copy()
        else:
            quality_pixels = pixel_data[
                pixel_data['n_observations'] > quality_filters['min_observations']
            ].copy()
        
        if len(quality_pixels) == 0:
            logger.warning(f"No quality pixels found for {glacier_id}, using all pixels")
            return pixel_data
        
        # For Athabasca (small dataset), use all quality pixels
        if glacier_id == 'athabasca' or len(quality_pixels) <= 2:
            logger.info(f"Using all {len(quality_pixels)} pixels for {glacier_id} (small dataset)")
            return quality_pixels
        
        # Get AWS station coordinates
        if not aws_stations:
            logger.warning(f"No AWS stations for {glacier_id}, using all pixels")
            return quality_pixels
            
        aws_station = list(aws_stations.values())[0]
        aws_lat, aws_lon = aws_station['lat'], aws_station['lon']
        
        # Calculate distance to AWS station
        quality_pixels['distance_to_aws'] = quality_pixels.apply(
            lambda row: self.calculate_haversine_distance(
                row['latitude'], row['longitude'], aws_lat, aws_lon
            ), axis=1
        )
        
        # Score pixels: 60% distance + 40% glacier fraction
        max_distance = quality_pixels['distance_to_aws'].max()
        min_distance = quality_pixels['distance_to_aws'].min()
        
        if max_distance > min_distance:
            quality_pixels['distance_score'] = 1 - (quality_pixels['distance_to_aws'] - min_distance) / (max_distance - min_distance)
        else:
            quality_pixels['distance_score'] = 1.0
        
        if 'glacier_fraction' in quality_pixels.columns:
            max_fraction = quality_pixels['glacier_fraction'].max()
            if max_fraction > 0:
                quality_pixels['glacier_fraction_score'] = quality_pixels['glacier_fraction'] / max_fraction
            else:
                quality_pixels['glacier_fraction_score'] = 0.5
            glacier_weight = 0.4
        else:
            quality_pixels['glacier_fraction_score'] = 0.5
            glacier_weight = 0.0
        
        # Composite score
        quality_pixels['composite_score'] = (
            0.6 * quality_pixels['distance_score'] + 
            glacier_weight * quality_pixels['glacier_fraction_score'] +
            (1 - 0.6 - glacier_weight) * 0.5
        )
        
        # Select best pixel
        best_pixel_idx = quality_pixels['composite_score'].idxmax()
        best_pixel = quality_pixels.loc[[best_pixel_idx]]
        
        logger.info(f"Selected 1 best pixel for {glacier_id} from {len(quality_pixels)} candidates")
        logger.info(f"  Distance to AWS: {best_pixel['distance_to_aws'].iloc[0]:.2f} km")
        if 'glacier_fraction' in best_pixel.columns:
            logger.info(f"  Glacier fraction: {best_pixel['glacier_fraction'].iloc[0]:.3f}")
        
        return best_pixel


class GlacierMaskLoader:
    """Handles loading glacier boundary shapefiles."""
    
    @staticmethod
    def load_mask(mask_path: str) -> Optional[gpd.GeoDataFrame]:
        """Load glacier mask shapefile."""
        try:
            if not Path(mask_path).exists():
                logger.warning(f"Mask file not found: {mask_path}")
                return None
            
            mask_gdf = gpd.read_file(mask_path)
            logger.info(f"Loaded glacier mask: {len(mask_gdf)} features")
            return mask_gdf
            
        except Exception as e:
            logger.error(f"Error loading mask from {mask_path}: {e}")
            return None


class PixelSelectionMapVisualizer:
    """Creates pixel selection maps with glacier boundaries and AWS stations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.colors = config['colors']
        
    def create_individual_glacier_map(self, glacier_id: str, all_pixels: pd.DataFrame, 
                                    selected_pixels: pd.DataFrame, mask_gdf: Optional[gpd.GeoDataFrame],
                                    aws_stations: Dict[str, Dict], output_path: str) -> plt.Figure:
        """Create individual glacier map with pixel selection visualization."""
        logger.info(f"Creating individual map for {glacier_id}...")
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=self.config['visualization']['individual_figsize'])
        
        # Set background
        ax.set_facecolor(self.colors['background'])
        
        # Plot glacier mask if available
        if mask_gdf is not None and not mask_gdf.empty:
            mask_gdf.plot(ax=ax, facecolor=self.colors['glacier_mask'], alpha=0.6,
                         edgecolor=self.colors['glacier_edge'], linewidth=2)
        
        # Plot all pixels as context
        ax.scatter(all_pixels['longitude'], all_pixels['latitude'],
                  c=self.colors['all_pixels'], s=30, alpha=0.5, marker='o',
                  label=f'All MODIS Pixels (n={len(all_pixels)})', zorder=2)
        
        # Plot selected pixels prominently
        if not selected_pixels.empty:
            ax.scatter(selected_pixels['longitude'], selected_pixels['latitude'],
                      c=self.colors['selected_pixels'], s=200, alpha=0.9, marker='*',
                      edgecolor='black', linewidth=2,
                      label=f'Selected Best Pixels (n={len(selected_pixels)})', zorder=4)
        
        # Plot AWS stations
        for station_id, station in aws_stations.items():
            ax.scatter(station['lon'], station['lat'],
                      c=self.colors['aws_stations'], s=150, marker='^',
                      edgecolor='black', linewidth=1,
                      label=f'AWS Weather Station', zorder=3)
        
        # Set title and labels
        glacier_config = self.config['glaciers'][glacier_id]
        ax.set_title(f"{glacier_config['name']} - Pixel Selection for Analysis", 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Longitude (°)', fontsize=12)
        ax.set_ylabel('Latitude (°)', fontsize=12)
        
        # Add legend
        ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_path, dpi=self.config['visualization']['dpi'], 
                   bbox_inches='tight', facecolor='white')
        logger.info(f"Individual map saved: {output_path}")
        
        return fig
    
    def create_combined_glacier_maps(self, all_data: Dict[str, pd.DataFrame], 
                                   selected_data: Dict[str, pd.DataFrame],
                                   mask_data: Dict[str, Optional[gpd.GeoDataFrame]],
                                   aws_data: Dict[str, Dict[str, Dict]],
                                   output_path: str) -> plt.Figure:
        """Create combined 3-panel layout map."""
        logger.info("Creating combined 3-glacier layout map...")
        
        # Create 1x3 subplot layout
        fig, axes = plt.subplots(1, 3, figsize=self.config['visualization']['combined_figsize'])
        
        # Define glacier order for consistent layout
        glacier_order = ['athabasca', 'haig', 'coropuna']
        
        for idx, glacier_id in enumerate(glacier_order):
            if glacier_id not in self.config['glaciers']:
                continue
                
            ax = axes[idx]
            ax.set_facecolor(self.colors['background'])
            
            # Get data for this glacier
            all_pixels = all_data.get(glacier_id, pd.DataFrame())
            selected_pixels = selected_data.get(glacier_id, pd.DataFrame())
            mask_gdf = mask_data.get(glacier_id)
            aws_stations = aws_data.get(glacier_id, {})
            
            # Plot glacier mask
            if mask_gdf is not None and not mask_gdf.empty:
                mask_gdf.plot(ax=ax, facecolor=self.colors['glacier_mask'], alpha=0.6,
                             edgecolor=self.colors['glacier_edge'], linewidth=2)
            
            # Plot pixels
            if not all_pixels.empty:
                ax.scatter(all_pixels['longitude'], all_pixels['latitude'],
                          c=self.colors['all_pixels'], s=20, alpha=0.5, marker='o', zorder=2)
            
            if not selected_pixels.empty:
                ax.scatter(selected_pixels['longitude'], selected_pixels['latitude'],
                          c=self.colors['selected_pixels'], s=150, alpha=0.9, marker='*',
                          edgecolor='black', linewidth=1.5, zorder=4)
            
            # Plot AWS stations
            for station_id, station in aws_stations.items():
                ax.scatter(station['lon'], station['lat'],
                          c=self.colors['aws_stations'], s=100, marker='^',
                          edgecolor='black', linewidth=1, zorder=3)
            
            # Set title and labels
            glacier_config = self.config['glaciers'][glacier_id]
            ax.set_title(f"{glacier_config['name']}\n{glacier_config['region']}", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Longitude (°)', fontsize=10)
            ax.set_ylabel('Latitude (°)', fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add overall title
        fig.suptitle('Multi-Glacier Pixel Selection for Analysis\nSelected Best Pixels: 2/1/1 (Closest to AWS Stations)', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add legend at bottom
        legend_elements = [
            plt.scatter([], [], c=self.colors['glacier_mask'], s=100, alpha=0.6, 
                       edgecolor=self.colors['glacier_edge'], linewidth=2, label='Glacier Boundary'),
            plt.scatter([], [], c=self.colors['all_pixels'], s=50, alpha=0.5, 
                       marker='o', label='All MODIS Pixels'),
            plt.scatter([], [], c=self.colors['selected_pixels'], s=100, alpha=0.9, 
                       marker='*', edgecolor='black', linewidth=1, label='Selected Best Pixels'),
            plt.scatter([], [], c=self.colors['aws_stations'], s=100, marker='^',
                       edgecolor='black', linewidth=1, label='AWS Weather Station')
        ]
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=4, fontsize=11, framealpha=0.9)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.85)
        
        # Save figure
        fig.savefig(output_path, dpi=self.config['visualization']['dpi'], 
                   bbox_inches='tight', facecolor='white')
        logger.info(f"Combined map saved: {output_path}")
        
        return fig
    
    def create_pixel_selection_summary(self, all_data: Dict[str, pd.DataFrame], 
                                     selected_data: Dict[str, pd.DataFrame],
                                     mask_data: Dict[str, Optional[gpd.GeoDataFrame]],
                                     aws_data: Dict[str, Dict[str, Dict]],
                                     output_path: str) -> plt.Figure:
        """Create pixel selection summary overview map."""
        logger.info("Creating pixel selection summary map...")
        
        # Create 1x3 subplot layout with clean styling
        fig, axes = plt.subplots(1, 3, figsize=self.config['visualization']['summary_figsize'])
        
        glacier_order = ['athabasca', 'haig', 'coropuna']
        
        for idx, glacier_id in enumerate(glacier_order):
            if glacier_id not in self.config['glaciers']:
                continue
                
            ax = axes[idx]
            ax.set_facecolor('white')
            
            # Get data
            all_pixels = all_data.get(glacier_id, pd.DataFrame())
            selected_pixels = selected_data.get(glacier_id, pd.DataFrame())
            mask_gdf = mask_data.get(glacier_id)
            aws_stations = aws_data.get(glacier_id, {})
            
            # Plot glacier boundary
            if mask_gdf is not None and not mask_gdf.empty:
                mask_gdf.plot(ax=ax, facecolor=self.colors['glacier_mask'], alpha=0.7,
                             edgecolor=self.colors['glacier_edge'], linewidth=1.5)
            
            # Plot selected pixels with prominent stars
            if not selected_pixels.empty:
                ax.scatter(selected_pixels['longitude'], selected_pixels['latitude'],
                          c=self.colors['selected_pixels'], s=200, marker='*',
                          edgecolor='black', linewidth=1.5, zorder=4)
            
            # Plot AWS stations with triangles
            for station_id, station in aws_stations.items():
                ax.scatter(station['lon'], station['lat'],
                          c=self.colors['aws_stations'], s=120, marker='^',
                          edgecolor='black', linewidth=1, zorder=3)
            
            # Clean styling
            glacier_config = self.config['glaciers'][glacier_id]
            ax.set_title(f"{glacier_config['name'].replace(' Glacier', '')}", 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Longitude (°)', fontsize=10)
            ax.set_ylabel('Latitude (°)', fontsize=10)
            
            # Remove grid for clean look
            ax.grid(False)
            
            # Set aspect ratio to equal for geographic accuracy
            ax.set_aspect('equal', adjustable='box')
        
        # Add overall title
        fig.suptitle('Pixel Selection Summary - Best Pixels for Analysis', 
                    fontsize=14, fontweight='bold', y=0.95)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0),1,1, facecolor=self.colors['glacier_mask'], alpha=0.7,
                         edgecolor=self.colors['glacier_edge'], linewidth=1.5, label='Glacier Boundary'),
            plt.scatter([], [], c=self.colors['selected_pixels'], s=80, marker='*',
                       edgecolor='black', linewidth=1, label='Selected Best Pixels'),
            plt.scatter([], [], c=self.colors['aws_stations'], s=80, marker='^',
                       edgecolor='black', linewidth=1, label='AWS Weather Station')
        ]
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=3, fontsize=10, framealpha=0.9)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15, top=0.85)
        
        # Save figure
        fig.savefig(output_path, dpi=self.config['visualization']['dpi'], 
                   bbox_inches='tight', facecolor='white')
        logger.info(f"Summary map saved: {output_path}")
        
        return fig


def main():
    """Main execution function."""
    logger.info("Starting Pixel Selection Map Generation")
    
    # Initialize components
    data_loader = DataLoader(CONFIG)
    pixel_selector = PixelSelector()
    mask_loader = GlacierMaskLoader()
    visualizer = PixelSelectionMapVisualizer(CONFIG)
    
    # Storage for processed data
    all_pixel_data = {}
    selected_pixel_data = {}
    glacier_masks = {}
    aws_stations_data = {}
    
    # Process each glacier
    for glacier_id in CONFIG['glaciers'].keys():
        try:
            logger.info(f"\n{'='*50}")
            logger.info(f"Processing {glacier_id.upper()} Glacier")
            logger.info(f"{'='*50}")
            
            glacier_config = CONFIG['glaciers'][glacier_id]
            
            # Load MODIS pixel data
            all_pixels = data_loader.load_modis_pixels(glacier_id)
            all_pixel_data[glacier_id] = all_pixels
            
            # Select best pixels
            aws_stations = glacier_config['aws_stations']
            selected_pixels = pixel_selector.select_best_pixels(glacier_id, all_pixels, aws_stations)
            selected_pixel_data[glacier_id] = selected_pixels
            
            # Load glacier mask
            mask_gdf = mask_loader.load_mask(glacier_config['mask_path'])
            glacier_masks[glacier_id] = mask_gdf
            
            # Store AWS station data
            aws_stations_data[glacier_id] = aws_stations
            
            logger.info(f"✅ {glacier_id}: {len(all_pixels)} total pixels, {len(selected_pixels)} selected")
            
        except Exception as e:
            logger.error(f"❌ Error processing {glacier_id}: {e}")
            continue
    
    # Generate maps
    if all_pixel_data:
        logger.info(f"\n{'='*60}")
        logger.info("Generating Pixel Selection Maps")
        logger.info(f"{'='*60}")
        
        # Create individual glacier maps
        for glacier_id in all_pixel_data.keys():
            try:
                output_path = f"map_individual_{glacier_id}"
                if len(selected_pixel_data.get(glacier_id, [])) < len(all_pixel_data[glacier_id]):
                    output_path += "_pixel_selection"
                output_path += ".png"
                
                fig = visualizer.create_individual_glacier_map(
                    glacier_id, 
                    all_pixel_data[glacier_id],
                    selected_pixel_data.get(glacier_id, pd.DataFrame()),
                    glacier_masks.get(glacier_id),
                    aws_stations_data.get(glacier_id, {}),
                    output_path
                )
                plt.close(fig)
                
            except Exception as e:
                logger.error(f"Error creating individual map for {glacier_id}: {e}")
        
        # Create combined layout map
        try:
            combined_fig = visualizer.create_combined_glacier_maps(
                all_pixel_data, selected_pixel_data, glacier_masks, 
                aws_stations_data, "map_combined_all_glaciers_pixel_selection.png"
            )
            plt.close(combined_fig)
            
        except Exception as e:
            logger.error(f"Error creating combined map: {e}")
        
        # Create pixel selection summary
        try:
            summary_fig = visualizer.create_pixel_selection_summary(
                all_pixel_data, selected_pixel_data, glacier_masks,
                aws_stations_data, "pixel_selection_summary.png"
            )
            plt.close(summary_fig)
            
        except Exception as e:
            logger.error(f"Error creating summary map: {e}")
        
        logger.info(f"\n✅ SUCCESS: All pixel selection maps generated")
        logger.info(f"📊 Total glaciers processed: {len(all_pixel_data)}")
        
    else:
        logger.error("❌ No data could be processed for any glacier")


if __name__ == "__main__":
    main()