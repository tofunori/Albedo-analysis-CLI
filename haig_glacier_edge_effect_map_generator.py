#!/usr/bin/env python3
"""
Haig Glacier Edge Effect Map Generator

Creates a spatial visualization showing edge effects in MOD09GA data across the Haig glacier.
Pixels are colored by edge effect values and sized by glacier fraction, with glacier boundary 
and AWS station overlays.

Features:
- Edge effect spatial patterns visualization
- Glacier fraction integration (pixel sizing)
- Haig glacier boundary from shapefile
- AWS weather station location
- Statistical summary of edge effects

Author: Generated from Multi-Glacier Albedo Analysis Framework
Date: 2025-08-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'data_paths': {
        'edge_effect_data': "D:/Downloads/MODIS_Terra_Aqua_MultiProduct_2023-01-01_to_2025-01-01_fraction_10 - MODIS_Terra_Aqua_MultiProduct_2023-01-01_to_2025-01-01_fraction_10.csv",
        'glacier_mask': "D:/Documents/Projects/Albedo_analysis_New/data/glacier_masks/haig/Haig_glacier_final.shp"
    },
    'aws_station': {
        'name': 'Haig Glacier AWS',
        'lat': 50.7124,
        'lon': -115.3018,
        'elevation': 2800
    },
    'edge_effect_colors': {
        # High contrast gradient: Red to Blue for better differentiation
        1: '#d73027',  # Bright red (worst - edge pixel)
        2: '#fc8d59',  # Red-orange  
        3: '#fee08b',  # Yellow-orange
        4: '#ffffcc',  # Light yellow
        5: '#c7e9b4',  # Light green
        6: '#7fcdbb',  # Teal-green
        7: '#41b6c4',  # Cyan-blue
        8: '#2c7fb8',  # Blue
        9: '#253494',  # Dark blue (best - interior pixel)
        10: '#253494'  # Dark blue (fallback)
    },
    'glacier_colors': {
        'boundary': '#0066cc',
        'fill': 'lightblue',
        'aws_station': '#ff0000'
    },
    'visualization': {
        'figsize': (16, 12),
        'dpi': 300,
        'min_pixel_size': 20,
        'max_pixel_size': 200
    }
}


class EdgeEffectDataLoader:
    """Handles loading and processing of MOD09GA edge effect data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_edge_effect_data(self) -> pd.DataFrame:
        """Load and process MOD09GA edge effect data."""
        logger.info("Loading MOD09GA edge effect data...")
        
        file_path = self.config['data_paths']['edge_effect_data']
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Edge effect data file not found: {file_path}")
        
        # Load data
        data = pd.read_csv(file_path)
        logger.info(f"Loaded {len(data):,} records from edge effect dataset")
        
        # Convert date to datetime
        data['date'] = pd.to_datetime(data['date'])
        
        # Basic data info
        logger.info(f"Date range: {data['date'].min()} to {data['date'].max()}")
        logger.info(f"Unique pixels: {data['pixel_id'].nunique():,}")
        logger.info(f"Edge effect values: {sorted(data['edge_effect'].unique())}")
        
        return data
    
    def aggregate_pixel_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Aggregate data by pixel_id to get unique pixels for mapping."""
        logger.info("Aggregating edge effect data by pixel...")
        
        # Aggregate by pixel_id
        pixel_summary = data.groupby('pixel_id').agg({
            'longitude': 'first',  # Coordinates should be consistent
            'latitude': 'first',
            'glacier_fraction': 'mean',  # Average glacier fraction over time
            'edge_effect': lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0],  # Most common edge effect
            'albedo': 'mean',  # Average albedo for reference
            'date': 'count'  # Number of observations
        }).reset_index()
        
        # Rename columns for clarity
        pixel_summary.columns = ['pixel_id', 'longitude', 'latitude', 'glacier_fraction', 
                                'edge_effect', 'mean_albedo', 'n_observations']
        
        logger.info(f"Aggregated to {len(pixel_summary):,} unique pixels")
        logger.info(f"Glacier fraction range: {pixel_summary['glacier_fraction'].min():.3f} to {pixel_summary['glacier_fraction'].max():.3f}")
        logger.info(f"Edge effect distribution:")
        for edge_val, count in pixel_summary['edge_effect'].value_counts().sort_index().items():
            logger.info(f"  Edge effect {edge_val}: {count} pixels ({count/len(pixel_summary)*100:.1f}%)")
        
        return pixel_summary


class GlacierMaskLoader:
    """Handles loading glacier boundary shapefiles."""
    
    @staticmethod
    def load_haig_glacier_mask(mask_path: str) -> Optional[gpd.GeoDataFrame]:
        """Load Haig glacier mask shapefile."""
        try:
            if not Path(mask_path).exists():
                logger.warning(f"Glacier mask file not found: {mask_path}")
                return None
            
            mask_gdf = gpd.read_file(mask_path)
            logger.info(f"Loaded Haig glacier mask: {len(mask_gdf)} features")
            return mask_gdf
            
        except Exception as e:
            logger.error(f"Error loading glacier mask from {mask_path}: {e}")
            return None


class EdgeEffectMapVisualizer:
    """Creates edge effect spatial visualization maps."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.edge_colors = config['edge_effect_colors']
        self.glacier_colors = config['glacier_colors']
        
    def create_edge_effect_map(self, pixel_data: pd.DataFrame, glacier_mask: Optional[gpd.GeoDataFrame],
                             output_path: str) -> plt.Figure:
        """Create the main edge effect map visualization."""
        logger.info("Creating Haig glacier edge effect map...")
        
        # Create figure with subplots: map + scatterplot
        fig = plt.figure(figsize=(20, 12))
        ax_map = plt.subplot(1, 2, 1)  # Map on left
        ax_scatter = plt.subplot(1, 2, 2)  # Scatterplot on right
        
        # Set background for map
        ax_map.set_facecolor('white')
        
        # Plot glacier mask if available
        if glacier_mask is not None and not glacier_mask.empty:
            glacier_mask.plot(ax=ax_map, facecolor=self.glacier_colors['fill'], alpha=0.3,
                            edgecolor=self.glacier_colors['boundary'], linewidth=2,
                            label='Glacier Boundary')
        
        # Get edge effect values and create colormap
        edge_values = sorted(pixel_data['edge_effect'].unique())
        colors = [self.edge_colors.get(val, '#888888') for val in edge_values]
        
        # Create scatter plot colored by edge effect, sized by glacier fraction
        for i, edge_val in enumerate(edge_values):
            subset = pixel_data[pixel_data['edge_effect'] == edge_val]
            
            if not subset.empty:
                # Use larger uniform pixel size
                sizes = np.full(len(subset), 200)  # Bigger fixed size for all pixels
                
                ax_map.scatter(subset['longitude'], subset['latitude'],
                              c=colors[i], s=sizes, alpha=0.8, 
                              edgecolors='white', linewidth=0.5,
                              label=f'Edge Effect {edge_val} (n={len(subset)})',
                              zorder=3)
        
        # Plot AWS station
        aws_station = self.config['aws_station']
        ax_map.scatter(aws_station['lon'], aws_station['lat'],
                      c=self.glacier_colors['aws_station'], s=300, marker='^',
                      edgecolor='black', linewidth=2,
                      label=f"AWS Station ({aws_station['elevation']}m)", zorder=4)
        
        # Set title and labels for map
        ax_map.set_title('Haig Glacier - MOD09GA Edge Effect Spatial Distribution', 
                        fontsize=14, fontweight='bold', pad=20)
        ax_map.set_xlabel('Longitude (°)', fontsize=12)
        ax_map.set_ylabel('Latitude (°)', fontsize=12)
        
        # Add legend for map
        ax_map.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9, framealpha=0.9)
        
        # Add grid for map
        ax_map.grid(True, alpha=0.3, linestyle='--')
        
        # Add statistics annotation for map
        self._add_map_statistics_annotation(ax_map, pixel_data)
        
        # Create scatterplot: Glacier Fraction vs Edge Effect
        for i, edge_val in enumerate(edge_values):
            subset = pixel_data[pixel_data['edge_effect'] == edge_val]
            if not subset.empty:
                ax_scatter.scatter(subset['glacier_fraction'], subset['edge_effect'],
                                 c=colors[i], s=120, alpha=0.8,
                                 edgecolors='white', linewidth=1,
                                 label=f'Edge Effect {edge_val}')
        
        ax_scatter.set_xlabel('Glacier Fraction', fontsize=12, fontweight='bold')
        ax_scatter.set_ylabel('Edge Effect Score', fontsize=12, fontweight='bold')
        ax_scatter.set_title('Glacier Fraction vs Edge Effect Score', fontsize=14, fontweight='bold')
        ax_scatter.grid(True, alpha=0.3, linestyle='--')
        ax_scatter.set_ylim(0.5, 8.5)
        ax_scatter.set_xlim(-0.05, 1.05)
        
        # Add trend information
        correlation = pixel_data['glacier_fraction'].corr(pixel_data['edge_effect'])
        ax_scatter.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                       transform=ax_scatter.transAxes, fontsize=11,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        fig.savefig(output_path, dpi=self.config['visualization']['dpi'], 
                   bbox_inches='tight', facecolor='white')
        logger.info(f"Edge effect map saved: {output_path}")
        
        return fig
    
    def _scale_pixel_sizes(self, glacier_fractions: pd.Series) -> np.ndarray:
        """Scale pixel sizes based on glacier fraction."""
        min_size = self.config['visualization']['min_pixel_size']
        max_size = self.config['visualization']['max_pixel_size']
        
        # Normalize glacier fractions to size range
        min_frac = glacier_fractions.min()
        max_frac = glacier_fractions.max()
        
        if max_frac > min_frac:
            normalized = (glacier_fractions - min_frac) / (max_frac - min_frac)
            sizes = min_size + normalized * (max_size - min_size)
        else:
            sizes = np.full(len(glacier_fractions), (min_size + max_size) / 2)
        
        return sizes
    
    def _add_map_statistics_annotation(self, ax, pixel_data: pd.DataFrame):
        """Add statistical summary to the map."""
        total_pixels = len(pixel_data)
        mean_glacier_frac = pixel_data['glacier_fraction'].mean()
        date_range = f"2023-2025"  # Based on current filename
        
        stats_text = (
            f"Dataset Summary:\\n"
            f"• Total pixels: {total_pixels:,}\\n"
            f"• Mean glacier fraction: {mean_glacier_frac:.3f}\\n"
            f"• Time period: {date_range}\\n"
            f"• Edge effect range: {pixel_data['edge_effect'].min()}-{pixel_data['edge_effect'].max()}"
        )
        
        ax.text(0.02, 0.02, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='bottom',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))


def main():
    """Main execution function."""
    logger.info("Starting Haig Glacier Edge Effect Map Generation")
    
    try:
        # Initialize components
        data_loader = EdgeEffectDataLoader(CONFIG)
        mask_loader = GlacierMaskLoader()
        visualizer = EdgeEffectMapVisualizer(CONFIG)
        
        # Load and process data
        logger.info(f"\\n{'='*60}")
        logger.info("Loading and Processing Edge Effect Data")
        logger.info(f"{'='*60}")
        
        # Load edge effect data
        edge_data = data_loader.load_edge_effect_data()
        
        # Aggregate by pixel
        pixel_data = data_loader.aggregate_pixel_data(edge_data)
        
        # Load glacier mask
        glacier_mask = mask_loader.load_haig_glacier_mask(CONFIG['data_paths']['glacier_mask'])
        
        # Create visualization
        logger.info(f"\\n{'='*60}")
        logger.info("Creating Edge Effect Map Visualization")
        logger.info(f"{'='*60}")
        
        output_path = "haig_glacier_edge_effect_map.png"
        fig = visualizer.create_edge_effect_map(pixel_data, glacier_mask, output_path)
        
        # Show the plot
        plt.show()
        
        logger.info(f"\\n✅ SUCCESS: Edge effect map generated and saved to {output_path}")
        logger.info(f"📊 Total pixels visualized: {len(pixel_data):,}")
        
    except Exception as e:
        logger.error(f"❌ Error generating edge effect map: {e}")
        raise


if __name__ == "__main__":
    main()