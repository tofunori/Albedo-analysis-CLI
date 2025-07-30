import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize, LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class MapGenerator:
    """Generate maps for spatial visualization of albedo data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.figure_size = self.viz_config.get('figure_size', [12, 10])
        self.dpi = self.viz_config.get('dpi', 300)
        
    def create_glacier_overview_map(self, glacier_mask: gpd.GeoDataFrame,
                                  aws_coordinates: Dict[str, Dict[str, float]],
                                  modis_data: Optional[pd.DataFrame] = None,
                                  title: str = "Glacier Overview Map",
                                  output_path: Optional[str] = None) -> plt.Figure:
        """Create overview map of glacier with AWS stations and MODIS coverage."""
        
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size, 
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, alpha=0.8)
        ax.add_feature(cfeature.BORDERS, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.3)
        
        # Plot glacier mask
        glacier_mask.plot(ax=ax, color='white', edgecolor='blue', 
                         linewidth=2, alpha=0.8, label='Glacier extent',
                         transform=ccrs.PlateCarree())
        
        # Plot AWS stations
        aws_lons = [coords['lon'] for coords in aws_coordinates.values() 
                   if coords['lon'] is not None]
        aws_lats = [coords['lat'] for coords in aws_coordinates.values() 
                   if coords['lat'] is not None]
        
        if aws_lons and aws_lats:
            ax.scatter(aws_lons, aws_lats, c='red', s=150, marker='^',
                      label='AWS stations', zorder=5, transform=ccrs.PlateCarree(),
                      edgecolors='black', linewidth=1)
            
            # Add station labels
            for station_id, coords in aws_coordinates.items():
                if coords['lon'] is not None and coords['lat'] is not None:
                    ax.annotate(station_id, (coords['lon'], coords['lat']),
                               xytext=(8, 8), textcoords='offset points',
                               fontsize=10, ha='left', fontweight='bold',
                               transform=ccrs.PlateCarree(),
                               bbox=dict(boxstyle='round,pad=0.2', 
                                        facecolor='white', alpha=0.8))
        
        # Plot MODIS data points if provided
        if modis_data is not None and not modis_data.empty:
            if 'lat' in modis_data.columns and 'lon' in modis_data.columns:
                if 'albedo' in modis_data.columns:
                    # Color by albedo value
                    scatter = ax.scatter(modis_data['lon'], modis_data['lat'], 
                                       c=modis_data['albedo'], cmap='viridis',
                                       s=30, alpha=0.7, transform=ccrs.PlateCarree(),
                                       label='MODIS pixels')
                    
                    # Add colorbar
                    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20)
                    cbar.set_label('Albedo', rotation=270, labelpad=20)
                else:
                    ax.scatter(modis_data['lon'], modis_data['lat'], 
                             c='orange', s=30, alpha=0.7, 
                             transform=ccrs.PlateCarree(),
                             label='MODIS pixels')
        
        # Set map extent
        bounds = glacier_mask.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.2
        ax.set_extent([bounds[0] - buffer, bounds[2] + buffer,
                       bounds[1] - buffer, bounds[3] + buffer],
                      crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
        gl.top_labels = False
        gl.right_labels = False
        
        # Customize plot
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Overview map saved to {output_path}")
        
        return fig
    
    def create_albedo_spatial_map(self, pixel_data: gpd.GeoDataFrame,
                                glacier_mask: gpd.GeoDataFrame,
                                aws_coordinates: Dict[str, Dict[str, float]],
                                value_column: str = 'albedo',
                                title: str = "Spatial Albedo Distribution",
                                cmap: str = 'RdYlBu_r',
                                output_path: Optional[str] = None) -> plt.Figure:
        """Create map showing spatial distribution of albedo values."""
        
        if value_column not in pixel_data.columns:
            logger.error(f"Column {value_column} not found in pixel data")
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size,
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add basic map features
        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
        
        # Plot glacier outline
        glacier_mask.plot(ax=ax, facecolor='none', edgecolor='black',
                         linewidth=2, alpha=0.8, transform=ccrs.PlateCarree())
        
        # Plot albedo pixels
        valid_data = pixel_data[pixel_data[value_column].notna()]
        
        if not valid_data.empty:
            # Create color normalization
            vmin = valid_data[value_column].quantile(0.02)
            vmax = valid_data[value_column].quantile(0.98)
            norm = Normalize(vmin=vmin, vmax=vmax)
            
            # Plot pixels
            valid_data.plot(ax=ax, column=value_column, cmap=cmap, norm=norm,
                           edgecolor='black', linewidth=0.5, alpha=0.8,
                           transform=ccrs.PlateCarree())
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
            cbar.set_label('Albedo', rotation=270, labelpad=20)
        
        # Plot AWS stations
        aws_lons = [coords['lon'] for coords in aws_coordinates.values() 
                   if coords['lon'] is not None]
        aws_lats = [coords['lat'] for coords in aws_coordinates.values() 
                   if coords['lat'] is not None]
        
        if aws_lons and aws_lats:
            ax.scatter(aws_lons, aws_lats, c='red', s=100, marker='^',
                      zorder=5, transform=ccrs.PlateCarree(),
                      edgecolors='white', linewidth=2, label='AWS stations')
        
        # Set extent
        bounds = glacier_mask.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax.set_extent([bounds[0] - buffer, bounds[2] + buffer,
                       bounds[1] - buffer, bounds[3] + buffer],
                      crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Spatial albedo map saved to {output_path}")
        
        return fig
    
    def create_method_comparison_map(self, modis_methods: Dict[str, gpd.GeoDataFrame],
                                   glacier_mask: gpd.GeoDataFrame,
                                   value_column: str = 'albedo',
                                   title: str = "MODIS Method Comparison",
                                   output_path: Optional[str] = None) -> plt.Figure:
        """Create map comparing multiple MODIS methods."""
        
        n_methods = len(modis_methods)
        if n_methods == 0:
            logger.error("No MODIS methods provided")
            return None
        
        # Create subplots
        fig, axes = plt.subplots(1, n_methods, figsize=(self.figure_size[0] * n_methods, self.figure_size[1]),
                                subplot_kw={'projection': ccrs.PlateCarree()})
        
        if n_methods == 1:
            axes = [axes]
        
        # Find common value range for consistent coloring
        all_values = []
        for method_data in modis_methods.values():
            if value_column in method_data.columns:
                all_values.extend(method_data[value_column].dropna().values)
        
        if all_values:
            vmin = np.percentile(all_values, 2)
            vmax = np.percentile(all_values, 98)
            norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            norm = None
        
        for i, (method_name, method_data) in enumerate(modis_methods.items()):
            ax = axes[i]
            
            # Add map features
            ax.add_feature(cfeature.COASTLINE, alpha=0.5)
            ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
            ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
            
            # Plot glacier outline
            glacier_mask.plot(ax=ax, facecolor='none', edgecolor='black',
                             linewidth=2, alpha=0.8, transform=ccrs.PlateCarree())
            
            # Plot method data
            if value_column in method_data.columns:
                valid_data = method_data[method_data[value_column].notna()]
                
                if not valid_data.empty:
                    im = valid_data.plot(ax=ax, column=value_column, cmap='RdYlBu_r',
                                        norm=norm, edgecolor='black', linewidth=0.3,
                                        alpha=0.8, transform=ccrs.PlateCarree())
            
            # Set extent
            bounds = glacier_mask.total_bounds
            buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
            ax.set_extent([bounds[0] - buffer, bounds[2] + buffer,
                           bounds[1] - buffer, bounds[3] + buffer],
                          crs=ccrs.PlateCarree())
            
            # Add gridlines
            gl = ax.gridlines(draw_labels=True, alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False
            
            ax.set_title(method_name, fontsize=12, fontweight='bold')
        
        # Add common colorbar
        if norm is not None:
            sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=axes, shrink=0.6, aspect=20, pad=0.02)
            cbar.set_label('Albedo', rotation=270, labelpad=20)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Method comparison map saved to {output_path}")
        
        return fig
    
    def create_difference_map(self, reference_data: gpd.GeoDataFrame,
                            comparison_data: gpd.GeoDataFrame,
                            glacier_mask: gpd.GeoDataFrame,
                            value_column: str = 'albedo',
                            reference_name: str = "Reference",
                            comparison_name: str = "Comparison",
                            title: Optional[str] = None,
                            output_path: Optional[str] = None) -> plt.Figure:
        """Create map showing differences between two datasets."""
        
        if title is None:
            title = f"Difference Map: {comparison_name} - {reference_name}"
        
        # Calculate differences
        # This is a simplified approach - in practice, you'd need spatial alignment
        common_pixels = comparison_data.index.intersection(reference_data.index)
        
        if len(common_pixels) == 0:
            logger.error("No common pixels found for difference calculation")
            return None
        
        difference_data = comparison_data.loc[common_pixels].copy()
        if value_column in reference_data.columns and value_column in comparison_data.columns:
            difference_data['difference'] = (comparison_data.loc[common_pixels, value_column] - 
                                           reference_data.loc[common_pixels, value_column])
        
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size,
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
        
        # Plot glacier outline
        glacier_mask.plot(ax=ax, facecolor='none', edgecolor='black',
                         linewidth=2, alpha=0.8, transform=ccrs.PlateCarree())
        
        # Plot differences
        if 'difference' in difference_data.columns:
            valid_diff = difference_data[difference_data['difference'].notna()]
            
            if not valid_diff.empty:
                # Use symmetric color scale around zero
                abs_max = np.abs(valid_diff['difference']).quantile(0.95)
                norm = Normalize(vmin=-abs_max, vmax=abs_max)
                
                valid_diff.plot(ax=ax, column='difference', cmap='RdBu_r', norm=norm,
                               edgecolor='black', linewidth=0.3, alpha=0.8,
                               transform=ccrs.PlateCarree())
                
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap='RdBu_r', norm=norm)
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20)
                cbar.set_label('Albedo Difference', rotation=270, labelpad=20)
        
        # Set extent
        bounds = glacier_mask.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax.set_extent([bounds[0] - buffer, bounds[2] + buffer,
                       bounds[1] - buffer, bounds[3] + buffer],
                      crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Difference map saved to {output_path}")
        
        return fig
    
    def create_validation_map(self, modis_data: gpd.GeoDataFrame,
                            aws_coordinates: Dict[str, Dict[str, float]],
                            aws_buffer_km: float = 1.0,
                            glacier_mask: Optional[gpd.GeoDataFrame] = None,
                            title: str = "Validation Data Coverage",
                            output_path: Optional[str] = None) -> plt.Figure:
        """Create map showing validation setup with AWS buffers and MODIS coverage."""
        
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size,
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
        
        # Plot glacier mask if provided
        if glacier_mask is not None:
            glacier_mask.plot(ax=ax, facecolor='white', edgecolor='blue',
                             linewidth=2, alpha=0.6, transform=ccrs.PlateCarree(),
                             label='Glacier extent')
        
        # Plot MODIS pixels
        if not modis_data.empty:
            modis_data.plot(ax=ax, color='orange', alpha=0.6, markersize=20,
                           transform=ccrs.PlateCarree(), label='MODIS pixels')
        
        # Plot AWS stations and buffers
        for station_id, coords in aws_coordinates.items():
            if coords['lon'] is not None and coords['lat'] is not None:
                # Station point
                ax.scatter(coords['lon'], coords['lat'], c='red', s=150, marker='^',
                          zorder=5, transform=ccrs.PlateCarree(),
                          edgecolors='white', linewidth=2)
                
                # Buffer zone (approximate)
                buffer_deg = aws_buffer_km / 111.0  # Rough conversion to degrees
                circle = patches.Circle((coords['lon'], coords['lat']), buffer_deg,
                                      fill=False, edgecolor='red', linewidth=2,
                                      linestyle='--', alpha=0.8, transform=ccrs.PlateCarree())
                ax.add_patch(circle)
                
                # Station label
                ax.annotate(station_id, (coords['lon'], coords['lat']),
                           xytext=(8, 8), textcoords='offset points',
                           fontsize=10, ha='left', fontweight='bold',
                           transform=ccrs.PlateCarree(),
                           bbox=dict(boxstyle='round,pad=0.2', 
                                    facecolor='white', alpha=0.8))
        
        # Set extent based on data
        if glacier_mask is not None:
            bounds = glacier_mask.total_bounds
        elif not modis_data.empty:
            bounds = modis_data.total_bounds
        else:
            # Use AWS coordinates
            aws_lons = [coords['lon'] for coords in aws_coordinates.values() 
                       if coords['lon'] is not None]
            aws_lats = [coords['lat'] for coords in aws_coordinates.values() 
                       if coords['lat'] is not None]
            bounds = [min(aws_lons), min(aws_lats), max(aws_lons), max(aws_lats)]
        
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.3
        ax.set_extent([bounds[0] - buffer, bounds[2] + buffer,
                       bounds[1] - buffer, bounds[3] + buffer],
                      crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='red',
                      markersize=10, label='AWS stations'),
            plt.Line2D([0], [0], color='red', linestyle='--', label=f'AWS buffer ({aws_buffer_km} km)'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                      markersize=8, label='MODIS pixels')
        ]
        
        if glacier_mask is not None:
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor='white',
                                               edgecolor='blue', label='Glacier extent'))
        
        ax.legend(handles=legend_elements, loc='upper right')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Validation map saved to {output_path}")
        
        return fig
    
    def create_elevation_map(self, elevation_zones: gpd.GeoDataFrame,
                           glacier_mask: gpd.GeoDataFrame,
                           title: str = "Glacier Elevation Zones",
                           output_path: Optional[str] = None) -> plt.Figure:
        """Create map showing elevation zones within glacier."""
        
        fig, ax = plt.subplots(1, 1, figsize=self.figure_size,
                              subplot_kw={'projection': ccrs.PlateCarree()})
        
        # Add map features
        ax.add_feature(cfeature.COASTLINE, alpha=0.5)
        ax.add_feature(cfeature.OCEAN, color='lightblue', alpha=0.3)
        ax.add_feature(cfeature.LAND, color='lightgray', alpha=0.2)
        
        # Plot glacier outline
        glacier_mask.plot(ax=ax, facecolor='none', edgecolor='black',
                         linewidth=2, alpha=1.0, transform=ccrs.PlateCarree())
        
        # Plot elevation zones
        if 'elevation_zone' in elevation_zones.columns:
            # Use different colors for each elevation zone
            unique_zones = elevation_zones['elevation_zone'].unique()
            colors = plt.cm.terrain(np.linspace(0.2, 0.8, len(unique_zones)))
            
            for i, zone in enumerate(unique_zones):
                zone_data = elevation_zones[elevation_zones['elevation_zone'] == zone]
                zone_data.plot(ax=ax, color=colors[i], alpha=0.7,
                              edgecolor='black', linewidth=0.5,
                              transform=ccrs.PlateCarree(),
                              label=zone)
        else:
            elevation_zones.plot(ax=ax, alpha=0.7, transform=ccrs.PlateCarree())
        
        # Set extent
        bounds = glacier_mask.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax.set_extent([bounds[0] - buffer, bounds[2] + buffer,
                       bounds[1] - buffer, bounds[3] + buffer],
                      crs=ccrs.PlateCarree())
        
        # Add gridlines
        gl = ax.gridlines(draw_labels=True, alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Elevation map saved to {output_path}")
        
        return fig