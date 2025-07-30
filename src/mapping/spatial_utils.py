import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

logger = logging.getLogger(__name__)


class SpatialProcessor:
    """Spatial processing utilities for MODIS and AWS data integration."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pixel_size = config.get('analysis', {}).get('spatial', {}).get('pixel_size', 500)  # meters
        
    def create_modis_grid(self, bounds: Tuple[float, float, float, float], 
                         pixel_size: Optional[float] = None) -> gpd.GeoDataFrame:
        """Create MODIS pixel grid for given bounds."""
        if pixel_size is None:
            pixel_size = self.pixel_size
            
        minx, miny, maxx, maxy = bounds
        
        # Convert pixel size from meters to degrees (approximate)
        # This is a rough approximation - in practice, you'd use proper projection
        pixel_size_deg = pixel_size / 111000  # ~111 km per degree
        
        # Generate grid coordinates
        x_coords = np.arange(minx, maxx, pixel_size_deg)
        y_coords = np.arange(miny, maxy, pixel_size_deg)
        
        # Create grid polygons
        polygons = []
        pixel_ids = []
        
        for i, x in enumerate(x_coords[:-1]):
            for j, y in enumerate(y_coords[:-1]):
                # Create pixel polygon
                pixel_poly = Polygon([
                    (x, y),
                    (x + pixel_size_deg, y),
                    (x + pixel_size_deg, y + pixel_size_deg),
                    (x, y + pixel_size_deg)
                ])
                
                polygons.append(pixel_poly)
                pixel_ids.append(f"pixel_{i}_{j}")
        
        # Create GeoDataFrame
        grid_gdf = gpd.GeoDataFrame({
            'pixel_id': pixel_ids,
            'geometry': polygons
        }, crs='EPSG:4326')
        
        return grid_gdf
    
    def load_glacier_mask(self, mask_path: str) -> gpd.GeoDataFrame:
        """Load glacier mask from shapefile or raster."""
        try:
            if mask_path.endswith('.shp'):
                mask_gdf = gpd.read_file(mask_path)
            elif mask_path.endswith(('.tif', '.tiff')):
                mask_gdf = self._raster_to_polygon(mask_path)
            else:
                raise ValueError(f"Unsupported mask format: {mask_path}")
            
            # Ensure CRS is WGS84
            if mask_gdf.crs != 'EPSG:4326':
                mask_gdf = mask_gdf.to_crs('EPSG:4326')
            
            return mask_gdf
            
        except Exception as e:
            logger.error(f"Error loading glacier mask from {mask_path}: {e}")
            raise
    
    def _raster_to_polygon(self, raster_path: str) -> gpd.GeoDataFrame:
        """Convert raster mask to polygon."""
        try:
            with rasterio.open(raster_path) as src:
                # Read the raster data
                mask_data = src.read(1)
                transform = src.transform
                crs = src.crs
                
                # Convert raster to polygons
                from rasterio.features import shapes
                
                # Get shapes of valid pixels (assuming non-zero values are glacier)
                mask_binary = mask_data > 0
                shapes_gen = shapes(mask_binary.astype(np.uint8), transform=transform)
                
                polygons = []
                for geom, value in shapes_gen:
                    if value == 1:  # Glacier pixels
                        polygons.append(geom)
                
                # Create GeoDataFrame
                gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
                
                return gdf
                
        except Exception as e:
            logger.error(f"Error converting raster to polygon: {e}")
            raise
    
    def clip_data_to_glacier(self, data_gdf: gpd.GeoDataFrame, 
                           glacier_mask: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clip spatial data to glacier extent."""
        try:
            # Ensure both have same CRS
            if data_gdf.crs != glacier_mask.crs:
                glacier_mask = glacier_mask.to_crs(data_gdf.crs)
            
            # Perform spatial intersection
            clipped_data = gpd.overlay(data_gdf, glacier_mask, how='intersection')
            
            return clipped_data
            
        except Exception as e:
            logger.error(f"Error clipping data to glacier: {e}")
            raise
    
    def create_aws_buffer_zones(self, aws_coordinates: Dict[str, Dict[str, float]], 
                              buffer_distance: Optional[float] = None) -> gpd.GeoDataFrame:
        """Create buffer zones around AWS stations."""
        if buffer_distance is None:
            buffer_distance = self.config.get('analysis', {}).get('spatial', {}).get('buffer_distance', 1000)
        
        # Convert buffer distance from meters to degrees (approximate)
        buffer_deg = buffer_distance / 111000
        
        stations = []
        geometries = []
        
        for station_id, coords in aws_coordinates.items():
            if coords['lat'] is not None and coords['lon'] is not None:
                # Create point and buffer
                point = Point(coords['lon'], coords['lat'])
                buffer_geom = point.buffer(buffer_deg)
                
                stations.append(station_id)
                geometries.append(buffer_geom)
        
        # Create GeoDataFrame
        buffer_gdf = gpd.GeoDataFrame({
            'station_id': stations,
            'geometry': geometries
        }, crs='EPSG:4326')
        
        return buffer_gdf
    
    def spatial_join_modis_aws(self, modis_gdf: gpd.GeoDataFrame, 
                              aws_buffers: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Spatially join MODIS pixels with AWS buffer zones."""
        try:
            # Ensure same CRS
            if modis_gdf.crs != aws_buffers.crs:
                aws_buffers = aws_buffers.to_crs(modis_gdf.crs)
            
            # Spatial join
            joined_gdf = gpd.sjoin(modis_gdf, aws_buffers, how='left', predicate='intersects')
            
            return joined_gdf
            
        except Exception as e:
            logger.error(f"Error in spatial join: {e}")
            raise
    
    def calculate_pixel_statistics(self, pixel_data: pd.DataFrame, 
                                 group_by: str = 'pixel_id') -> pd.DataFrame:
        """Calculate statistics for each pixel."""
        if 'albedo' not in pixel_data.columns:
            logger.error("No albedo column found for pixel statistics")
            return pd.DataFrame()
        
        stats = pixel_data.groupby(group_by)['albedo'].agg([
            'count',
            'mean',
            'std',
            'min',
            'max',
            'median'
        ]).reset_index()
        
        # Rename columns
        stats.columns = [group_by, 'n_observations', 'mean_albedo', 'std_albedo',
                        'min_albedo', 'max_albedo', 'median_albedo']
        
        return stats
    
    def create_spatial_overview_map(self, glacier_mask: gpd.GeoDataFrame,
                                  aws_coordinates: Dict[str, Dict[str, float]],
                                  modis_data: Optional[gpd.GeoDataFrame] = None,
                                  output_path: Optional[str] = None) -> plt.Figure:
        """Create overview map showing glacier, AWS stations, and MODIS pixels."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot glacier mask
        glacier_mask.plot(ax=ax, color='lightblue', alpha=0.7, 
                         edgecolor='blue', linewidth=1, label='Glacier extent')
        
        # Plot AWS stations
        aws_lons = [coords['lon'] for coords in aws_coordinates.values() 
                   if coords['lon'] is not None]
        aws_lats = [coords['lat'] for coords in aws_coordinates.values() 
                   if coords['lat'] is not None]
        
        if aws_lons and aws_lats:
            ax.scatter(aws_lons, aws_lats, c='red', s=100, marker='^',
                      label='AWS stations', zorder=5)
            
            # Add station labels
            for station_id, coords in aws_coordinates.items():
                if coords['lon'] is not None and coords['lat'] is not None:
                    ax.annotate(station_id, (coords['lon'], coords['lat']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=8, ha='left')
        
        # Plot MODIS pixels if provided
        if modis_data is not None and not modis_data.empty:
            if 'albedo' in modis_data.columns:
                # Color pixels by albedo value
                modis_data.plot(ax=ax, column='albedo', cmap='viridis',
                               alpha=0.6, edgecolor='none', legend=True)
            else:
                modis_data.plot(ax=ax, color='orange', alpha=0.5,
                               edgecolor='black', linewidth=0.5,
                               label='MODIS pixels')
        
        # Set map properties
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Glacier Analysis Spatial Overview')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio
        ax.set_aspect('equal')
        
        # Adjust map extent
        bounds = glacier_mask.total_bounds
        buffer = 0.01  # degree buffer
        ax.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
        ax.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Spatial overview map saved to {output_path}")
        
        return fig
    
    def create_pixel_albedo_map(self, pixel_data: gpd.GeoDataFrame,
                              value_column: str = 'mean_albedo',
                              output_path: Optional[str] = None) -> plt.Figure:
        """Create map showing albedo values by pixel."""
        if value_column not in pixel_data.columns:
            logger.error(f"Column {value_column} not found in pixel data")
            return None
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot pixels colored by albedo
        pixel_data.plot(ax=ax, column=value_column, cmap='RdYlBu_r',
                       edgecolor='black', linewidth=0.5, legend=True)
        
        # Customize colorbar
        cbar = ax.get_figure().get_axes()[-1]  # Get colorbar axis
        cbar.set_ylabel('Albedo', rotation=270, labelpad=20)
        
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(f'MODIS Pixel {value_column.replace("_", " ").title()}')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Pixel albedo map saved to {output_path}")
        
        return fig
    
    def calculate_spatial_statistics(self, data_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Calculate spatial statistics for the dataset."""
        if data_gdf.empty:
            return {}
        
        bounds = data_gdf.total_bounds
        
        stats = {
            'total_features': len(data_gdf),
            'spatial_extent': {
                'min_lon': bounds[0],
                'min_lat': bounds[1],
                'max_lon': bounds[2],
                'max_lat': bounds[3],
                'width_deg': bounds[2] - bounds[0],
                'height_deg': bounds[3] - bounds[1]
            },
            'total_area_km2': data_gdf.to_crs('EPSG:3857').area.sum() / 1e6  # Convert to km²
        }
        
        # If albedo data is present, calculate spatial albedo statistics
        if 'albedo' in data_gdf.columns:
            albedo_data = data_gdf['albedo'].dropna()
            stats['albedo_spatial'] = {
                'mean': albedo_data.mean(),
                'std': albedo_data.std(),
                'min': albedo_data.min(),
                'max': albedo_data.max(),
                'median': albedo_data.median()
            }
        
        return stats
    
    def export_spatial_data(self, data_gdf: gpd.GeoDataFrame, 
                           output_path: str,
                           format: str = 'shapefile') -> None:
        """Export spatial data to file."""
        try:
            if format.lower() == 'shapefile':
                data_gdf.to_file(output_path)
            elif format.lower() == 'geojson':
                data_gdf.to_file(output_path, driver='GeoJSON')
            elif format.lower() == 'gpkg':
                data_gdf.to_file(output_path, driver='GPKG')
            else:
                raise ValueError(f"Unsupported export format: {format}")
                
            logger.info(f"Spatial data exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting spatial data: {e}")
            raise