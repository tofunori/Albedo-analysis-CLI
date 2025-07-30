import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import logging
from typing import Dict, Any, List, Tuple, Optional, Union
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class GlacierMaskProcessor:
    """Processing utilities for glacier masks and extent analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_and_validate_mask(self, mask_path: str) -> gpd.GeoDataFrame:
        """Load glacier mask and perform validation."""
        try:
            # Load the mask
            if mask_path.endswith('.shp'):
                mask_gdf = gpd.read_file(mask_path)
            elif mask_path.endswith(('.tif', '.tiff')):
                mask_gdf = self._raster_to_vector(mask_path)
            else:
                raise ValueError(f"Unsupported mask format: {mask_path}")
            
            # Validate the mask
            validation_results = self._validate_mask(mask_gdf)
            
            if not validation_results['is_valid']:
                logger.warning(f"Mask validation issues: {validation_results['issues']}")
            
            # Clean the mask if needed
            mask_gdf = self._clean_mask(mask_gdf)
            
            # Ensure WGS84 CRS
            if mask_gdf.crs != 'EPSG:4326':
                mask_gdf = mask_gdf.to_crs('EPSG:4326')
            
            logger.info(f"Successfully loaded glacier mask with {len(mask_gdf)} features")
            return mask_gdf
            
        except Exception as e:
            logger.error(f"Error loading glacier mask: {e}")
            raise
    
    def _raster_to_vector(self, raster_path: str) -> gpd.GeoDataFrame:
        """Convert raster mask to vector format."""
        with rasterio.open(raster_path) as src:
            # Read raster data
            raster_data = src.read(1)
            transform = src.transform
            crs = src.crs
            
            # Create binary mask (assuming non-zero values are glacier)
            binary_mask = raster_data > 0
            
            # Convert to polygons
            from rasterio.features import shapes
            
            polygons = []
            for geom, value in shapes(binary_mask.astype(np.uint8), transform=transform):
                if value == 1:
                    polygons.append(geom)
            
            # Create GeoDataFrame
            gdf = gpd.GeoDataFrame({'geometry': polygons}, crs=crs)
            
            # Dissolve to create single glacier outline
            dissolved = gdf.dissolve()
            
            return dissolved.reset_index(drop=True)
    
    def _validate_mask(self, mask_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Validate glacier mask geometry and properties."""
        issues = []
        
        # Check if empty
        if mask_gdf.empty:
            issues.append("Mask is empty")
        
        # Check geometry validity
        invalid_geoms = ~mask_gdf.geometry.is_valid
        if invalid_geoms.any():
            issues.append(f"{invalid_geoms.sum()} invalid geometries found")
        
        # Check for very small features (likely noise)
        if mask_gdf.crs and mask_gdf.crs.to_epsg() == 4326:
            # Convert to projected CRS for area calculation
            mask_projected = mask_gdf.to_crs('EPSG:3857')
            areas_m2 = mask_projected.geometry.area
            small_features = areas_m2 < 1000  # Less than 1000 m²
            
            if small_features.any():
                issues.append(f"{small_features.sum()} very small features found")
        
        # Check coordinate ranges (should be reasonable for Earth)
        bounds = mask_gdf.total_bounds
        if not (-180 <= bounds[0] <= bounds[2] <= 180):
            issues.append("Invalid longitude range")
        if not (-90 <= bounds[1] <= bounds[3] <= 90):
            issues.append("Invalid latitude range")
        
        return {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'feature_count': len(mask_gdf),
            'bounds': bounds
        }
    
    def _clean_mask(self, mask_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Clean glacier mask geometry."""
        cleaned = mask_gdf.copy()
        
        # Fix invalid geometries
        invalid_mask = ~cleaned.geometry.is_valid
        if invalid_mask.any():
            logger.info(f"Fixing {invalid_mask.sum()} invalid geometries")
            cleaned.loc[invalid_mask, 'geometry'] = cleaned.loc[invalid_mask, 'geometry'].buffer(0)
        
        # Remove very small features if in geographic coordinates
        if cleaned.crs and cleaned.crs.to_epsg() == 4326:
            # Use rough area threshold in degrees²
            area_threshold = 1e-8  # Very small area in degrees²
            large_enough = cleaned.geometry.area >= area_threshold
            cleaned = cleaned[large_enough]
            
            if not large_enough.all():
                logger.info(f"Removed {(~large_enough).sum()} very small features")
        
        return cleaned.reset_index(drop=True)
    
    def create_glacier_outline(self, mask_gdf: gpd.GeoDataFrame, 
                             simplify_tolerance: float = None) -> gpd.GeoDataFrame:
        """Create simplified glacier outline."""
        # Dissolve all features into single outline
        outline = mask_gdf.dissolve()
        
        # Simplify geometry if requested
        if simplify_tolerance is not None:
            outline['geometry'] = outline['geometry'].simplify(simplify_tolerance)
        
        return outline.reset_index(drop=True)
    
    def calculate_glacier_properties(self, mask_gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Calculate glacier geometric properties."""
        if mask_gdf.empty:
            return {}
        
        # Get bounds
        bounds = mask_gdf.total_bounds
        
        # Convert to projected CRS for accurate area calculation
        if mask_gdf.crs.to_epsg() == 4326:
            # Use appropriate UTM zone or equal area projection
            mask_projected = mask_gdf.to_crs('EPSG:3857')  # Web Mercator (rough approximation)
        else:
            mask_projected = mask_gdf
        
        # Calculate areas
        areas_m2 = mask_projected.geometry.area
        total_area_km2 = areas_m2.sum() / 1e6
        
        # Calculate perimeter
        perimeters_m = mask_projected.geometry.length
        total_perimeter_km = perimeters_m.sum() / 1000
        
        # Calculate centroids
        centroids = mask_gdf.geometry.centroid
        
        properties = {
            'total_area_km2': total_area_km2,
            'total_perimeter_km': total_perimeter_km,
            'number_of_features': len(mask_gdf),
            'centroid_lat': centroids.y.mean(),
            'centroid_lon': centroids.x.mean(),
            'bounds': {
                'min_lon': bounds[0],
                'min_lat': bounds[1],
                'max_lon': bounds[2],
                'max_lat': bounds[3]
            },
            'extent': {
                'width_deg': bounds[2] - bounds[0],
                'height_deg': bounds[3] - bounds[1]
            }
        }
        
        # Additional properties for individual features
        if len(mask_gdf) > 1:
            properties['feature_areas_km2'] = (areas_m2 / 1e6).tolist()
            properties['largest_feature_km2'] = (areas_m2 / 1e6).max()
            properties['smallest_feature_km2'] = (areas_m2 / 1e6).min()
        
        return properties
    
    def create_elevation_zones(self, mask_gdf: gpd.GeoDataFrame,
                             dem_path: Optional[str] = None,
                             elevation_bands: List[int] = None) -> gpd.GeoDataFrame:
        """Create elevation zones within glacier mask."""
        if dem_path is None:
            logger.warning("No DEM provided for elevation zones")
            return mask_gdf
        
        if elevation_bands is None:
            elevation_bands = [0, 500, 1000, 1500, 2000, 3000, 5000]
        
        try:
            with rasterio.open(dem_path) as dem_src:
                # Ensure mask is in same CRS as DEM
                if mask_gdf.crs != dem_src.crs:
                    mask_reprojected = mask_gdf.to_crs(dem_src.crs)
                else:
                    mask_reprojected = mask_gdf
                
                # Extract elevation values within glacier mask
                elevation_zones = []
                
                for i, (low, high) in enumerate(zip(elevation_bands[:-1], elevation_bands[1:])):
                    # Read DEM data for current elevation band
                    dem_data, dem_transform = mask(dem_src, mask_reprojected.geometry, 
                                                  crop=True, nodata=np.nan)
                    dem_data = dem_data[0]  # Get first band
                    
                    # Create mask for current elevation band
                    elevation_mask = (dem_data >= low) & (dem_data < high)
                    
                    if elevation_mask.any():
                        # Convert elevation mask to polygons
                        from rasterio.features import shapes
                        
                        polygons = []
                        for geom, value in shapes(elevation_mask.astype(np.uint8), 
                                                transform=dem_transform):
                            if value == 1:
                                polygons.append(geom)
                        
                        if polygons:
                            zone_gdf = gpd.GeoDataFrame({
                                'elevation_zone': f"{low}-{high}m",
                                'min_elevation': low,
                                'max_elevation': high,
                                'geometry': polygons
                            }, crs=dem_src.crs)
                            
                            elevation_zones.append(zone_gdf)
                
                if elevation_zones:
                    all_zones = pd.concat(elevation_zones, ignore_index=True)
                    
                    # Convert back to original CRS
                    if all_zones.crs != mask_gdf.crs:
                        all_zones = all_zones.to_crs(mask_gdf.crs)
                    
                    return all_zones
                else:
                    logger.warning("No elevation zones created")
                    return gpd.GeoDataFrame()
                    
        except Exception as e:
            logger.error(f"Error creating elevation zones: {e}")
            return mask_gdf
    
    def mask_modis_data(self, modis_data: pd.DataFrame, 
                       mask_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """Mask MODIS data to glacier extent."""
        if 'lat' not in modis_data.columns or 'lon' not in modis_data.columns:
            logger.error("MODIS data missing lat/lon columns")
            return pd.DataFrame()
        
        # Create points from MODIS coordinates
        modis_points = gpd.GeoDataFrame(
            modis_data,
            geometry=gpd.points_from_xy(modis_data.lon, modis_data.lat),
            crs='EPSG:4326'
        )
        
        # Ensure same CRS
        if modis_points.crs != mask_gdf.crs:
            mask_gdf = mask_gdf.to_crs(modis_points.crs)
        
        # Perform spatial join to keep only points within glacier
        masked_data = gpd.sjoin(modis_points, mask_gdf, how='inner', predicate='within')
        
        # Convert back to regular DataFrame
        result_df = pd.DataFrame(masked_data.drop('geometry', axis=1))
        
        logger.info(f"Masked MODIS data: {len(modis_data)} -> {len(result_df)} points")
        
        return result_df
    
    def create_mask_visualization(self, mask_gdf: gpd.GeoDataFrame,
                                aws_coordinates: Optional[Dict[str, Dict[str, float]]] = None,
                                modis_points: Optional[pd.DataFrame] = None,
                                output_path: Optional[str] = None) -> plt.Figure:
        """Create visualization of glacier mask with optional data overlay."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        
        # Plot glacier mask
        mask_gdf.plot(ax=ax, color='lightblue', alpha=0.7, 
                     edgecolor='blue', linewidth=2, label='Glacier extent')
        
        # Plot AWS stations if provided
        if aws_coordinates:
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
        
        # Plot MODIS points if provided
        if modis_points is not None and not modis_points.empty:
            if 'lat' in modis_points.columns and 'lon' in modis_points.columns:
                ax.scatter(modis_points.lon, modis_points.lat, 
                          c='orange', s=20, alpha=0.6, 
                          label='MODIS pixels', zorder=3)
        
        # Set map properties
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Glacier Mask and Data Coverage')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Adjust extent
        bounds = mask_gdf.total_bounds
        buffer = max(bounds[2] - bounds[0], bounds[3] - bounds[1]) * 0.1
        ax.set_xlim(bounds[0] - buffer, bounds[2] + buffer)
        ax.set_ylim(bounds[1] - buffer, bounds[3] + buffer)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Mask visualization saved to {output_path}")
        
        return fig
    
    def export_mask_summary(self, mask_gdf: gpd.GeoDataFrame, 
                           output_path: str) -> None:
        """Export glacier mask summary to file."""
        properties = self.calculate_glacier_properties(mask_gdf)
        
        # Create summary DataFrame
        summary_data = []
        for key, value in properties.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    summary_data.append({
                        'property': f"{key}.{sub_key}",
                        'value': sub_value
                    })
            else:
                summary_data.append({
                    'property': key,
                    'value': value
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path, index=False)
        
        logger.info(f"Glacier mask summary exported to {output_path}")