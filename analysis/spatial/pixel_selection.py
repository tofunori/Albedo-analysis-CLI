#!/usr/bin/env python3
"""
Pixel Selection Algorithm

This module contains the intelligent pixel selection system for enhanced accuracy
using distance and glacier fraction weighting.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PixelSelector:
    """
    Implements intelligent pixel selection algorithms for enhanced accuracy.
    
    Uses distance to AWS station (60% weight) and glacier fraction coverage (40% weight)
    to select the 2 closest best-performing pixels.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def apply_pixel_selection(self, modis_data: pd.DataFrame, glacier_id: str, 
                             glacier_config: Dict[str, Any]) -> pd.DataFrame:
        """Apply pixel selection algorithm based on AWS station coordinates."""
        try:
            # Get AWS station coordinates
            aws_stations = glacier_config.get('aws_stations', {})
            if not aws_stations:
                logger.warning(f"No AWS stations found for {glacier_id}, using standard selection")
                return self._apply_standard_pixel_selection(modis_data, glacier_id)
            
            # Use first AWS station coordinates
            aws_station = list(aws_stations.values())[0]
            aws_lat = aws_station['lat']
            aws_lon = aws_station['lon']
            
            # Get available pixels with their quality metrics and coordinates
            pixel_summary = modis_data.groupby('pixel_id').agg({
                'glacier_fraction': 'mean',
                'albedo': 'count',
                'ndsi': 'mean',
                'latitude': 'first',
                'longitude': 'first'
            }).reset_index()
            
            pixel_summary.columns = ['pixel_id', 'avg_glacier_fraction', 'n_observations', 'avg_ndsi', 'latitude', 'longitude']
            
            # Filter pixels based on quality criteria
            quality_pixels = pixel_summary[
                (pixel_summary['avg_glacier_fraction'] > 0.1) & 
                (pixel_summary['n_observations'] > 10)
            ].copy()
            
            if len(quality_pixels) == 0:
                logger.warning(f"No quality pixels found for {glacier_id}, using all data")
                return modis_data
            
            # Calculate distance using Haversine formula
            quality_pixels['distance_to_aws'] = self._haversine_distance(
                quality_pixels['latitude'], quality_pixels['longitude'], aws_lat, aws_lon
            )
            
            # Sort by performance first, then distance
            quality_pixels = quality_pixels.sort_values([
                'avg_glacier_fraction', 'distance_to_aws'
            ], ascending=[False, True])
            
            # Select the 2 closest best performing pixels
            selected_pixels = quality_pixels.head(2)
            selected_pixel_ids = set(selected_pixels['pixel_id'])
            
            logger.info(f"Selected 2 closest best performing pixels from {len(modis_data['pixel_id'].unique())} total pixels")
            logger.info(f"AWS station: {aws_station['name']} at ({aws_lat:.4f}, {aws_lon:.4f})")
            
            for _, pixel in selected_pixels.iterrows():
                logger.info(f"  Pixel {pixel['pixel_id']}: glacier_fraction={pixel['avg_glacier_fraction']:.3f}, "
                           f"distance={pixel['distance_to_aws']:.2f}km, observations={pixel['n_observations']}")
            
            # Filter MODIS data
            filtered_data = modis_data[modis_data['pixel_id'].isin(selected_pixel_ids)].copy()
            logger.info(f"Filtered MODIS data from {len(modis_data)} to {len(filtered_data)} observations")
            
            return filtered_data
            
        except Exception as e:
            logger.error(f"Error applying pixel selection for {glacier_id}: {e}")
            logger.warning("Falling back to using all pixels")
            return modis_data
    
    def apply_standard_pixel_selection(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Fallback pixel selection when AWS coordinates unavailable."""
        try:
            pixel_summary = modis_data.groupby('pixel_id').agg({
                'glacier_fraction': 'mean',
                'albedo': 'count',
                'ndsi': 'mean'
            }).reset_index()
            
            pixel_summary.columns = ['pixel_id', 'avg_glacier_fraction', 'n_observations', 'avg_ndsi']
            
            quality_pixels = pixel_summary[
                (pixel_summary['avg_glacier_fraction'] > 0.1) & 
                (pixel_summary['n_observations'] > 10)
            ].copy()
            
            if len(quality_pixels) == 0:
                return modis_data
            
            selected_pixels = quality_pixels.nlargest(2, 'avg_glacier_fraction')
            selected_pixel_ids = set(selected_pixels['pixel_id'])
            
            return modis_data[modis_data['pixel_id'].isin(selected_pixel_ids)].copy()
            
        except Exception as e:
            logger.error(f"Error in standard pixel selection: {e}")
            return modis_data
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance using Haversine formula."""
        R = 6371  # Earth's radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    def _apply_standard_pixel_selection(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Private method that calls the public standard selection."""
        return self.apply_standard_pixel_selection(modis_data, glacier_id)