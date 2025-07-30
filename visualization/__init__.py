#!/usr/bin/env python3
"""
Visualization Module

This module provides comprehensive visualization capabilities for albedo analysis,
including statistical plots, spatial maps, and interactive visualizations.
"""

from .plots.statistical_plots import PlotGenerator
from .plots.time_series_plots import TimeSeriesPlotter

# Optional map generator (requires cartopy)
try:
    from .maps.map_generator import MapGenerator
    MAP_GENERATOR_AVAILABLE = True
except ImportError:
    MapGenerator = None
    MAP_GENERATOR_AVAILABLE = False

__all__ = [
    'PlotGenerator',
    'TimeSeriesPlotter'
]

if MAP_GENERATOR_AVAILABLE:
    __all__.append('MapGenerator')