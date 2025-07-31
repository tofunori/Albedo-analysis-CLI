#!/usr/bin/env python3
"""
Statistical Plots Module

This module contains all statistical plotting functionality for albedo analysis.
The modular system provides specialized plotters while maintaining backward compatibility.
"""

# Main facade class for backward compatibility
from .plot_generator import PlotGenerator

# Individual specialized plotters for advanced usage
from .base import BasePlotter
from .scatter_plots import ScatterPlotter
from .distribution_plots import DistributionPlotter
from .time_series_plots import TimeSeriesPlotter
from .comparison_plots import ComparisonPlotter
from .comprehensive_plots import ComprehensivePlotter
from .correlation_plots import CorrelationPlotter

# Backward compatibility - import from original file if needed
try:
    from .statistical_plots import PlotGenerator as LegacyPlotGenerator
except ImportError:
    # If original file doesn't exist or has issues, use new system
    LegacyPlotGenerator = PlotGenerator

__all__ = [
    # Main interface
    'PlotGenerator',
    
    # Specialized plotters
    'BasePlotter',
    'ScatterPlotter', 
    'DistributionPlotter',
    'TimeSeriesPlotter',
    'ComparisonPlotter',
    'ComprehensivePlotter',
    'CorrelationPlotter',
    
    # Backward compatibility
    'LegacyPlotGenerator'
]