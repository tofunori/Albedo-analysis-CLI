#!/usr/bin/env python3
"""
Plot Generator - Facade class for the modular plotting system.

This module provides a unified interface that composes all specialized
plotters while maintaining backward compatibility with the original
statistical_plots.py interface.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Union

# Import all specialized plotters
from .scatter_plots import ScatterPlotter
from .distribution_plots import DistributionPlotter
from .time_series_plots import TimeSeriesPlotter
from .comparison_plots import ComparisonPlotter
from .comprehensive_plots import ComprehensivePlotter
from .correlation_plots import CorrelationPlotter

logger = logging.getLogger(__name__)


class PlotGenerator:
    """
    Facade class that composes all specialized plotters.
    
    This class maintains backward compatibility with the original
    statistical_plots.py interface while using the new modular system.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the plot generator with configuration."""
        self.config = config
        
        # Initialize all specialized plotters
        self.scatter_plotter = ScatterPlotter(config)
        self.distribution_plotter = DistributionPlotter(config)
        self.time_series_plotter = TimeSeriesPlotter(config)
        self.comparison_plotter = ComparisonPlotter(config)
        self.comprehensive_plotter = ComprehensivePlotter(config)
        self.correlation_plotter = CorrelationPlotter(config)
        
        logger.info("PlotGenerator initialized with modular plotting system")
    
    # === SCATTER PLOTS ===
    
    def create_scatter_plot(self, x_data: pd.Series, y_data: pd.Series,
                          x_label: str = "X", y_label: str = "Y",
                          title: str = "Scatter Plot",
                          output_path: Optional[str] = None) -> plt.Figure:
        """Create scatter plot."""
        return self.scatter_plotter.create_scatterplot(
            x_data, y_data, x_label, y_label, title, output_path=output_path)
    
    def create_enhanced_scatter_plot(self, aws_data: pd.Series, 
                                   modis_data: pd.Series,
                                   method_name: str = "MODIS",
                                   title: str = "Enhanced Scatter Plot",
                                   output_path: Optional[str] = None) -> plt.Figure:
        """Create enhanced scatter plot with statistics."""
        return self.scatter_plotter.create_scatterplot(
            aws_data, modis_data, "AWS Albedo", f"{method_name} Albedo", 
            title, method_name, output_path=output_path)
    
    def create_multi_method_scatter(self, aws_data: pd.Series,
                                  modis_methods: Dict[str, pd.Series],
                                  title: str = "Multi-Method Scatter Analysis",
                                  output_path: Optional[str] = None) -> plt.Figure:
        """Create multi-method scatter plot analysis."""
        return self.scatter_plotter.create_multi_method_scatter(
            aws_data, modis_methods, title, output_path)
    
    # === DISTRIBUTION PLOTS ===
    
    def create_boxplot(self, data: Dict[str, pd.Series],
                      title: str = "Albedo Distribution Comparison",
                      y_label: str = "Albedo",
                      output_path: Optional[str] = None) -> plt.Figure:
        """Create box plot comparing albedo distributions."""
        return self.distribution_plotter.create_boxplot(
            data, title, y_label, output_path)
    
    def create_difference_boxplot(self, reference_data: pd.Series,
                                comparison_data: Dict[str, pd.Series],
                                title: str = "Albedo Difference Distribution",
                                reference_name: str = "AWS",
                                output_path: Optional[str] = None) -> plt.Figure:
        """Create boxplot of differences between methods and reference."""
        return self.distribution_plotter.create_difference_boxplot(
            reference_data, comparison_data, title, reference_name, output_path)
    
    def create_refined_distribution_analysis(self, data: Dict[str, pd.Series],
                                           title: str = "Distribution Analysis",
                                           output_path: Optional[str] = None) -> plt.Figure:
        """Create refined distribution analysis with multiple visualization types."""
        return self.distribution_plotter.create_refined_distribution_analysis(
            data, title, output_path)
    
    # === TIME SERIES PLOTS ===
    
    def create_time_series_plot(self, data: pd.DataFrame,
                              date_column: str = 'date',
                              value_column: str = 'albedo',
                              method_column: str = 'method',
                              title: str = "Albedo Time Series",
                              output_path: Optional[str] = None) -> plt.Figure:
        """Create multi-panel time series plot with separate subplot for each year."""
        return self.time_series_plotter.create_time_series_plot(
            data, date_column, value_column, method_column, title, output_path)
    
    def create_seasonal_comparison(self, data: pd.DataFrame,
                                 date_column: str = 'date',
                                 value_column: str = 'albedo',
                                 method_column: str = 'method',
                                 title: str = "Seasonal Albedo Comparison",
                                 output_path: Optional[str] = None) -> plt.Figure:
        """Create seasonal comparison plot."""
        return self.time_series_plotter.create_seasonal_comparison(
            data, date_column, value_column, method_column, title, output_path)
    
    def create_original_seasonal_analysis(self, data: pd.DataFrame,
                                        date_column: str = 'date',
                                        value_column: str = 'albedo',
                                        method_column: str = 'method',
                                        glacier_name: str = "Glacier",
                                        output_path: Optional[str] = None) -> plt.Figure:
        """Create original 4-panel seasonal analysis (monthly boxplots)."""
        return self.time_series_plotter.create_original_seasonal_analysis(
            data, date_column, value_column, method_column, glacier_name, output_path)
    
    # === COMPARISON PLOTS ===
    
    def create_four_metrics_boxplot_summary(self, aws_data: pd.Series, 
                                           modis_methods: Dict[str, pd.Series],
                                           glacier_name: str = "Glacier",
                                           output_path: Optional[str] = None) -> plt.Figure:
        """Create publication-ready single grouped bar chart showing performance metrics for all methods."""
        return self.comparison_plotter.create_four_metrics_boxplot_summary(
            aws_data, modis_methods, glacier_name, output_path)
    
    def create_refined_method_comparison(self, aws_data: pd.Series, 
                                       modis_methods: Dict[str, pd.Series],
                                       title: str = "Method Performance Analysis",
                                       output_path: Optional[str] = None) -> plt.Figure:
        """Create refined method comparison with multiple visualization approaches."""
        return self.comparison_plotter.create_refined_method_comparison(
            aws_data, modis_methods, title, output_path)
    
    def create_original_correlation_bias_analysis(self, aws_data: pd.Series, 
                                                modis_methods: Dict[str, pd.Series],
                                                glacier_name: str = "Glacier",
                                                output_path: Optional[str] = None) -> plt.Figure:
        """Create original 4-panel correlation and bias analysis."""
        return self.comparison_plotter.create_original_correlation_bias_analysis(
            aws_data, modis_methods, glacier_name, output_path)
    
    # === COMPREHENSIVE PLOTS ===
    
    def create_comprehensive_analysis(self, aws_data: pd.Series, 
                                    modis_methods: Dict[str, pd.Series],
                                    glacier_name: str = "Glacier",
                                    output_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive analysis dashboard."""
        return self.comprehensive_plotter.create_comprehensive_analysis(
            aws_data, modis_methods, glacier_name, output_path)
    
    def create_method_comparison_dashboard(self, aws_data: pd.Series, 
                                         modis_methods: Dict[str, pd.Series],
                                         glacier_name: str = "Glacier",
                                         output_path: Optional[str] = None) -> plt.Figure:
        """Create method comparison dashboard with advanced visualizations."""
        return self.comprehensive_plotter.create_method_comparison_dashboard(
            aws_data, modis_methods, glacier_name, output_path)
    
    # === CORRELATION PLOTS ===
    
    def create_correlation_matrix(self, data: Dict[str, pd.Series],
                                title: str = "Albedo Correlation Matrix",
                                output_path: Optional[str] = None) -> plt.Figure:
        """Create correlation matrix heatmap."""
        return self.correlation_plotter.create_correlation_matrix(
            data, title, output_path)
    
    def create_scatter_matrix(self, data: Dict[str, pd.Series],
                            title: str = "Albedo Scatter Matrix",
                            output_path: Optional[str] = None) -> plt.Figure:
        """Create scatter plot matrix for multiple datasets."""
        return self.correlation_plotter.create_scatter_matrix(
            data, title, output_path)
    
    def create_pairwise_correlation_analysis(self, reference_data: pd.Series,
                                           comparison_data: Dict[str, pd.Series],
                                           reference_name: str = "AWS",
                                           title: str = "Pairwise Correlation Analysis",
                                           output_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive pairwise correlation analysis."""
        return self.correlation_plotter.create_pairwise_correlation_analysis(
            reference_data, comparison_data, reference_name, title, output_path)
    
    def create_method_relationship_analysis(self, data: Dict[str, pd.Series],
                                          title: str = "Method Relationship Analysis",
                                          output_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive method relationship analysis."""
        return self.correlation_plotter.create_method_relationship_analysis(
            data, title, output_path)
    
    def create_advanced_correlation_plot(self, reference_data: pd.Series,
                                       comparison_data: Dict[str, pd.Series],
                                       reference_name: str = "AWS",
                                       title: str = "Advanced Correlation Analysis",
                                       output_path: Optional[str] = None) -> plt.Figure:
        """Create advanced correlation plot with confidence intervals and regression."""
        return self.correlation_plotter.create_advanced_correlation_plot(
            reference_data, comparison_data, reference_name, title, output_path)
    
    # === BACKWARD COMPATIBILITY METHODS ===
    
    def create_histogram(self, data: Union[pd.Series, Dict[str, pd.Series]],
                        title: str = "Histogram",
                        output_path: Optional[str] = None) -> plt.Figure:
        """Create histogram (backward compatibility)."""
        if isinstance(data, dict):
            return self.distribution_plotter.create_refined_distribution_analysis(
                data, title, output_path)
        else:
            # Single series histogram
            return self.distribution_plotter.create_boxplot(
                {'Data': data}, title, 'Value', output_path)
    
    def generate_all_plots(self, aws_data: pd.Series,
                          modis_methods: Dict[str, pd.Series],
                          time_series_data: Optional[pd.DataFrame] = None,
                          glacier_name: str = "Glacier",
                          output_dir: str = "plots") -> Dict[str, plt.Figure]:
        """Generate all standard plots for a complete analysis."""
        logger.info("Generating complete plot suite")
        
        plots = {}
        
        try:
            # Core comparison plots
            plots['performance_metrics'] = self.create_four_metrics_boxplot_summary(
                aws_data, modis_methods, glacier_name, 
                f"{output_dir}/{glacier_name}_performance_metrics.png")
            
            plots['correlation_bias'] = self.create_original_correlation_bias_analysis(
                aws_data, modis_methods, glacier_name,
                f"{output_dir}/{glacier_name}_correlation_bias.png")
            
            plots['scatter_analysis'] = self.create_multi_method_scatter(
                aws_data, modis_methods, f"{glacier_name} Multi-Method Scatter Analysis",
                f"{output_dir}/{glacier_name}_scatter_analysis.png")
            
            # Distribution analysis
            all_data = {'AWS': aws_data}
            all_data.update(modis_methods)
            plots['distribution'] = self.create_refined_distribution_analysis(
                all_data, f"{glacier_name} Distribution Analysis",
                f"{output_dir}/{glacier_name}_distribution.png")
            
            # Correlation analysis
            plots['correlation_matrix'] = self.create_correlation_matrix(
                all_data, f"{glacier_name} Correlation Matrix",
                f"{output_dir}/{glacier_name}_correlation_matrix.png")
            
            # Comprehensive dashboard
            plots['comprehensive'] = self.create_comprehensive_analysis(
                aws_data, modis_methods, glacier_name,
                f"{output_dir}/{glacier_name}_comprehensive.png")
            
            # Time series if data provided
            if time_series_data is not None:
                plots['time_series'] = self.create_time_series_plot(
                    time_series_data, title=f"{glacier_name} Time Series Analysis",
                    output_path=f"{output_dir}/{glacier_name}_time_series.png")
                
                plots['seasonal'] = self.create_original_seasonal_analysis(
                    time_series_data, glacier_name=glacier_name,
                    output_path=f"{output_dir}/{glacier_name}_seasonal.png")
            
            logger.info(f"Generated {len(plots)} plots successfully")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
        
        return plots
    
    def close_all_figures(self):
        """Close all matplotlib figures to free memory."""
        plt.close('all')
        logger.info("All matplotlib figures closed")
    
    def get_plotter_info(self) -> Dict[str, Any]:
        """Get information about available plotters and methods."""
        return {
            'scatter_plotter': {
                'class': 'ScatterPlotter',
                'methods': ['create_scatter_plot', 'create_enhanced_scatter_plot', 'create_multi_method_scatter']
            },
            'distribution_plotter': {
                'class': 'DistributionPlotter',
                'methods': ['create_boxplot', 'create_difference_boxplot', 'create_refined_distribution_analysis']
            },
            'time_series_plotter': {
                'class': 'TimeSeriesPlotter',
                'methods': ['create_time_series_plot', 'create_seasonal_comparison', 'create_original_seasonal_analysis']
            },
            'comparison_plotter': {
                'class': 'ComparisonPlotter',
                'methods': ['create_four_metrics_boxplot_summary', 'create_refined_method_comparison', 'create_original_correlation_bias_analysis']
            },
            'comprehensive_plotter': {
                'class': 'ComprehensivePlotter',
                'methods': ['create_comprehensive_analysis', 'create_method_comparison_dashboard']
            },
            'correlation_plotter': {
                'class': 'CorrelationPlotter',
                'methods': ['create_correlation_matrix', 'create_scatter_matrix', 'create_pairwise_correlation_analysis', 'create_method_relationship_analysis', 'create_advanced_correlation_plot']
            }
        }