#!/usr/bin/env python3
"""
Time series plot implementations for albedo analysis.

This module contains temporal analysis visualizations including
multi-panel yearly time series and seasonal comparison plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from .base import BasePlotter

logger = logging.getLogger(__name__)


class TimeSeriesPlotter(BasePlotter):
    """Specialized plotter for time series and temporal analysis."""
    
    def create_time_series_plot(self, data: pd.DataFrame,
                              date_column: str = 'date',
                              value_column: str = 'albedo',
                              method_column: str = 'method',
                              title: str = "Albedo Time Series",
                              output_path: Optional[str] = None) -> plt.Figure:
        """Create multi-panel time series plot with separate subplot for each year."""
        
        if any(col not in data.columns for col in [date_column, value_column, method_column]):
            logger.error("Required columns missing for time series plot")
            return None
        
        # Prepare data
        data_copy = data.copy()
        data_copy[date_column] = pd.to_datetime(data_copy[date_column])
        data_copy = data_copy.sort_values(date_column)
        data_copy['year'] = data_copy[date_column].dt.year
        data_copy['day_of_year'] = data_copy[date_column].dt.dayofyear
        
        # Get unique years and methods
        years = sorted(data_copy['year'].unique())
        methods = sorted(data_copy[method_column].unique())
        
        if len(years) == 0:
            logger.error("No years found in data")
            return None
        
        # Calculate grid layout
        n_years = len(years)
        if n_years <= 6:
            rows, cols = 2, 3
        elif n_years <= 12:
            rows, cols = 3, 4
        else:
            rows, cols = 4, 4  # Up to 16 years max
        
        # Create figure with subplots
        fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier indexing
        if n_years == 1:
            axes = [axes]
        elif rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        # Find global y-axis range for consistency
        y_min = data_copy[value_column].min()
        y_max = data_copy[value_column].max()
        y_range = y_max - y_min
        y_min_plot = max(0, y_min - 0.05 * y_range)
        y_max_plot = min(1, y_max + 0.05 * y_range)
        
        # Plot data for each year
        for i, year in enumerate(years):
            if i >= len(axes):
                break
                
            ax = axes[i]
            year_data = data_copy[data_copy['year'] == year]
            
            if year_data.empty:
                ax.text(0.5, 0.5, f'No {year} data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=10)
                ax.set_title(str(year), fontweight='bold')
                continue
            
            # Plot each method for this year
            for method in methods:
                method_data = year_data[year_data[method_column] == method]
                
                if not method_data.empty:
                    color = self._get_method_color(method, list(methods).index(method))
                    
                    # Create date within year for x-axis (using day of year)
                    ax.plot(method_data['day_of_year'], method_data[value_column],
                           label=method, color=color, alpha=0.8, linewidth=1.5, 
                           marker='o', markersize=2)
            
            # Customize each subplot
            ax.set_title(str(year), fontweight='bold', fontsize=12)
            ax.set_ylim(y_min_plot, y_max_plot)
            ax.grid(True, alpha=0.3)
            
            # Dynamic x-axis: only show months with actual data
            self._optimize_time_axis(ax, year_data, 'day_of_year')
            
            # Only show y-axis label on leftmost plots
            if i % cols == 0:
                ax.set_ylabel('Albedo', fontsize=10)
            
            # Only show x-axis label on bottom plots
            if i >= (rows - 1) * cols:
                ax.set_xlabel('Month', fontsize=10)
        
        # Hide unused subplots
        for j in range(len(years), len(axes)):
            axes[j].set_visible(False)
        
        # Add legend to the figure
        if methods:
            legend_handles = []
            for method in methods:
                color = self._get_method_color(method, list(methods).index(method))
                legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, 
                                               marker='o', markersize=4, label=method))
            
            fig.legend(legend_handles, methods, loc='upper right', bbox_to_anchor=(0.98, 0.95))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, right=0.92)
        
        # Save figure
        self._save_figure(fig, output_path, "Time series plot")
        
        return fig
    
    def _optimize_time_axis(self, ax, year_data, day_column):
        """Optimize x-axis to show only months with data."""
        if year_data.empty:
            ax.set_xlim(150, 270)  # Roughly Jun-Sep fallback
            return
        
        # Find actual data range for this year
        min_day = year_data[day_column].min()
        max_day = year_data[day_column].max()
        
        # Add small padding
        day_range = max_day - min_day
        padding = max(5, day_range * 0.05)  # 5% padding or minimum 5 days
        x_min = max(1, min_day - padding)
        x_max = min(365, max_day + padding)
        
        # Calculate which months are in the data range
        month_starts = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366]
        month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
        
        # Find months that intersect with data range
        visible_months = []
        visible_starts = []
        for j, (start, label) in enumerate(zip(month_starts[:-1], month_labels)):
            end = month_starts[j + 1]
            # Check if this month overlaps with data range
            if start <= x_max and end >= x_min:
                visible_months.append(label)
                visible_starts.append(start)
        
        # Set optimized x-axis
        ax.set_xlim(x_min, x_max)
        if visible_starts:
            # Only show ticks for months with data, but filter to reasonable spacing
            if len(visible_starts) <= 6:
                ax.set_xticks(visible_starts)
                ax.set_xticklabels(visible_months, fontsize=9)
            else:
                # Too many months, show every other one
                ax.set_xticks(visible_starts[::2])
                ax.set_xticklabels(visible_months[::2], fontsize=9)
    
    def create_seasonal_comparison(self, data: pd.DataFrame,
                                 date_column: str = 'date',
                                 value_column: str = 'albedo',
                                 method_column: str = 'method',
                                 title: str = "Seasonal Albedo Comparison",
                                 output_path: Optional[str] = None) -> plt.Figure:
        """Create seasonal comparison plot."""
        
        if any(col not in data.columns for col in [date_column, value_column, method_column]):
            logger.error("Required columns missing for seasonal comparison")
            return None
        
        # Add season column
        data_copy = data.copy()
        data_copy[date_column] = pd.to_datetime(data_copy[date_column])
        
        def get_season(month):
            if month in [12, 1, 2]:
                return 'Winter'
            elif month in [3, 4, 5]:
                return 'Spring'
            elif month in [6, 7, 8]:
                return 'Summer'
            else:
                return 'Autumn'
        
        data_copy['season'] = data_copy[date_column].dt.month.apply(get_season)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(self.figure_size[0] * 1.5, self.figure_size[1] * 1.5))
        axes = axes.flatten()
        
        seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        
        for i, season in enumerate(seasons):
            season_data = data_copy[data_copy['season'] == season]
            
            if season_data.empty:
                axes[i].text(0.5, 0.5, f'No {season} data', transform=axes[i].transAxes,
                           ha='center', va='center')
                axes[i].set_title(season)
                continue
            
            # Create boxplot for this season
            methods = season_data[method_column].unique()
            season_plot_data = []
            season_labels = []
            
            for method in methods:
                method_data = season_data[season_data[method_column] == method][value_column].dropna()
                if len(method_data) > 0:
                    season_plot_data.append(method_data)
                    season_labels.append(f"{method}\n(n={len(method_data)})")
            
            if season_plot_data:
                box_plot = axes[i].boxplot(season_plot_data, labels=season_labels, patch_artist=True)
                
                # Color boxes
                for j, method in enumerate(methods):
                    if j < len(box_plot['boxes']):
                        color = self._get_method_color(method, j)
                        box_plot['boxes'][j].set_facecolor(color)
                        box_plot['boxes'][j].set_alpha(0.7)
            
            axes[i].set_title(season)
            axes[i].set_ylabel('Albedo' if i % 2 == 0 else '')
            axes[i].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Seasonal comparison plot")
        
        return fig
    
    def create_original_seasonal_analysis(self, data: pd.DataFrame,
                                        date_column: str = 'date',
                                        value_column: str = 'albedo',
                                        method_column: str = 'method',
                                        glacier_name: str = "Glacier",
                                        output_path: Optional[str] = None) -> plt.Figure:
        """Create original 4-panel seasonal analysis (monthly boxplots)."""
        logger.info("Creating original seasonal analysis (4-panel monthly)")
        
        if any(col not in data.columns for col in [date_column, value_column, method_column]):
            logger.error("Required columns missing for seasonal analysis")
            return None
        
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{glacier_name.upper()} Glacier - Seasonal Analysis', 
                    fontsize=16, fontweight='bold')
        
        # Add month column
        data_copy = data.copy()
        data_copy[date_column] = pd.to_datetime(data_copy[date_column])
        data_copy['month'] = data_copy[date_column].dt.month
        
        # Define months to analyze (typically summer months for glaciers)
        target_months = [6, 7, 8, 9]  # Jun, Jul, Aug, Sep
        month_names = ['June', 'July', 'August', 'September']
        
        methods = data_copy[method_column].unique()
        
        for i, (month, month_name) in enumerate(zip(target_months, month_names)):
            ax = axes[i // 2, i % 2]
            
            month_data = data_copy[data_copy['month'] == month]
            
            if month_data.empty:
                ax.text(0.5, 0.5, f'No {month_name} data', transform=ax.transAxes,
                       ha='center', va='center', fontsize=12)
                ax.set_title(month_name)
                continue
            
            # Prepare boxplot data for this month
            month_plot_data = []
            month_labels = []
            
            for method in methods:
                method_data = month_data[month_data[method_column] == method][value_column].dropna()
                if len(method_data) > 0:
                    month_plot_data.append(method_data)
                    month_labels.append(f"{method}\n(n={len(method_data)})")
            
            if month_plot_data:
                box_plot = ax.boxplot(month_plot_data, labels=month_labels, patch_artist=True)
                
                # Color boxes by method
                for j, method in enumerate(methods):
                    if j < len(box_plot['boxes']):
                        color = self._get_method_color(method, j)
                        box_plot['boxes'][j].set_facecolor(color)
                        box_plot['boxes'][j].set_alpha(0.7)
            
            ax.set_title(month_name, fontweight='bold')
            ax.set_ylabel('Albedo')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Original seasonal analysis")
        
        return fig