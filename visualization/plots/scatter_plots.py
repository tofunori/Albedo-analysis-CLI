#!/usr/bin/env python3
"""
Scatter plot implementations for albedo analysis.

This module contains all scatter plot related functionality including
single method scatterplots, multi-method comparisons, and difference plots.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from .base import BasePlotter

logger = logging.getLogger(__name__)


class ScatterPlotter(BasePlotter):
    """Specialized plotter for scatter plot visualizations."""
    
    def create_scatterplot(self, x_data: pd.Series, y_data: pd.Series,
                          x_label: str = "AWS Albedo", y_label: str = "MODIS Albedo",
                          title: str = "MODIS vs AWS Albedo Comparison",
                          method_name: str = "MODIS",
                          show_stats: bool = True,
                          add_regression_line: bool = True,
                          add_identity_line: bool = True,
                          output_path: Optional[str] = None) -> plt.Figure:
        """Create scatterplot comparing two albedo datasets."""
        
        # Clean and align data
        x_clean, y_clean = self._clean_and_align_data(x_data, y_data)
        
        if len(x_clean) == 0:
            logger.error("No valid data points for scatterplot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Get color for method
        color = self._get_method_color(method_name)
        
        # Create scatter plot
        scatter = ax.scatter(x_clean, y_clean, alpha=0.6, s=50, 
                           c=color, edgecolors='black', linewidth=0.5,
                           label=f'{method_name} (n={len(x_clean)})')
        
        # Add identity line (1:1)
        if add_identity_line:
            self._add_identity_line(ax, x_clean, y_clean, label='1:1 line')
        
        # Add regression line
        if add_regression_line and len(x_clean) > 1:
            self._add_regression_line(ax, x_clean, y_clean)
        
        # Add statistics text box
        if show_stats:
            self._add_statistics_text(ax, x_clean, y_clean)
        
        # Customize plot
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio and limits
        self._setup_equal_aspect_plot(ax, x_clean, y_clean)
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Scatterplot")
        
        return fig
    
    def create_multi_method_scatterplot(self, aws_data: pd.Series, 
                                      modis_methods: Dict[str, pd.Series],
                                      title: str = "Multi-Method MODIS vs AWS Comparison",
                                      output_path: Optional[str] = None) -> plt.Figure:
        """Create scatterplot comparing multiple MODIS methods against AWS."""
        
        n_methods = len(modis_methods)
        if n_methods == 0:
            logger.error("No MODIS methods provided")
            return None
        
        fig, axes = plt.subplots(1, n_methods, 
                                figsize=(self.figure_size[0] * n_methods, self.figure_size[1]))
        
        if n_methods == 1:
            axes = [axes]
        
        for i, (method_name, modis_data) in enumerate(modis_methods.items()):
            self._create_single_scatterplot_axis(
                axes[i], aws_data, modis_data, method_name,
                f"AWS vs {method_name}", "AWS Albedo", f"{method_name} Albedo"
            )
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Multi-method scatterplot")
        
        return fig
    
    def create_difference_scatterplot(self, reference_data: pd.Series,
                                    comparison_data: Dict[str, pd.Series],
                                    title: str = "Albedo Difference Scatter Plot",
                                    reference_name: str = "AWS",
                                    output_path: Optional[str] = None) -> plt.Figure:
        """Create scatterplot of differences between methods and reference."""
        
        # Calculate differences
        differences = {}
        for method_name, method_data in comparison_data.items():
            # Align data
            common_idx = reference_data.index.intersection(method_data.index)
            if len(common_idx) > 0:
                ref_aligned = reference_data.loc[common_idx]
                method_aligned = method_data.loc[common_idx]
                
                # Calculate difference (method - reference)
                diff = method_aligned - ref_aligned
                differences[method_name] = diff.dropna()
        
        if not differences:
            logger.error("No valid data for difference scatterplot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Plot scatter points for each method
        x_positions = []
        y_values = []
        labels = []
        colors = []
        
        for i, (method_name, diff_values) in enumerate(differences.items()):
            if len(diff_values) > 0:
                # Add jitter to x positions for better visualization
                jitter = np.random.normal(0, 0.1, len(diff_values))
                x_pos = np.full(len(diff_values), i) + jitter
                
                x_positions.extend(x_pos)
                y_values.extend(diff_values)
                labels.extend([method_name] * len(diff_values))
                colors.extend([self._get_method_color(method_name, i)] * len(diff_values))
        
        if x_positions:
            # Create scatter plot
            ax.scatter(x_positions, y_values, c=colors, alpha=0.6, s=30)
            
            # Add box plots for comparison
            box_data = [diff_values for diff_values in differences.values() if len(diff_values) > 0]
            box_labels = [method_name for method_name in differences.keys() if len(differences[method_name]) > 0]
            
            box_plot = ax.boxplot(box_data, labels=box_labels, patch_artist=True, 
                                positions=range(len(box_data)), widths=0.3)
            
            # Color boxes
            for i, (method_name, box) in enumerate(zip(box_labels, box_plot['boxes'])):
                color = self._get_method_color(method_name, i)
                box.set_facecolor(color)
                box.set_alpha(0.3)
        
        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1)
        
        # Customize plot
        ax.set_ylabel(f"Albedo Difference (Method - {reference_name})", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Difference scatterplot")
        
        return fig