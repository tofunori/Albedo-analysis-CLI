#!/usr/bin/env python3
"""
Distribution plot implementations for albedo analysis.

This module contains box plots, histograms, and other distribution
visualization methods for comparing albedo datasets.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from .base import BasePlotter

logger = logging.getLogger(__name__)


class DistributionPlotter(BasePlotter):
    """Specialized plotter for distribution visualizations."""
    
    def create_boxplot(self, data: Dict[str, pd.Series],
                      title: str = "Albedo Distribution Comparison",
                      y_label: str = "Albedo",
                      output_path: Optional[str] = None) -> plt.Figure:
        """Create box plot comparing albedo distributions."""
        
        # Prepare data for boxplot
        plot_data = []
        labels = []
        
        for method_name, values in data.items():
            clean_values = values.dropna()
            if len(clean_values) > 0:
                plot_data.append(clean_values)
                labels.append(f"{method_name}\n(n={len(clean_values)})")
        
        if not plot_data:
            logger.error("No valid data for boxplot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create boxplot
        box_plot = ax.boxplot(plot_data, labels=labels, patch_artist=True,
                             boxprops=dict(facecolor='lightblue', alpha=0.7),
                             medianprops=dict(color='red', linewidth=2),
                             whiskerprops=dict(color='black'),
                             capprops=dict(color='black'))
        
        # Color boxes according to method
        for i, (method_name, _) in enumerate(data.items()):
            if i < len(box_plot['boxes']):
                color = self._get_method_color(method_name, i)
                box_plot['boxes'][i].set_facecolor(color)
                box_plot['boxes'][i].set_alpha(0.7)
        
        # Customize plot
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate x-axis labels if needed
        if max(len(label) for label in labels) > 10:
            plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Boxplot")
        
        return fig
    
    def create_difference_boxplot(self, reference_data: pd.Series,
                                comparison_data: Dict[str, pd.Series],
                                title: str = "Albedo Difference Distribution",
                                reference_name: str = "AWS",
                                output_path: Optional[str] = None) -> plt.Figure:
        """Create boxplot of differences between methods and reference."""
        
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
            logger.error("No valid data for difference boxplot")
            return None
        
        # Create boxplot
        fig = self.create_boxplot(
            differences,
            title=title,
            y_label=f"Albedo Difference (Method - {reference_name})",
            output_path=None  # Don't save yet, we need to modify
        )
        
        # Add zero line
        if fig is not None:
            ax = fig.get_axes()[0]
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1)
            
            # Now save with the modification
            self._save_figure(fig, output_path, "Difference boxplot")
        
        return fig
    
    def create_refined_distribution_analysis(self, data: Dict[str, pd.Series],
                                           title: str = "Distribution Analysis",
                                           output_path: Optional[str] = None) -> plt.Figure:
        """Create refined distribution analysis with multiple visualization types."""
        
        if not data:
            logger.error("No data provided for distribution analysis")
            return None
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(self.figure_size[0] * 1.5, self.figure_size[1] * 1.5))
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        method_names = list(data.keys())
        
        # 1. Box plot (top-left)
        ax = axes[0]
        plot_data = []
        labels = []
        
        for method_name, values in data.items():
            clean_values = values.dropna()
            if len(clean_values) > 0:
                plot_data.append(clean_values)
                labels.append(f"{method_name}\n(n={len(clean_values)})")
        
        if plot_data:
            box_plot = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            
            # Color boxes
            for i, (method_name, _) in enumerate(data.items()):
                if i < len(box_plot['boxes']):
                    color = self._get_method_color(method_name, i)
                    box_plot['boxes'][i].set_facecolor(color)
                    box_plot['boxes'][i].set_alpha(0.7)
        
        ax.set_title("Distribution Comparison")
        ax.set_ylabel("Albedo")
        ax.grid(True, alpha=0.3)
        
        # 2. Histogram overlay (top-right)
        ax = axes[1]
        for i, (method_name, values) in enumerate(data.items()):
            clean_values = values.dropna()
            if len(clean_values) > 0:
                color = self._get_method_color(method_name, i)
                ax.hist(clean_values, alpha=0.6, bins=30, label=method_name, 
                       color=color, density=True)
        
        ax.set_title("Probability Density")
        ax.set_xlabel("Albedo")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Violin plot (bottom-left)
        ax = axes[2]
        if plot_data:
            parts = ax.violinplot(plot_data, positions=range(1, len(plot_data) + 1))
            
            # Color violin plots
            for i, pc in enumerate(parts['bodies']):
                if i < len(method_names):
                    color = self._get_method_color(method_names[i], i)
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
            
            ax.set_xticks(range(1, len(labels) + 1))
            ax.set_xticklabels([label.split('\n')[0] for label in labels])
        
        ax.set_title("Distribution Shape")
        ax.set_ylabel("Albedo")
        ax.grid(True, alpha=0.3)
        
        # 4. Summary statistics table (bottom-right)
        ax = axes[3]
        ax.axis('off')
        
        # Create summary statistics
        stats_data = []
        for method_name, values in data.items():
            clean_values = values.dropna()
            if len(clean_values) > 0:
                stats_data.append([
                    method_name,
                    f"{clean_values.mean():.3f}",
                    f"{clean_values.std():.3f}",
                    f"{clean_values.median():.3f}",
                    f"{len(clean_values)}"
                ])
        
        if stats_data:
            table = ax.table(cellText=stats_data,
                           colLabels=['Method', 'Mean', 'Std', 'Median', 'Count'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
        
        ax.set_title("Summary Statistics")
        
        # Overall title
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Refined distribution analysis")
        
        return fig