#!/usr/bin/env python3
"""
Base plotting functionality for albedo analysis visualizations.

This module contains the BasePlotter class with shared configuration,
styling, and helper methods used across all specialized plotters.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from scipy import stats
import warnings

logger = logging.getLogger(__name__)

# Set consistent plotting style
plt.style.use('default')
sns.set_palette("husl")


class BasePlotter:
    """Base class for all specialized plotters with shared functionality."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base plotter with configuration."""
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.plot_config = self.viz_config.get('plot_output', {})
        
        # Standard color scheme for methods
        self.colors = self.viz_config.get('colors', {
            'MOD09GA': '#1f77b4',
            'MOD10A1': '#ff7f0e', 
            'MCD43A3': '#2ca02c',
            'AWS': '#d62728'
        })
        
        # Figure configuration
        self.figure_size = self.viz_config.get('figure_size', [10, 8])
        self.dpi = self.viz_config.get('dpi', 300)
        
        # Plot generation flags
        self.eliminate_redundancy = self.plot_config.get('eliminate_redundancy', True)
        self.generate_individual = self.plot_config.get('individual_plots', True)
        self.generate_dashboard = self.plot_config.get('dashboard_plot', True)
        
        # Track generated plots to avoid redundancy
        self._generated_boxplots = set()
        
        # Set plotting style
        style = self.viz_config.get('style', 'seaborn-v0_8')
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")
    
    def _create_single_scatterplot_axis(self, ax, x_data, y_data, method_name, 
                                      title, x_label, y_label):
        """Helper function to create scatterplot on given axis."""
        # Align data on common indices and remove NaN values
        if hasattr(x_data, 'index') and hasattr(y_data, 'index'):
            # Both are pandas Series - align on common index
            aligned_data = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
            x_clean = aligned_data['x']
            y_clean = aligned_data['y']
        else:
            # Handle arrays or other data types
            x_arr = np.array(x_data)
            y_arr = np.array(y_data)
            
            # Ensure same length
            min_len = min(len(x_arr), len(y_arr))
            x_arr = x_arr[:min_len]
            y_arr = y_arr[:min_len]
            
            # Remove NaN pairs
            mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
            x_clean = x_arr[mask]
            y_clean = y_arr[mask]
        
        if len(x_clean) == 0:
            ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
            return
        
        # Get color
        color = self.colors.get(method_name, '#1f77b4')
        
        # Scatter plot
        ax.scatter(x_clean, y_clean, alpha=0.6, s=50, c=color, 
                  edgecolors='black', linewidth=0.5)
        
        # 1:1 line
        min_val = min(x_clean.min(), y_clean.min())
        max_val = max(x_clean.max(), y_clean.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, linewidth=1)
        
        # Statistics
        metrics = self._calculate_basic_metrics(x_clean, y_clean)
        
        stats_text = f'RMSE: {metrics["rmse"]:.3f}\nR²: {metrics["r2"]:.3f}\nn: {metrics["n_samples"]}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
        
        # Formatting
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    def _calculate_basic_metrics(self, observed, predicted):
        """Calculate basic statistical metrics between observed and predicted values."""
        if len(observed) == 0:
            return {
                'rmse': np.nan, 'bias': np.nan, 'r2': np.nan, 
                'correlation': np.nan, 'n_samples': 0
            }
        
        rmse = np.sqrt(np.mean((predicted - observed) ** 2))
        bias = np.mean(predicted - observed)
        
        if len(observed) > 1:
            correlation_matrix = np.corrcoef(observed, predicted)
            r_value = correlation_matrix[0, 1] if not np.isnan(correlation_matrix).any() else np.nan
            r_squared = r_value ** 2 if not np.isnan(r_value) else np.nan
        else:
            r_value = np.nan
            r_squared = np.nan
        
        return {
            'rmse': rmse,
            'bias': bias,
            'r2': r_squared,
            'correlation': r_value,
            'n_samples': len(observed)
        }
    
    def _apply_publication_style(self):
        """Apply publication-quality matplotlib parameters."""
        return {
            'font.family': 'Arial',
            'font.size': 11,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 1.5
        }
    
    def _clean_and_align_data(self, x_data: pd.Series, y_data: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """Clean and align two pandas Series on common indices."""
        # Remove NaN values and align on common indices
        mask = ~(x_data.isna() | y_data.isna())
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        # Ensure same indices
        common_idx = x_clean.index.intersection(y_clean.index)
        if len(common_idx) > 0:
            return x_clean.loc[common_idx], y_clean.loc[common_idx]
        else:
            return pd.Series([], dtype=float), pd.Series([], dtype=float)
    
    def _setup_equal_aspect_plot(self, ax, x_data, y_data, margin_factor=0.05):
        """Setup equal aspect ratio plot with appropriate margins."""
        ax.set_aspect('equal', adjustable='box')
        
        if len(x_data) > 0 and len(y_data) > 0:
            all_values = np.concatenate([x_data, y_data])
            margin = (all_values.max() - all_values.min()) * margin_factor
            ax.set_xlim(all_values.min() - margin, all_values.max() + margin)
            ax.set_ylim(all_values.min() - margin, all_values.max() + margin)
    
    def _add_identity_line(self, ax, x_data, y_data, **kwargs):
        """Add 1:1 identity line to plot."""
        if len(x_data) > 0 and len(y_data) > 0:
            min_val = min(x_data.min(), y_data.min())
            max_val = max(x_data.max(), y_data.max())
            
            default_kwargs = {'color': 'black', 'linestyle': '--', 'alpha': 0.8, 'linewidth': 2}
            default_kwargs.update(kwargs)
            
            ax.plot([min_val, max_val], [min_val, max_val], **default_kwargs)
    
    def _add_regression_line(self, ax, x_data, y_data, **kwargs):
        """Add regression line with R² in legend."""
        if len(x_data) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
            line_x = np.array([x_data.min(), x_data.max()])
            line_y = slope * line_x + intercept
            
            default_kwargs = {'color': 'red', 'alpha': 0.8, 'linewidth': 2}
            default_kwargs.update(kwargs)
            
            label = kwargs.pop('label', f'Regression (R²={r_value**2:.3f})')
            ax.plot(line_x, line_y, label=label, **default_kwargs)
    
    def _add_statistics_text(self, ax, x_data, y_data, position=(0.05, 0.95)):
        """Add statistics text box to plot."""
        if len(x_data) == 0 or len(y_data) == 0:
            return
        
        metrics = self._calculate_basic_metrics(x_data, y_data)
        
        stats_text = f'RMSE: {metrics["rmse"]:.3f}\n'
        stats_text += f'Bias: {metrics["bias"]:.3f}\n'
        stats_text += f'R²: {metrics["r2"]:.3f}\n'
        stats_text += f'n: {metrics["n_samples"]}'
        
        ax.text(position[0], position[1], stats_text, transform=ax.transAxes,
               verticalalignment='top', bbox=dict(boxstyle='round',
               facecolor='white', alpha=0.8), fontsize=10)
    
    def _save_figure(self, fig, output_path: Optional[str], plot_type: str):
        """Save figure with appropriate settings and logging."""
        if output_path:
            fig.savefig(output_path, dpi=self.dpi, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            logger.info(f"{plot_type} saved to {output_path}")
    
    def _get_method_color(self, method_name: str, index: int = 0) -> str:
        """Get color for method, with fallback to indexed colors."""
        return self.colors.get(method_name, f'C{index}')