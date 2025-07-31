#!/usr/bin/env python3
"""
Correlation plot implementations for albedo analysis.

This module contains correlation matrices, scatter plots, and 
relationship analysis visualizations for comparing albedo datasets.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from .base import BasePlotter

logger = logging.getLogger(__name__)


class CorrelationPlotter(BasePlotter):
    """Specialized plotter for correlation and relationship analysis."""
    
    def create_correlation_matrix(self, data: Dict[str, pd.Series],
                                title: str = "Albedo Correlation Matrix",
                                output_path: Optional[str] = None) -> plt.Figure:
        """Create correlation matrix heatmap."""
        logger.info("Creating correlation matrix heatmap")
        
        if not data or len(data) < 2:
            logger.error("Need at least 2 datasets for correlation matrix")
            return None
        
        try:
            # Create correlation matrix
            corr_df = pd.DataFrame(data).corr()
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.figure_size)
            
            # Create heatmap
            im = ax.imshow(corr_df.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Add correlation values as text
            for i in range(len(corr_df)):
                for j in range(len(corr_df.columns)):
                    corr_val = corr_df.iloc[i, j]
                    text_color = 'white' if abs(corr_val) > 0.5 else 'black'
                    ax.text(j, i, f'{corr_val:.3f}',
                           ha="center", va="center", color=text_color, 
                           fontsize=12, fontweight='bold')
            
            # Customize axes
            ax.set_xticks(range(len(corr_df.columns)))
            ax.set_yticks(range(len(corr_df)))
            ax.set_xticklabels(corr_df.columns, fontsize=11)
            ax.set_yticklabels(corr_df.index, fontsize=11)
            
            # Rotate x-axis labels for better readability
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
            
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Correlation Coefficient', fontsize=12)
            
            plt.tight_layout()
            
            # Save figure
            self._save_figure(fig, output_path, "Correlation matrix")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating correlation matrix: {e}")
            return None
    
    def create_scatter_matrix(self, data: Dict[str, pd.Series],
                            title: str = "Albedo Scatter Matrix",
                            output_path: Optional[str] = None) -> plt.Figure:
        """Create scatter plot matrix for multiple datasets."""
        logger.info("Creating scatter plot matrix")
        
        if not data or len(data) < 2:
            logger.error("Need at least 2 datasets for scatter matrix")
            return None
        
        methods = list(data.keys())
        n_methods = len(methods)
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_methods, n_methods, 
                               figsize=(3 * n_methods, 3 * n_methods))
        
        if n_methods == 1:
            axes = [[axes]]
        elif n_methods == 2:
            axes = axes.reshape(2, 2)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        for i, method_x in enumerate(methods):
            for j, method_y in enumerate(methods):
                ax = axes[i, j]
                
                if i == j:
                    # Diagonal: histogram
                    clean_data = data[method_x].dropna()
                    if len(clean_data) > 0:
                        color = self._get_method_color(method_x, i)
                        ax.hist(clean_data, bins=20, alpha=0.7, color=color, density=True)
                        ax.set_ylabel('Density')
                        ax.set_title(f'{method_x} Distribution')
                    else:
                        ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, 
                               ha='center', va='center')
                else:
                    # Off-diagonal: scatter plot
                    x_data = data[method_x]
                    y_data = data[method_y]
                    
                    # Align data
                    common_idx = x_data.index.intersection(y_data.index)
                    if len(common_idx) > 0:
                        x_aligned = x_data.loc[common_idx].dropna()
                        y_aligned = y_data.loc[common_idx].dropna()
                        final_common = x_aligned.index.intersection(y_aligned.index)
                        
                        if len(final_common) > 0:
                            x_clean = x_aligned.loc[final_common]
                            y_clean = y_aligned.loc[final_common]
                            
                            # Scatter plot
                            ax.scatter(x_clean, y_clean, alpha=0.6, s=10)
                            
                            # Add 1:1 line
                            min_val = min(x_clean.min(), y_clean.min())
                            max_val = max(x_clean.max(), y_clean.max())
                            ax.plot([min_val, max_val], [min_val, max_val], 
                                   'r--', alpha=0.8, linewidth=1)
                            
                            # Calculate and display correlation
                            r = np.corrcoef(x_clean, y_clean)[0, 1]
                            ax.text(0.05, 0.95, f'R = {r:.3f}', 
                                   transform=ax.transAxes,
                                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                        else:
                            ax.text(0.5, 0.5, 'No common data', 
                                   transform=ax.transAxes, ha='center', va='center')
                    else:
                        ax.text(0.5, 0.5, 'No common data', 
                               transform=ax.transAxes, ha='center', va='center')
                
                # Set labels
                if i == n_methods - 1:  # Bottom row
                    ax.set_xlabel(f'{method_x} Albedo')
                if j == 0:  # Left column
                    ax.set_ylabel(f'{method_y} Albedo')
                
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Scatter matrix")
        
        return fig
    
    def create_pairwise_correlation_analysis(self, reference_data: pd.Series,
                                           comparison_data: Dict[str, pd.Series],
                                           reference_name: str = "AWS",
                                           title: str = "Pairwise Correlation Analysis",
                                           output_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive pairwise correlation analysis."""
        logger.info("Creating pairwise correlation analysis")
        
        if not comparison_data:
            logger.error("No comparison data provided")
            return None
        
        n_methods = len(comparison_data)
        
        # Calculate grid layout
        if n_methods <= 2:
            rows, cols = 1, n_methods
        elif n_methods <= 4:
            rows, cols = 2, 2
        elif n_methods <= 6:
            rows, cols = 2, 3
        else:
            rows, cols = 3, 3
        
        # Create figure
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Flatten axes array for easier indexing
        if n_methods == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        for i, (method_name, method_data) in enumerate(comparison_data.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Align data
            aws_clean, method_clean = self._clean_and_align_data(reference_data, method_data)
            
            if len(aws_clean) > 0:
                # Scatter plot
                color = self._get_method_color(method_name, i)
                ax.scatter(aws_clean, method_clean, alpha=0.6, color=color, s=20)
                
                # Add 1:1 line
                min_val = min(aws_clean.min(), method_clean.min())
                max_val = max(aws_clean.max(), method_clean.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
                
                # Calculate statistics
                r = np.corrcoef(aws_clean, method_clean)[0, 1]
                rmse = np.sqrt(np.mean((method_clean - aws_clean) ** 2))
                bias = np.mean(method_clean - aws_clean)
                n_samples = len(aws_clean)
                
                # Add statistics text
                stats_text = f'R = {r:.3f}\nRMSE = {rmse:.3f}\nBias = {bias:.3f}\nn = {n_samples}'
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                # Add trend line
                try:
                    z = np.polyfit(aws_clean, method_clean, 1)
                    p = np.poly1d(z)
                    ax.plot(aws_clean, p(aws_clean), color='blue', linestyle='-', 
                           alpha=0.8, linewidth=1, label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
                    ax.legend(loc='lower right', fontsize=8)
                except:
                    pass
                
                ax.set_xlabel(f'{reference_name} Albedo')
                ax.set_ylabel(f'{method_name} Albedo')
                ax.set_title(f'{reference_name} vs {method_name}')
            else:
                ax.text(0.5, 0.5, f'No data for {method_name}', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{reference_name} vs {method_name}')
            
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for j in range(n_methods, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Pairwise correlation analysis")
        
        return fig
    
    def create_method_relationship_analysis(self, data: Dict[str, pd.Series],
                                          title: str = "Method Relationship Analysis",
                                          output_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive method relationship analysis."""
        logger.info("Creating method relationship analysis")
        
        if not data or len(data) < 2:
            logger.error("Need at least 2 datasets for relationship analysis")
            return None
        
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Panel 1: Correlation matrix (top-left)
        ax1 = axes[0, 0]
        try:
            corr_df = pd.DataFrame(data).corr()
            im = ax1.imshow(corr_df.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            for i in range(len(corr_df)):
                for j in range(len(corr_df.columns)):
                    corr_val = corr_df.iloc[i, j]
                    text_color = 'white' if abs(corr_val) > 0.5 else 'black'
                    ax1.text(j, i, f'{corr_val:.3f}',
                            ha="center", va="center", color=text_color, 
                            fontsize=10, fontweight='bold')
            
            ax1.set_xticks(range(len(corr_df.columns)))
            ax1.set_yticks(range(len(corr_df)))
            ax1.set_xticklabels(corr_df.columns, rotation=45, ha='right')
            ax1.set_yticklabels(corr_df.index)
            ax1.set_title('Correlation Matrix')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
            cbar.set_label('Correlation')
            
        except Exception as e:
            logger.warning(f"Could not create correlation matrix: {e}")
            ax1.text(0.5, 0.5, 'Correlation matrix unavailable', 
                    transform=ax1.transAxes, ha='center', va='center')
            ax1.set_title('Correlation Matrix')
        
        # Panel 2: Method comparison scatter (top-right)
        ax2 = axes[0, 1]
        methods = list(data.keys())
        if len(methods) >= 2:
            method1, method2 = methods[0], methods[1]
            data1, data2 = data[method1], data[method2]
            
            # Align data
            common_idx = data1.index.intersection(data2.index)
            if len(common_idx) > 0:
                aligned1 = data1.loc[common_idx].dropna()
                aligned2 = data2.loc[common_idx].dropna()
                final_common = aligned1.index.intersection(aligned2.index)
                
                if len(final_common) > 0:
                    clean1 = aligned1.loc[final_common]
                    clean2 = aligned2.loc[final_common]
                    
                    ax2.scatter(clean1, clean2, alpha=0.6, s=20)
                    
                    # 1:1 line
                    min_val = min(clean1.min(), clean2.min())
                    max_val = max(clean1.max(), clean2.max())
                    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
                    
                    r = np.corrcoef(clean1, clean2)[0, 1]
                    ax2.text(0.05, 0.95, f'R = {r:.3f}', transform=ax2.transAxes,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                    
                    ax2.set_xlabel(f'{method1} Albedo')
                    ax2.set_ylabel(f'{method2} Albedo')
        
        ax2.set_title(f'{methods[0]} vs {methods[1] if len(methods) > 1 else "N/A"}')
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Distribution comparison (bottom-left)
        ax3 = axes[1, 0]
        for i, (method_name, values) in enumerate(data.items()):
            clean_values = values.dropna()
            if len(clean_values) > 0:
                color = self._get_method_color(method_name, i)
                ax3.hist(clean_values, alpha=0.6, bins=20, label=method_name, 
                        color=color, density=True)
        
        ax3.set_xlabel('Albedo')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Summary statistics (bottom-right)
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Create summary table
        stats_data = []
        for method_name, values in data.items():
            clean_values = values.dropna()
            if len(clean_values) > 0:
                stats_data.append([
                    method_name,
                    f"{clean_values.mean():.4f}",
                    f"{clean_values.std():.4f}",
                    f"{clean_values.median():.4f}",
                    f"{len(clean_values)}"
                ])
        
        if stats_data:
            table = ax4.table(cellText=stats_data,
                             colLabels=['Method', 'Mean', 'Std', 'Median', 'Count'],
                             cellLoc='center',
                             loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
        
        ax4.set_title('Summary Statistics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Method relationship analysis")
        
        return fig
    
    def create_advanced_correlation_plot(self, reference_data: pd.Series,
                                       comparison_data: Dict[str, pd.Series],
                                       reference_name: str = "AWS",
                                       title: str = "Advanced Correlation Analysis",
                                       output_path: Optional[str] = None) -> plt.Figure:
        """Create advanced correlation plot with confidence intervals and regression."""
        logger.info("Creating advanced correlation plot")
        
        if not comparison_data:
            logger.error("No comparison data provided")
            return None
        
        n_methods = len(comparison_data)
        
        # Calculate layout
        if n_methods <= 2:
            rows, cols = 1, n_methods
        elif n_methods <= 4:
            rows, cols = 2, 2
        else:
            rows, cols = 2, 3
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        if n_methods == 1:
            axes = [axes]
        elif rows == 1 or cols == 1:
            axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        else:
            axes = axes.flatten()
        
        for i, (method_name, method_data) in enumerate(comparison_data.items()):
            if i >= len(axes):
                break
            
            ax = axes[i]
            
            # Align data
            aws_clean, method_clean = self._clean_and_align_data(reference_data, method_data)
            
            if len(aws_clean) > 10:  # Need sufficient data for advanced analysis
                color = self._get_method_color(method_name, i)
                
                # Scatter plot with density coloring
                ax.scatter(aws_clean, method_clean, alpha=0.6, color=color, s=15)
                
                # Add 1:1 line
                min_val = min(aws_clean.min(), method_clean.min())
                max_val = max(aws_clean.max(), method_clean.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', 
                       alpha=0.8, linewidth=2, label='1:1 line')
                
                # Regression line with confidence interval
                try:
                    from scipy import stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(aws_clean, method_clean)
                    
                    # Create regression line
                    x_reg = np.linspace(aws_clean.min(), aws_clean.max(), 100)
                    y_reg = slope * x_reg + intercept
                    ax.plot(x_reg, y_reg, 'b-', alpha=0.8, linewidth=2, 
                           label=f'Regression (R²={r_value**2:.3f})')
                    
                    # Add confidence interval (approximate)
                    # This is a simplified version - for publication quality, use proper methods
                    y_err = std_err * x_reg
                    ax.fill_between(x_reg, y_reg - 1.96*y_err, y_reg + 1.96*y_err, 
                                   alpha=0.2, color='blue', label='95% CI')
                    
                except ImportError:
                    logger.warning("scipy not available for regression analysis")
                    # Fallback to simple linear fit
                    z = np.polyfit(aws_clean, method_clean, 1)
                    p = np.poly1d(z)
                    x_reg = np.linspace(aws_clean.min(), aws_clean.max(), 100)
                    ax.plot(x_reg, p(x_reg), 'b-', alpha=0.8, linewidth=2, 
                           label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
                
                # Statistics
                r = np.corrcoef(aws_clean, method_clean)[0, 1]
                rmse = np.sqrt(np.mean((method_clean - aws_clean) ** 2))
                bias = np.mean(method_clean - aws_clean)
                mae = np.mean(np.abs(method_clean - aws_clean))
                n_samples = len(aws_clean)
                
                # Comprehensive statistics text
                stats_text = (f'R = {r:.3f}\n'
                             f'R² = {r**2:.3f}\n'
                             f'RMSE = {rmse:.4f}\n'
                             f'Bias = {bias:.4f}\n'
                             f'MAE = {mae:.4f}\n'
                             f'n = {n_samples}')
                
                ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', fontsize=9,
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))
                
            else:
                ax.text(0.5, 0.5, f'Insufficient data for {method_name}\n(n={len(aws_clean)})', 
                       transform=ax.transAxes, ha='center', va='center')
            
            ax.set_xlabel(f'{reference_name} Albedo')
            ax.set_ylabel(f'{method_name} Albedo')
            ax.set_title(f'{method_name} Advanced Correlation')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='lower right', fontsize=8)
        
        # Hide unused subplots
        for j in range(n_methods, len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Advanced correlation plot")
        
        return fig