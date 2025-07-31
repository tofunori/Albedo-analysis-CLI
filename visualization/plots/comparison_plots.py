#!/usr/bin/env python3
"""
Comparison plot implementations for albedo analysis.

This module contains method performance comparison plots including
the publication-ready grouped bar chart and correlation/bias analysis.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from .base import BasePlotter

logger = logging.getLogger(__name__)


class ComparisonPlotter(BasePlotter):
    """Specialized plotter for method performance comparisons."""
    
    def create_four_metrics_boxplot_summary(self, aws_data: pd.Series, 
                                           modis_methods: Dict[str, pd.Series],
                                           glacier_name: str = "Glacier",
                                           output_path: Optional[str] = None) -> plt.Figure:
        """Create publication-ready single grouped bar chart showing performance metrics for all methods."""
        logger.info("Creating publication-ready grouped bar chart for method performance")
        
        # Set publication-quality matplotlib parameters
        pub_style = self._apply_publication_style()
        plt.rcParams.update(pub_style)
        
        methods = list(modis_methods.keys())
        
        # Collect all metrics for each method
        method_data = {}
        
        for method in methods:
            modis_data = modis_methods[method]
            common_idx = aws_data.index.intersection(modis_data.index)
            
            if len(common_idx) > 0:
                aws_aligned = aws_data.loc[common_idx].dropna()
                modis_aligned = modis_data.loc[common_idx].dropna()
                final_common = aws_aligned.index.intersection(modis_aligned.index)
                
                if len(final_common) > 0:
                    aws_clean = aws_aligned.loc[final_common]
                    modis_clean = modis_aligned.loc[final_common]
                    
                    # Calculate all metrics
                    r = np.corrcoef(modis_clean, aws_clean)[0, 1] if len(aws_clean) > 1 else np.nan
                    bias = np.mean(modis_clean - aws_clean)
                    mae = np.mean(np.abs(modis_clean - aws_clean))
                    rmse = np.sqrt(np.mean((modis_clean - aws_clean) ** 2))
                    n_samples = len(aws_clean)
                    
                    method_data[method] = {
                        'correlation': r,
                        'bias': bias,
                        'mae': mae,
                        'rmse': rmse,
                        'n_samples': n_samples
                    }
        
        if not method_data:
            logger.warning("No valid data for metrics chart")
            return None
        
        # Create publication-ready single chart
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Define metrics and their properties
        metrics = ['correlation', 'bias', 'mae', 'rmse']
        metric_labels = ['Correlation (R)', 'Bias', 'MAE', 'RMSE']
        metric_colors = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e']  # Professional colors
        
        # Prepare data for grouped bar chart
        method_names = list(method_data.keys())
        n_methods = len(method_names)
        n_metrics = len(metrics)
        
        # Set up bar positioning
        bar_width = 0.2
        x_positions = np.arange(n_methods)
        
        # Create bars for each metric
        for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, metric_colors)):
            values = [method_data[method][metric] for method in method_names]
            positions = x_positions + i * bar_width
            
            bars = ax.bar(positions, values, bar_width, 
                         label=label, color=color, alpha=0.8, 
                         edgecolor='black', linewidth=0.8)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                # Position label above or below bar depending on sign
                if height >= 0:
                    va = 'bottom'
                    y_pos = height + abs(height) * 0.02
                else:
                    va = 'top'
                    y_pos = height - abs(height) * 0.02
                
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{value:.3f}', ha='center', va=va, 
                       fontsize=10, fontweight='bold')
        
        # Add horizontal line at y=0 for reference
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
        
        # Customize the chart
        ax.set_xlabel('MODIS Albedo Retrieval Methods', fontsize=13, fontweight='bold')
        ax.set_ylabel('Performance Metrics', fontsize=13, fontweight='bold')
        ax.set_title(f'{glacier_name.upper()} Glacier - Method Performance Comparison', 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Set x-axis
        ax.set_xticks(x_positions + bar_width * (n_metrics - 1) / 2)
        
        # Add sample sizes to method labels
        method_labels_with_n = []
        for method in method_names:
            n = method_data[method]['n_samples']
            method_labels_with_n.append(f'{method}\n(n={n})')
        
        ax.set_xticklabels(method_labels_with_n, fontsize=11)
        
        # Professional legend
        ax.legend(loc='upper left', frameon=True, fancybox=True, 
                 shadow=True, ncol=2, fontsize=11)
        
        # Grid for better readability
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Professional styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.2)
        ax.spines['bottom'].set_linewidth(1.2)
        
        # Optimize y-axis limits for better visualization
        all_values = []
        for method in method_data.values():
            all_values.extend([method['correlation'], method['bias'], method['mae'], method['rmse']])
        
        y_min = min(all_values)
        y_max = max(all_values)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.15 * y_range)
        
        plt.tight_layout()
        
        # Save with publication quality
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Publication-ready grouped bar chart saved to {output_path}")
        
        # Reset matplotlib parameters
        plt.rcParams.update(plt.rcParamsDefault)
        
        return fig
    
    def create_refined_method_comparison(self, aws_data: pd.Series, 
                                       modis_methods: Dict[str, pd.Series],
                                       title: str = "Method Performance Analysis",
                                       output_path: Optional[str] = None) -> plt.Figure:
        """Create refined method comparison with multiple visualization approaches."""
        
        if not modis_methods:
            logger.error("No MODIS methods provided for comparison")
            return None
        
        # Create figure with 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(self.figure_size[0] * 1.5, self.figure_size[1] * 1.5))
        
        # Calculate metrics for all methods
        method_stats = {}
        for method_name, modis_data in modis_methods.items():
            aws_clean, modis_clean = self._clean_and_align_data(aws_data, modis_data)
            if len(aws_clean) > 0:
                metrics = self._calculate_basic_metrics(aws_clean, modis_clean)
                method_stats[method_name] = metrics
        
        if not method_stats:
            logger.error("No valid data for method comparison")
            return None
        
        # 1. Performance radar chart (top-left)
        ax1 = axes[0, 0]
        self._create_performance_radar(ax1, method_stats)
        ax1.set_title("Performance Overview")
        
        # 2. Correlation vs RMSE scatter (top-right)
        ax2 = axes[0, 1]
        for method_name, stats in method_stats.items():
            color = self._get_method_color(method_name)
            ax2.scatter(stats['correlation'], stats['rmse'], 
                       s=100, color=color, alpha=0.7, label=method_name)
        
        ax2.set_xlabel('Correlation (R)')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Correlation vs RMSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Bias comparison (bottom-left)
        ax3 = axes[1, 0]
        methods = list(method_stats.keys())
        biases = [method_stats[method]['bias'] for method in methods]
        colors = [self._get_method_color(method, i) for i, method in enumerate(methods)]
        
        bars = ax3.bar(methods, biases, color=colors, alpha=0.7)
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax3.set_ylabel('Bias (MODIS - AWS)')
        ax3.set_title('Method Bias Comparison')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, bias in zip(bars, biases):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            y_pos = height + (0.01 if height >= 0 else -0.01)
            ax3.text(bar.get_x() + bar.get_width()/2., y_pos,
                    f'{bias:.3f}', ha='center', va=va, fontsize=9)
        
        # 4. Sample size and performance (bottom-right)
        ax4 = axes[1, 1]
        n_samples = [method_stats[method]['n_samples'] for method in methods]
        correlations = [method_stats[method]['correlation'] for method in methods]
        
        for method, n, r, color in zip(methods, n_samples, correlations, colors):
            ax4.scatter(n, r, s=100, color=color, alpha=0.7, label=method)
        
        ax4.set_xlabel('Sample Size (n)')
        ax4.set_ylabel('Correlation (R)')
        ax4.set_title('Sample Size vs Performance')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Refined method comparison")
        
        return fig
    
    def _create_performance_radar(self, ax, method_stats):
        """Create radar chart for method performance comparison."""
        import math
        
        # Define metrics for radar chart (normalized)
        metrics = ['correlation', 'rmse_inv', 'bias_abs_inv', 'n_samples_norm']
        metric_labels = ['Correlation', '1/RMSE', '1/|Bias|', 'Sample Size\n(normalized)']
        
        # Normalize metrics to 0-1 scale
        all_values = {metric: [] for metric in metrics}
        
        for method_name, stats in method_stats.items():
            all_values['correlation'].append(stats['correlation'])
            all_values['rmse_inv'].append(1 / max(stats['rmse'], 0.001))  # Avoid division by zero
            all_values['bias_abs_inv'].append(1 / max(abs(stats['bias']), 0.001))
            all_values['n_samples_norm'].append(stats['n_samples'])
        
        # Normalize to 0-1 scale
        normalized_stats = {}
        for method_name, stats in method_stats.items():
            normalized_stats[method_name] = {
                'correlation': stats['correlation'],
                'rmse_inv': (1 / max(stats['rmse'], 0.001)) / max(all_values['rmse_inv']),
                'bias_abs_inv': (1 / max(abs(stats['bias']), 0.001)) / max(all_values['bias_abs_inv']),
                'n_samples_norm': stats['n_samples'] / max(all_values['n_samples_norm'])
            }
        
        # Set up radar chart
        angles = [n / len(metrics) * 2 * math.pi for n in range(len(metrics))]
        angles += angles[:1]  # Complete the circle
        
        ax.set_theta_offset(math.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids([a * 180/math.pi for a in angles[:-1]], metric_labels)
        
        # Plot each method
        for i, (method_name, stats) in enumerate(normalized_stats.items()):
            values = [stats[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            color = self._get_method_color(method_name, i)
            ax.plot(angles, values, color=color, linewidth=2, label=method_name)
            ax.fill(angles, values, color=color, alpha=0.2)
        
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    def create_original_correlation_bias_analysis(self, aws_data: pd.Series, 
                                                modis_methods: Dict[str, pd.Series],
                                                glacier_name: str = "Glacier",
                                                output_path: Optional[str] = None) -> plt.Figure:
        """Create original 4-panel correlation and bias analysis."""
        logger.info("Creating original correlation and bias analysis (4-panel)")
        
        # Create figure with 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{glacier_name.upper()} Glacier - Correlation and Bias Analysis', 
                    fontsize=16, fontweight='bold')
        
        methods = list(modis_methods.keys())
        all_methods = ['AWS'] + methods
        
        # Prepare correlation matrix data
        correlation_data = {}
        performance_data = {'Method': [], 'R': [], 'RMSE': [], 'BIAS': [], 'n': []}
        bias_scatter_data = {'aws': [], 'bias': [], 'method': []}
        rmse_sample_data = {'n_samples': [], 'rmse': [], 'method': []}
        
        for method in methods:
            modis_data = modis_methods[method]
            common_idx = aws_data.index.intersection(modis_data.index)
            
            if len(common_idx) > 0:
                aws_aligned = aws_data.loc[common_idx].dropna()
                modis_aligned = modis_data.loc[common_idx].dropna()
                final_common = aws_aligned.index.intersection(modis_aligned.index)
                
                if len(final_common) > 0:
                    aws_clean = aws_aligned.loc[final_common]
                    modis_clean = modis_aligned.loc[final_common]
                    
                    # Store for correlation matrix
                    correlation_data[method] = modis_clean
                    
                    # Calculate metrics
                    r = np.corrcoef(modis_clean, aws_clean)[0, 1] if len(aws_clean) > 1 else np.nan
                    rmse = np.sqrt(np.mean((modis_clean - aws_clean) ** 2))
                    bias = np.mean(modis_clean - aws_clean)
                    
                    # Store performance data
                    performance_data['Method'].append(method)
                    performance_data['R'].append(r)
                    performance_data['RMSE'].append(rmse)
                    performance_data['BIAS'].append(bias)
                    performance_data['n'].append(len(aws_clean))
                    
                    # Store bias scatter data
                    bias_scatter_data['aws'].extend(aws_clean.tolist())
                    bias_scatter_data['bias'].extend((modis_clean - aws_clean).tolist())
                    bias_scatter_data['method'].extend([method] * len(aws_clean))
                    
                    # Store RMSE vs sample size data
                    rmse_sample_data['n_samples'].append(len(aws_clean))
                    rmse_sample_data['rmse'].append(rmse)
                    rmse_sample_data['method'].append(method)
        
        # Add AWS to correlation data
        correlation_data['AWS'] = aws_data
        
        # Panel 1: Correlation Matrix (top-left)
        self._create_correlation_matrix_panel(axes[0, 0], correlation_data)
        
        # Panel 2: Method Performance Comparison (top-right)
        self._create_performance_comparison_panel(axes[0, 1], performance_data)
        
        # Panel 3: Bias vs AWS Scatter (bottom-left)
        self._create_bias_scatter_panel(axes[1, 0], bias_scatter_data)
        
        # Panel 4: RMSE vs Sample Size (bottom-right)
        self._create_rmse_sample_panel(axes[1, 1], rmse_sample_data)
        
        plt.tight_layout()
        
        # Save figure
        self._save_figure(fig, output_path, "Original correlation and bias analysis")
        
        return fig
    
    def _create_correlation_matrix_panel(self, ax, correlation_data):
        """Create correlation matrix panel."""
        if not correlation_data:
            return
        
        # Create correlation matrix
        corr_df = pd.DataFrame(correlation_data).corr()
        
        # Create heatmap
        im = ax.imshow(corr_df.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add correlation values as text
        for i in range(len(corr_df)):
            for j in range(len(corr_df.columns)):
                text = ax.text(j, i, f'{corr_df.iloc[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        ax.set_xticks(range(len(corr_df.columns)))
        ax.set_yticks(range(len(corr_df)))
        ax.set_xticklabels(corr_df.columns)
        ax.set_yticklabels(corr_df.index)
        ax.set_title('Correlation Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Correlation Coefficient')
    
    def _create_performance_comparison_panel(self, ax, performance_data):
        """Create performance comparison panel."""
        if not performance_data['Method']:
            return
        
        df = pd.DataFrame(performance_data)
        
        # Create grouped bar chart
        x = np.arange(len(df))
        width = 0.25
        
        ax.bar(x - width, df['R'], width, label='Correlation (R)', alpha=0.8)
        ax.bar(x, df['RMSE'], width, label='RMSE', alpha=0.8)
        ax.bar(x + width, np.abs(df['BIAS']), width, label='|Bias|', alpha=0.8)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(df['Method'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_bias_scatter_panel(self, ax, bias_scatter_data):
        """Create bias scatter panel."""
        if not bias_scatter_data['aws']:
            return
        
        # Group by method and plot
        unique_methods = list(set(bias_scatter_data['method']))
        for i, method in enumerate(unique_methods):
            method_mask = [m == method for m in bias_scatter_data['method']]
            aws_values = [bias_scatter_data['aws'][j] for j, mask in enumerate(method_mask) if mask]
            bias_values = [bias_scatter_data['bias'][j] for j, mask in enumerate(method_mask) if mask]
            
            color = self._get_method_color(method, i)
            ax.scatter(aws_values, bias_values, alpha=0.6, color=color, label=method)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax.set_xlabel('AWS Albedo')
        ax.set_ylabel('Bias (MODIS - AWS)')
        ax.set_title('Bias vs AWS Albedo')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_rmse_sample_panel(self, ax, rmse_sample_data):
        """Create RMSE vs sample size panel."""
        if not rmse_sample_data['rmse']:
            return
        
        for i, method in enumerate(rmse_sample_data['method']):
            color = self._get_method_color(method, i)
            ax.scatter(rmse_sample_data['n_samples'][i], rmse_sample_data['rmse'][i], 
                      s=100, color=color, alpha=0.7, label=method)
        
        ax.set_xlabel('Sample Size (n)')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE vs Sample Size')
        ax.legend()
        ax.grid(True, alpha=0.3)