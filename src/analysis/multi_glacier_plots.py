#!/usr/bin/env python3
"""
Multi-Glacier Comparative Visualization Suite

This module provides 8 specialized visualization types for comparing
MODIS albedo method performance across different glaciers and environments.

Visualization Types:
1. Method Performance Matrix (Heatmap)
2. Cross-Glacier Scatterplot Matrix
3. Regional Comparison Boxplots  
4. Sample Size vs Performance Analysis
5. Bias Comparison Radar Chart
6. Environmental Factor Analysis
7. Temporal Coverage Comparison
8. Statistical Confidence Dashboard
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class MultiGlacierVisualizer:
    """
    Comprehensive visualization suite for multi-glacier comparative analysis.
    """
    
    def __init__(self, figsize_base: Tuple[int, int] = (12, 8), dpi: int = 300):
        """Initialize the visualizer with default settings."""
        self.figsize_base = figsize_base
        self.dpi = dpi
        
        # Color scheme for consistent visualization
        self.colors = {
            'athabasca': '#1f77b4',   # Blue
            'haig': '#ff7f0e',        # Orange  
            'coropuna': '#2ca02c',    # Green
            'MCD43A3': '#d62728',     # Red
            'MOD09GA': '#9467bd',     # Purple
            'MOD10A1': '#8c564b'      # Brown
        }
        
        # Regional colors
        self.region_colors = {
            'Canadian Rockies': '#2E8B57',  # Sea Green
            'Peruvian Andes': '#DAA520'     # Goldenrod
        }
    
    def plot_method_performance_matrix(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot 1: Method Performance Matrix Heatmap
        
        Shows performance metrics (r, RMSE, bias) for each method-glacier combination.
        """
        logger.info("Creating method performance matrix heatmap...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Method Performance Matrix Across Glaciers', fontsize=16, fontweight='bold')
        
        # Prepare data for heatmaps
        metrics = ['r', 'rmse', 'bias', 'n_samples']
        metric_titles = ['Correlation (r)', 'RMSE', 'Bias', 'Sample Size']
        
        # Create pivot tables for each metric
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            ax = axes[i//2, i%2]
            
            # Create pivot table
            pivot_data = df.pivot(index='glacier_id', columns='method', values=metric)
            
            # Create heatmap
            if metric == 'n_samples':
                sns.heatmap(pivot_data, annot=True, fmt='.0f', cmap='Blues', 
                           ax=ax, cbar_kws={'label': title})
            else:
                sns.heatmap(pivot_data, annot=True, fmt='.3f', 
                           cmap='RdYlBu_r' if metric in ['rmse', 'bias'] else 'RdYlBu',
                           ax=ax, cbar_kws={'label': title}, center=0 if metric == 'bias' else None)
            
            ax.set_title(f'{title} by Method and Glacier')
            ax.set_xlabel('MODIS Method')
            ax.set_ylabel('Glacier')
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / "plots" / "01_method_performance_matrix.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved method performance matrix: {output_file}")
    
    def plot_cross_glacier_scatterplot_matrix(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot 2: AWS vs MODIS Albedo Scatterplot Matrix
        
        Shows AWS albedo vs MODIS albedo for each method-glacier combination.
        Layout: 3 glaciers (rows) × 3 methods (columns)
        """
        logger.info("Creating AWS vs MODIS albedo scatterplot matrix...")
        
        # Need to get merged data for scatterplots
        from src.analysis.comparative_analysis import MultiGlacierComparativeAnalysis
        analyzer = MultiGlacierComparativeAnalysis()
        merged_data = analyzer.aggregate_merged_data_for_scatterplots()
        
        if merged_data.empty:
            logger.error("No merged data available for scatterplots")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle('AWS vs MODIS Albedo Correlations', fontsize=16, fontweight='bold')
        
        # Define glaciers and methods
        glaciers = ['athabasca', 'coropuna', 'haig'] 
        methods = ['MCD43A3', 'MOD09GA', 'MOD10A1']
        
        # Apply outlier filtering using 2.5σ threshold (same as rest of analysis)
        def apply_outlier_filtering(aws_vals, modis_vals):
            """Apply 2.5σ outlier filtering to AWS-MODIS pairs"""
            if len(aws_vals) < 3:  # Need minimum data for stats
                return aws_vals, modis_vals
            
            # Calculate residuals
            residuals = modis_vals - aws_vals
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            
            # 2.5σ threshold
            threshold = 2.5 * std_residual
            mask = np.abs(residuals - mean_residual) <= threshold
            
            return aws_vals[mask], modis_vals[mask]
        
        for i, glacier_id in enumerate(glaciers):
            glacier_data = merged_data[merged_data['glacier_id'] == glacier_id]
            
            for j, method in enumerate(methods):
                ax = axes[i, j]
                
                # Get AWS and MODIS data for this glacier-method combination
                aws_data = glacier_data['AWS'].dropna()
                modis_data = glacier_data[method].dropna()
                
                # Find common indices (both AWS and MODIS available)
                common_idx = aws_data.index.intersection(modis_data.index)
                
                if len(common_idx) > 0:
                    aws_vals = aws_data.loc[common_idx].values
                    modis_vals = modis_data.loc[common_idx].values
                    
                    # Apply outlier filtering
                    aws_clean, modis_clean = apply_outlier_filtering(aws_vals, modis_vals)
                    
                    if len(aws_clean) > 0:
                        # Create scatterplot
                        ax.scatter(aws_clean, modis_clean, alpha=0.6, s=20,
                                 color=self.colors.get(glacier_id, 'blue'))
                        
                        # Add 1:1 reference line
                        min_val = min(np.min(aws_clean), np.min(modis_clean))
                        max_val = max(np.max(aws_clean), np.max(modis_clean))
                        ax.plot([min_val, max_val], [min_val, max_val], 
                               'k--', alpha=0.7, linewidth=1, label='1:1 line')
                        
                        # Calculate and add trend line
                        if len(aws_clean) > 1:
                            z = np.polyfit(aws_clean, modis_clean, 1)
                            p = np.poly1d(z)
                            ax.plot(aws_clean, p(aws_clean), 'r-', alpha=0.8, linewidth=1.5)
                            
                            # Calculate comprehensive performance metrics
                            correlation_matrix = np.corrcoef(aws_clean, modis_clean)
                            r = correlation_matrix[0,1]
                            r_squared = r**2
                            
                            # Calculate error metrics
                            residuals = modis_clean - aws_clean
                            rmse = np.sqrt(np.mean(residuals**2))
                            mae = np.mean(np.abs(residuals))
                            bias = np.mean(residuals)
                            
                            # Add comprehensive statistics text
                            stats_text = (f'R = {r:.3f}\n'
                                        f'R² = {r_squared:.3f}\n'
                                        f'RMSE = {rmse:.3f}\n'
                                        f'MAE = {mae:.3f}\n'
                                        f'Bias = {bias:.3f}\n'
                                        f'n = {len(aws_clean)}')
                            
                            ax.text(0.05, 0.95, stats_text, 
                                   transform=ax.transAxes, verticalalignment='top',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                                   fontsize=8)
                
                # Customize subplot
                ax.set_xlabel('AWS Albedo' if i == 2 else '')
                ax.set_ylabel(f'{glacier_id.title()}' if j == 0 else '')
                ax.set_title(f'{method}' if i == 0 else '')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                
                # Set consistent axis limits
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / "plots" / "02_aws_vs_modis_scatterplot_matrix.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved AWS vs MODIS scatterplot matrix: {output_file}")
    
    def plot_regional_comparison_boxplots(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot 3: Glacier-Specific Method Performance Analysis
        
        Shows method performance for each glacier individually across all metrics.
        Each glacier gets its own row with 4 metrics (correlation, RMSE, bias, MAE).
        """
        logger.info("Creating glacier-specific method performance analysis...")
        
        # Create 3x4 subplot layout (3 glaciers x 4 metrics)
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        fig.suptitle('Method Performance by Glacier and Metric', 
                     fontsize=16, fontweight='bold')
        
        metrics = ['r', 'rmse', 'bias', 'mae']
        metric_titles = ['Correlation (r)', 'RMSE', 'Bias', 'MAE']
        glaciers = sorted(df['glacier_id'].unique())
        methods = sorted(df['method'].unique())
        
        # Create plots for each glacier-metric combination
        for glacier_idx, glacier_id in enumerate(glaciers):
            glacier_data = df[df['glacier_id'] == glacier_id]
            
            for metric_idx, (metric, metric_title) in enumerate(zip(metrics, metric_titles)):
                ax = axes[glacier_idx, metric_idx]
                
                # Prepare data for this glacier-metric combination
                method_values = []
                method_labels = []
                colors = []
                
                for method in methods:
                    method_data = glacier_data[glacier_data['method'] == method]
                    if not method_data.empty:
                        method_values.append(method_data[metric].iloc[0])
                        method_labels.append(method)
                        colors.append(self.colors.get(method, 'gray'))
                
                # Create bar chart (since we have single values, not distributions)
                bars = ax.bar(range(len(method_labels)), method_values, 
                             color=colors, alpha=0.7, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for i, (bar, value) in enumerate(zip(bars, method_values)):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.02,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                # Customize subplot
                ax.set_xticks(range(len(method_labels)))
                ax.set_xticklabels(method_labels, rotation=45, ha='right')
                ax.set_ylabel(metric_title)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Add glacier name to leftmost plots
                if metric_idx == 0:
                    ax.set_ylabel(f'{glacier_id.title()}\n{metric_title}', fontweight='bold')
                
                # Add metric title to top row
                if glacier_idx == 0:
                    ax.set_title(metric_title, fontweight='bold')
                
                # Set y-axis limits for better comparison across glaciers
                if metric == 'r':
                    ax.set_ylim(0, 1)
                elif metric in ['rmse', 'mae']:
                    # Set common scale for error metrics
                    all_values = df[metric].values
                    ax.set_ylim(0, max(all_values) * 1.1)
                elif metric == 'bias':
                    # Center bias around zero
                    all_values = df[metric].values
                    max_abs = max(abs(all_values))
                    ax.set_ylim(-max_abs * 1.1, max_abs * 1.1)
                    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                # Highlight best performing method
                if metric in ['r']:  # Higher is better
                    best_idx = method_values.index(max(method_values))
                elif metric in ['rmse', 'mae', 'bias']:  # Lower is better (absolute value for bias)
                    if metric == 'bias':
                        best_idx = method_values.index(min(method_values, key=abs))
                    else:
                        best_idx = method_values.index(min(method_values))
                
                # Add a star to the best performing method
                bars[best_idx].set_edgecolor('gold')
                bars[best_idx].set_linewidth(3)
                ax.text(best_idx, method_values[best_idx] + abs(method_values[best_idx])*0.05,
                       '★', ha='center', va='bottom', fontsize=12, color='gold')
        
        # Add legend for method colors
        legend_elements = [plt.Rectangle((0,0),1,1, facecolor=self.colors.get(method, 'gray'), 
                                       edgecolor='black', alpha=0.7, label=method) 
                          for method in methods]
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='none', 
                                           edgecolor='gold', linewidth=3, 
                                           label='Best Performance'))
        
        fig.legend(handles=legend_elements, loc='center', bbox_to_anchor=(0.5, 0.02), 
                  ncol=len(legend_elements), frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1)  # Make room for legend
        
        # Save plot
        output_file = output_dir / "plots" / "03_glacier_specific_method_performance.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved glacier-specific method performance analysis: {output_file}")
    
    def plot_sample_size_vs_performance(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot 4: Sample Size vs Performance Analysis
        
        Bubble plot showing relationship between sample size and method performance.
        """
        logger.info("Creating sample size vs performance analysis...")
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Sample Size vs Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Sample size vs Correlation
        ax1 = axes[0]
        for glacier_id in df['glacier_id'].unique():
            glacier_data = df[df['glacier_id'] == glacier_id]
            scatter = ax1.scatter(glacier_data['n_samples'], glacier_data['r'], 
                                s=glacier_data['rmse']*1000,  # Size represents RMSE
                                alpha=0.6, label=glacier_id.title(),
                                color=self.colors.get(glacier_id, 'gray'))
            
            # Add method labels
            for _, row in glacier_data.iterrows():
                ax1.annotate(f"{row['method']}", 
                           (row['n_samples'], row['r']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
        
        ax1.set_xlabel('Number of Samples')
        ax1.set_ylabel('Correlation (r)')
        ax1.set_title('Sample Size vs Correlation\n(Bubble size = RMSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample size vs RMSE
        ax2 = axes[1]
        for glacier_id in df['glacier_id'].unique():
            glacier_data = df[df['glacier_id'] == glacier_id]
            ax2.scatter(glacier_data['n_samples'], glacier_data['rmse'], 
                       s=np.abs(glacier_data['bias'])*2000,  # Size represents absolute bias
                       alpha=0.6, label=glacier_id.title(),
                       color=self.colors.get(glacier_id, 'gray'))
            
            # Add method labels
            for _, row in glacier_data.iterrows():
                ax2.annotate(f"{row['method']}", 
                           (row['n_samples'], row['rmse']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=9, alpha=0.8)
        
        ax2.set_xlabel('Number of Samples')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Sample Size vs RMSE\n(Bubble size = |Bias|)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / "plots" / "04_sample_size_vs_performance.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sample size vs performance analysis: {output_file}")
    
    def plot_bias_comparison_radar(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot 5: Bias Comparison Radar Chart
        
        Multi-dimensional bias pattern analysis across glaciers and methods.
        """
        logger.info("Creating bias comparison radar chart...")
        
        # Prepare data for radar chart
        methods = df['method'].unique()
        glaciers = df['glacier_id'].unique()
        
        fig, axes = plt.subplots(1, len(glaciers), figsize=(5*len(glaciers), 6), 
                                subplot_kw=dict(projection='polar'))
        if len(glaciers) == 1:
            axes = [axes]
        
        fig.suptitle('Bias Patterns Across Methods and Glaciers', fontsize=16, fontweight='bold')
        
        # Number of variables (methods)
        N = len(methods)
        
        # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        for i, glacier_id in enumerate(glaciers):
            ax = axes[i]
            glacier_data = df[df['glacier_id'] == glacier_id]
            
            # Get bias values for each method
            bias_values = []
            for method in methods:
                method_data = glacier_data[glacier_data['method'] == method]
                if not method_data.empty:
                    bias_values.append(method_data['bias'].iloc[0])
                else:
                    bias_values.append(0)
            
            bias_values += bias_values[:1]  # Complete the circle
            
            # Plot
            ax.plot(angles, bias_values, 'o-', linewidth=2, 
                   label=glacier_id.title(), color=self.colors.get(glacier_id, 'blue'))
            ax.fill(angles, bias_values, alpha=0.25, color=self.colors.get(glacier_id, 'blue'))
            
            # Add labels
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(methods)
            ax.set_title(f'{glacier_id.title()} Glacier\nBias Patterns', fontweight='bold')
            
            # Set y-axis limits based on data range
            bias_range = df['bias'].max() - df['bias'].min()
            ax.set_ylim(df['bias'].min() - 0.1*bias_range, df['bias'].max() + 0.1*bias_range)
            
            # Add grid
            ax.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / "plots" / "05_bias_comparison_radar.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved bias comparison radar chart: {output_file}")
    
    def plot_environmental_factors(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot 6: Environmental Factor Analysis
        
        Analyzes how elevation and latitude affect method performance.
        """
        logger.info("Creating environmental factor analysis...")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Environmental Factors vs Method Performance', fontsize=16, fontweight='bold')
        
        # Plot 1: Elevation vs Correlation
        ax1 = axes[0]
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            ax1.scatter(method_data['elevation'], method_data['r'], 
                       label=method, alpha=0.7, s=80,
                       color=self.colors.get(method, 'gray'))
        
        ax1.set_xlabel('Elevation (m)')
        ax1.set_ylabel('Correlation (r)')
        ax1.set_title('Elevation vs Correlation')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Elevation vs RMSE
        ax2 = axes[1]
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            ax2.scatter(method_data['elevation'], method_data['rmse'], 
                       label=method, alpha=0.7, s=80,
                       color=self.colors.get(method, 'gray'))
        
        ax2.set_xlabel('Elevation (m)')
        ax2.set_ylabel('RMSE')
        ax2.set_title('Elevation vs RMSE')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Latitude vs Correlation
        ax3 = axes[2]
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            ax3.scatter(method_data['latitude'], method_data['r'], 
                       label=method, alpha=0.7, s=80,
                       color=self.colors.get(method, 'gray'))
        
        ax3.set_xlabel('Latitude (°)')
        ax3.set_ylabel('Correlation (r)')
        ax3.set_title('Latitude vs Correlation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / "plots" / "06_environmental_factors.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved environmental factor analysis: {output_file}")
    
    def plot_temporal_coverage(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot 7: Sample Size Coverage Comparison
        
        Shows sample sizes by glacier and method.
        """
        logger.info("Creating sample size coverage comparison...")
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        fig.suptitle('Sample Sizes by Glacier and Method', fontsize=16, fontweight='bold')
        
        # Sample sizes by glacier and method
        glacier_method_counts = df.groupby(['glacier_id', 'method'])['n_samples'].first().unstack(fill_value=0)
        
        glacier_method_counts.plot(kind='bar', ax=ax, 
                                  color=[self.colors.get(method, 'gray') for method in glacier_method_counts.columns])
        ax.set_title('Sample Sizes by Glacier and Method')
        ax.set_xlabel('Glacier')
        ax.set_ylabel('Number of Samples')
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / "plots" / "07_sample_size_coverage.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved sample size coverage comparison: {output_file}")
    
    def plot_statistical_confidence_dashboard(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot 7: Combined Statistical Dashboard with Sample Sizes
        
        Shows performance metrics with confidence intervals and sample size distribution.
        """
        logger.info("Creating combined statistical confidence dashboard...")
        
        # Create layout with 2 rows, 3 columns
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 3, figure=fig, height_ratios=[1, 1])
        
        fig.suptitle('Statistical Confidence Dashboard with Sample Sizes (Outlier-Filtered Data)', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Correlation with estimated error bars
        ax1 = fig.add_subplot(gs[0, :2])
        
        # Group data for visualization
        glacier_method_stats = df.groupby(['glacier_id', 'method']).agg({
            'r': 'mean',
            'n_samples': 'mean'
        }).reset_index()
        
        # Estimate standard error based on sample size (SE ≈ sqrt((1-r²)/(n-2)))
        glacier_method_stats['estimated_se'] = np.sqrt(
            (1 - glacier_method_stats['r']**2) / (glacier_method_stats['n_samples'] - 2)
        )
        
        x_pos = np.arange(len(glacier_method_stats))
        bars = ax1.bar(x_pos, glacier_method_stats['r'], 
                      yerr=glacier_method_stats['estimated_se'], 
                      capsize=5, alpha=0.7)
        
        # Color bars by method
        for i, (_, row) in enumerate(glacier_method_stats.iterrows()):
            bars[i].set_color(self.colors.get(row['method'], 'gray'))
        
        ax1.set_xlabel('Glacier - Method')
        ax1.set_ylabel('Correlation (r)')
        ax1.set_title('Correlation with Estimated Standard Error')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([f"{row['glacier_id']}\n{row['method']}" 
                            for _, row in glacier_method_stats.iterrows()], 
                           rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sample Size Distribution Pie Chart
        ax2 = fig.add_subplot(gs[0, 2])
        
        # Show sample size distribution by method
        method_counts = df.groupby('method')['n_samples'].sum()
        colors = [self.colors.get(method, 'gray') for method in method_counts.index]
        
        wedges, texts, autotexts = ax2.pie(method_counts.values, 
                                          labels=method_counts.index,
                                          colors=colors,
                                          autopct='%1.0f',
                                          startangle=90)
        ax2.set_title('Sample Size Distribution\n(Total Observations)')
        
        # Plot 3: Sample sizes bar chart by glacier and method
        ax3 = fig.add_subplot(gs[1, :])
        
        # Sample sizes by glacier and method
        glacier_method_counts = df.groupby(['glacier_id', 'method'])['n_samples'].first().unstack(fill_value=0)
        
        glacier_method_counts.plot(kind='bar', ax=ax3, 
                                  color=[self.colors.get(method, 'gray') for method in glacier_method_counts.columns])
        ax3.set_title('Sample Sizes by Glacier and Method')
        ax3.set_xlabel('Glacier')
        ax3.set_ylabel('Number of Samples')
        ax3.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        # Save plot
        output_file = output_dir / "plots" / "07_combined_statistical_dashboard.png"
        plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved combined statistical confidence dashboard: {output_file}")
    
    def plot_glacier_maps(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot 7: Multi-Glacier Mapping Suite
        
        Creates comprehensive maps showing glacier extents, AWS stations, and MODIS coverage.
        """
        logger.info("Creating multi-glacier mapping suite...")
        
        try:
            from src.analysis.glacier_mapping_simple import MultiGlacierMapperSimple
            
            # Initialize simplified mapper (no cartopy required)
            mapper = MultiGlacierMapperSimple()
            
            # Generate all maps
            mapper.generate_all_maps(output_dir)
            
            logger.info("Successfully generated glacier mapping suite")
            
        except Exception as e:
            logger.error(f"Error generating glacier maps: {e}")
            # Don't raise - allow other plots to continue

    def generate_all_plots(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Generate all 7 comparative visualization types (including maps)."""
        logger.info("Starting generation of all comparative plots...")
        
        if df.empty:
            logger.error("No data available for plotting")
            return
        
        try:
            # Generate all plots (excluding environmental factors and standalone sample sizes)
            self.plot_method_performance_matrix(df, output_dir)
            self.plot_cross_glacier_scatterplot_matrix(df, output_dir)
            self.plot_regional_comparison_boxplots(df, output_dir)
            self.plot_sample_size_vs_performance(df, output_dir)
            self.plot_bias_comparison_radar(df, output_dir)
            self.plot_statistical_confidence_dashboard(df, output_dir)  # Combined with sample sizes
            self.plot_glacier_maps(df, output_dir)  # New mapping suite
            
            logger.info("Successfully generated all 7 comparative visualization types")
            
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            raise


def main():
    """Test the visualization suite."""
    # This would typically be called with real data from the comparative analysis
    print("Multi-Glacier Visualization Suite initialized")
    print("Available plot types:")
    print("1. Method Performance Matrix")
    print("2. AWS vs MODIS Albedo Scatterplot Matrix (3x3)")
    print("3. Glacier-Specific Method Performance")
    print("4. Sample Size vs Performance")
    print("5. Bias Comparison Radar Chart")
    print("6. Combined Statistical Dashboard (with sample sizes)")


if __name__ == "__main__":
    main()