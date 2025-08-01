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
6. Statistical Confidence Dashboard
7. Multi-Glacier Mapping Suite
8. Seasonal Analysis: Multi-Glacier Monthly Boxplots (NEW)
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
        
        # Pixel selection metadata
        self.pixel_selection_info = None
        self.analysis_mode = "unknown"
        
        # Color scheme for consistent visualization
        self.colors = {
            'athabasca': '#1f77b4',   # Blue
            'haig': '#ff7f0e',        # Orange  
            'coropuna': '#2ca02c',    # Green
            'MCD43A3': '#d62728',     # Red
            'MOD09GA': '#9467bd',     # Purple
            'MOD10A1': '#8c564b'      # Brown
        }
    
    def set_pixel_selection_info(self, pixel_info: Dict[str, int], analysis_mode: str = "selected_pixels"):
        """
        Set pixel selection information for enhanced plot labeling.
        
        Args:
            pixel_info: Dictionary with glacier_id -> pixel_count mapping
            analysis_mode: "selected_pixels", "all_pixels", or "mixed"
        """
        self.pixel_selection_info = pixel_info
        self.analysis_mode = analysis_mode
        
    def _get_pixel_selection_subtitle(self) -> str:
        """Generate subtitle text showing pixel selection details."""
        if not self.pixel_selection_info or self.analysis_mode == "unknown":
            return ""
            
        if self.analysis_mode == "all_pixels":
            pixel_counts = "/".join([str(count) for count in self.pixel_selection_info.values()])
            return f"All Available Pixels: {pixel_counts} (Athabasca/Haig/Coropuna)"
        elif self.analysis_mode == "selected_pixels":
            pixel_counts = "/".join([str(count) for count in self.pixel_selection_info.values()])
            return f"Selected Best Pixels: {pixel_counts} (Closest to AWS Stations)"
        else:
            return "Mixed Pixel Selection"
    
    def _get_enhanced_filename(self, base_filename: str) -> str:
        """Add pixel selection indicator to filename."""
        if self.analysis_mode == "selected_pixels":
            return base_filename.replace(".png", "_selected_pixels.png")
        elif self.analysis_mode == "all_pixels":
            return base_filename.replace(".png", "_all_pixels.png")
        else:
            return base_filename
        
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
        
        # Enhanced title with pixel selection information
        main_title = 'Method Performance Matrix Across Glaciers'
        subtitle = self._get_pixel_selection_subtitle()
        if subtitle:
            fig.suptitle(f'{main_title}\n{subtitle}', fontsize=16, fontweight='bold')
        else:
            fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
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
        
        # Save plot with enhanced filename
        base_filename = "01_method_performance_matrix.png"
        enhanced_filename = self._get_enhanced_filename(base_filename)
        output_file = output_dir / "plots" / enhanced_filename
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
        from analysis.comparative.multi_glacier import MultiGlacierComparativeAnalysis
        analyzer = MultiGlacierComparativeAnalysis()
        merged_data = analyzer.aggregate_merged_data_for_scatterplots()
        
        if merged_data.empty:
            logger.error("No merged data available for scatterplots")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        # Enhanced title with pixel selection information
        main_title = 'AWS vs MODIS Albedo Correlations'
        subtitle = self._get_pixel_selection_subtitle()
        if subtitle:
            fig.suptitle(f'{main_title}\n{subtitle}', fontsize=16, fontweight='bold')
        else:
            fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
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
                ax.set_xlabel('AWS Albedo', fontsize=10)
                ax.set_ylabel('MODIS Albedo', fontsize=10)
                ax.set_title(f'{glacier_id.title()} - {method}', fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
                
                # Set consistent axis limits
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.subplots_adjust(wspace=0.05)  # Make plots much closer horizontally
        
        # Save plot
        # Save plot with enhanced filename
        base_filename = "02_aws_vs_modis_scatterplot_matrix.png"
        enhanced_filename = self._get_enhanced_filename(base_filename)
        output_file = output_dir / "plots" / enhanced_filename
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
        # Enhanced title with pixel selection information
        main_title = 'Method Performance by Glacier and Metric'
        subtitle = self._get_pixel_selection_subtitle()
        if subtitle:
            fig.suptitle(f'{main_title}\n{subtitle}', fontsize=16, fontweight='bold')
        else:
            fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
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
        # Save plot with enhanced filename
        base_filename = "03_glacier_specific_method_performance.png"
        enhanced_filename = self._get_enhanced_filename(base_filename)
        output_file = output_dir / "plots" / enhanced_filename
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
        # Enhanced title with pixel selection information
        main_title = 'Sample Size vs Performance Analysis'
        subtitle = self._get_pixel_selection_subtitle()
        if subtitle:
            fig.suptitle(f'{main_title}\n{subtitle}', fontsize=16, fontweight='bold')
        else:
            fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
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
        # Save plot with enhanced filename
        base_filename = "04_sample_size_vs_performance.png"
        enhanced_filename = self._get_enhanced_filename(base_filename)
        output_file = output_dir / "plots" / enhanced_filename
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
        
        # Enhanced title with pixel selection information
        main_title = 'Bias Patterns Across Methods and Glaciers'
        subtitle = self._get_pixel_selection_subtitle()
        if subtitle:
            fig.suptitle(f'{main_title}\n{subtitle}', fontsize=16, fontweight='bold')
        else:
            fig.suptitle(main_title, fontsize=16, fontweight='bold')
        
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
        # Save plot with enhanced filename
        base_filename = "05_bias_comparison_radar.png"
        enhanced_filename = self._get_enhanced_filename(base_filename)
        output_file = output_dir / "plots" / enhanced_filename
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
        
        # Enhanced title with pixel selection information
        main_title = 'Statistical Confidence Dashboard with Sample Sizes'
        subtitle = self._get_pixel_selection_subtitle()
        if subtitle:
            fig.suptitle(f'{main_title}\n{subtitle}', fontsize=16, fontweight='bold')
        else:
            fig.suptitle(f'{main_title}\n(Outlier-Filtered Data)', fontsize=16, fontweight='bold')
        
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
        # Save plot with enhanced filename
        base_filename = "07_combined_statistical_dashboard.png"
        enhanced_filename = self._get_enhanced_filename(base_filename)
        output_file = output_dir / "plots" / enhanced_filename
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
            from analysis.spatial.glacier_mapping_simple import MultiGlacierMapperSimple
            
            # Initialize simplified mapper (no cartopy required)
            mapper = MultiGlacierMapperSimple()
            
            # Generate all maps
            mapper.generate_all_maps(output_dir)
            
            logger.info("Successfully generated glacier mapping suite")
            
        except Exception as e:
            logger.error(f"Error generating glacier maps: {e}")
            # Don't raise - allow other plots to continue

    def plot_seasonal_analysis_multi_glacier(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot 8: Multi-Glacier Seasonal Analysis with Method-Separated Subplots
        
        Creates a grid where each glacier has 3 vertical subplots (one per method)
        showing year-based seasonal patterns. AWS reference gets its own subplot.
        """
        logger.info("Creating multi-glacier seasonal analysis...")
        
        try:
            # Load seasonal data
            seasonal_data = self._load_seasonal_data()
            
            if seasonal_data is None or seasonal_data.empty:
                logger.warning("No seasonal data available for analysis")
                return
            
            # Create subplot layout: 3 glaciers × 3 methods = 9 subplots
            # Layout: 3 columns (glaciers), 3 rows (methods)
            fig = plt.figure(figsize=(20, 16))
            
            # Enhanced title with pixel selection information
            main_title = 'Multi-Glacier Seasonal Analysis - Method-Separated Time Series'
            subtitle = self._get_pixel_selection_subtitle()
            if subtitle:
                fig.suptitle(f'{main_title}\n{subtitle}\nEach Year-Month Pattern by Method', 
                           fontsize=16, fontweight='bold', y=0.96)
            else:
                fig.suptitle(f'{main_title}\nEach Year-Month Pattern by Method', 
                           fontsize=16, fontweight='bold', y=0.96)
            
            # Define glaciers and methods
            glaciers = ['athabasca', 'haig', 'coropuna']
            methods = ['MCD43A3', 'MOD09GA', 'MOD10A1']
            
            # Method colors (since each method has its own subplot, we can use a single color per method)
            method_colors = {
                'MCD43A3': '#d62728',  # Red
                'MOD09GA': '#9467bd',  # Purple  
                'MOD10A1': '#8c564b',  # Brown
            }
            
            # Create subplots grid: 3 rows (methods) × 3 columns (glaciers)
            subplot_grid = {}
            for method_idx, method in enumerate(methods):
                for glacier_idx, glacier in enumerate(glaciers):
                    # Position in grid (1-indexed for matplotlib)
                    subplot_position = method_idx * 3 + glacier_idx + 1
                    ax = fig.add_subplot(3, 3, subplot_position)
                    subplot_grid[(method, glacier)] = ax
            
            # Process each glacier-method combination
            for glacier in glaciers:
                glacier_data = seasonal_data[seasonal_data['glacier_id'] == glacier]
                
                if glacier_data.empty:
                    # Handle empty data for all methods of this glacier
                    for method in methods:
                        ax = subplot_grid[(method, glacier)]
                        ax.text(0.5, 0.5, f'No data available\nfor {glacier}', 
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=10, style='italic')
                        ax.set_title(f'{glacier.title()}\n{method}', fontsize=12, fontweight='bold')
                    continue
                
                for method in methods:
                    ax = subplot_grid[(method, glacier)]
                    
                    # Filter data for this method
                    method_data = glacier_data[glacier_data['method'] == method]
                    
                    if not method_data.empty:
                        self._plot_single_method_seasonal_boxplots(ax, method_data, method, method_colors[method])
                        
                        # Count data points for subtitle
                        unique_periods = len(method_data['year_month'].unique())
                        logger.info(f"Plotted {method} for {glacier}: {unique_periods} time periods")
                    else:
                        ax.text(0.5, 0.5, f'No {method} data\navailable', 
                               ha='center', va='center', transform=ax.transAxes,
                               fontsize=10, style='italic')
                    
                    # Set subplot title and styling
                    ax.set_title(f'{glacier.title()} Glacier\n{method}', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Albedo', fontsize=10)
                    ax.set_ylim(0, 1.0)
                    ax.grid(True, alpha=0.3)
                    
                    # Only show x-axis labels on bottom row
                    if method == methods[-1]:  # Last method (bottom row)
                        ax.set_xlabel('Year', fontsize=10)
                    else:
                        ax.set_xticklabels([])
            
            # Create method legend (since colors now represent methods clearly)
            legend_elements = []
            for method, color in method_colors.items():
                legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, 
                                                   alpha=0.7, edgecolor='black', 
                                                   label=method))
            
            # Add legend at the bottom
            fig.legend(handles=legend_elements, loc='lower center', ncol=3, 
                      fontsize=12, frameon=True, fancybox=True, shadow=True,
                      bbox_to_anchor=(0.5, 0.01))
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.08, top=0.90)  # Make room for title and legend
            
            # Save plot
            filename = self._get_enhanced_filename("seasonal_analysis_multi_glacier.png")
            output_file = output_dir / "plots" / filename
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Saved seasonal analysis plot: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating seasonal analysis plot: {e}")
            # Don't raise - allow other plots to continue

    def plot_seasonal_analysis_time_series(self, df: pd.DataFrame, output_dir: Path) -> None:
        """
        Plot 9: Multi-Glacier Seasonal Time Series Analysis (Ren et al. 2021 Style)
        
        Creates 3 subplots (one per glacier) showing continuous time series
        with AWS data as lines and MODIS methods as scatter points.
        """
        logger.info("Creating multi-glacier seasonal time series analysis...")
        
        try:
            # Load seasonal data with actual dates
            seasonal_data = self._load_seasonal_data()
            
            if seasonal_data is None or seasonal_data.empty:
                logger.warning("No seasonal data available for time series analysis")
                return
            
            # Create 3 subplots layout (one per glacier)
            fig, axes = plt.subplots(3, 1, figsize=(16, 12))
            
            # Enhanced title with pixel selection information
            main_title = 'Multi-Glacier Seasonal Time Series Analysis'
            subtitle = self._get_pixel_selection_subtitle()
            if subtitle:
                fig.suptitle(f'{main_title}\n{subtitle}\nContinuous Time Series by Glacier', 
                           fontsize=16, fontweight='bold', y=0.96)
            else:
                fig.suptitle(f'{main_title}\nContinuous Time Series by Glacier', 
                           fontsize=16, fontweight='bold', y=0.96)
            
            # Define glaciers and colors/markers (Ren et al. style)
            glaciers = ['athabasca', 'haig', 'coropuna']
            
            # Data source styling (matching Ren et al. 2021)
            source_styles = {
                'AWS': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'markersize': 3, 'linewidth': 1.5, 'alpha': 0.8},
                'MCD43A3': {'color': '#d62728', 'marker': '^', 'linestyle': 'None', 'markersize': 4, 'alpha': 0.7},
                'MOD09GA': {'color': '#9467bd', 'marker': 'o', 'linestyle': 'None', 'markersize': 4, 'alpha': 0.7},
                'MOD10A1': {'color': '#2ca02c', 'marker': 's', 'linestyle': 'None', 'markersize': 4, 'alpha': 0.7}
            }
            
            # Process each glacier
            for glacier_idx, glacier in enumerate(glaciers):
                ax = axes[glacier_idx]
                glacier_data = seasonal_data[seasonal_data['glacier_id'] == glacier]
                
                if glacier_data.empty:
                    ax.text(0.5, 0.5, f'No data available for {glacier}', 
                           ha='center', va='center', transform=ax.transAxes,
                           fontsize=12, style='italic')
                    ax.set_title(f'{glacier.title()} Glacier', fontsize=14, fontweight='bold')
                    continue
                
                # Convert date column to datetime
                glacier_data['date'] = pd.to_datetime(glacier_data['date'])
                
                # Plot AWS data as continuous line
                aws_data = glacier_data.dropna(subset=['aws_albedo']).sort_values('date')
                if not aws_data.empty:
                    style = source_styles['AWS']
                    ax.plot(aws_data['date'], aws_data['aws_albedo'], 
                           color=style['color'], marker=style['marker'], 
                           linestyle=style['linestyle'], markersize=style['markersize'],
                           linewidth=style['linewidth'], alpha=style['alpha'],
                           label='Field observation', zorder=1)
                
                # Plot each MODIS method as scatter points
                methods = ['MCD43A3', 'MOD09GA', 'MOD10A1']
                for method in methods:
                    method_data = glacier_data[glacier_data['method'] == method].dropna(subset=['albedo'])
                    if not method_data.empty:
                        style = source_styles[method]
                        ax.scatter(method_data['date'], method_data['albedo'],
                                 color=style['color'], marker=style['marker'],
                                 s=style['markersize']**2, alpha=style['alpha'],
                                 label=method, zorder=2)
                
                # Styling for each subplot
                ax.set_title(f'{glacier.title()} Glacier', fontsize=14, fontweight='bold')
                ax.set_ylabel('Albedo', fontsize=12)
                ax.set_ylim(0, 1.0)
                ax.grid(True, alpha=0.3)
                
                # Format dates on x-axis
                ax.tick_params(axis='x', rotation=45)
                
                # Only show x-axis label on bottom subplot
                if glacier_idx == len(glaciers) - 1:
                    ax.set_xlabel('Date', fontsize=12)
                else:
                    ax.set_xticklabels([])
            
            # Create unified legend
            legend_elements = []
            for source, style in source_styles.items():
                if source == 'AWS':
                    legend_elements.append(plt.Line2D([0], [0], color=style['color'], 
                                                    marker=style['marker'], linestyle=style['linestyle'],
                                                    markersize=6, linewidth=2, label='Field observation'))
                else:
                    legend_elements.append(plt.Line2D([0], [0], color=style['color'], 
                                                    marker=style['marker'], linestyle='None',
                                                    markersize=6, label=source))
            
            # Add legend at the bottom
            fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
                      fontsize=12, frameon=True, fancybox=True, shadow=True,
                      bbox_to_anchor=(0.5, 0.01))
            
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.12, top=0.88)  # Make room for title and legend
            
            # Save plot
            filename = self._get_enhanced_filename("seasonal_time_series_multi_glacier.png")
            output_file = output_dir / "plots" / filename
            plt.savefig(output_file, dpi=self.dpi, bbox_inches='tight', facecolor='white')
            plt.close()
            
            logger.info(f"Saved seasonal time series plot: {output_file}")
            
        except Exception as e:
            logger.error(f"Error creating seasonal time series plot: {e}")
            # Don't raise - allow other plots to continue

    def _load_seasonal_data(self) -> Optional[pd.DataFrame]:
        """Load real merged data with dates for seasonal analysis."""
        try:
            logger.info("Loading real seasonal data from analysis results...")
            
            # Import required modules
            import pandas as pd
            import numpy as np
            from pathlib import Path
            
            # Find the latest analysis results for each glacier
            outputs_dir = Path("outputs")
            seasonal_data_list = []
            
            # Glacier data mapping - find latest results for each glacier
            glacier_patterns = {
                'athabasca': 'athabasca_comprehensive_*',
                'haig': 'haig_best_pixel_*',  # Use best pixel analysis for Haig
                'coropuna': 'coropuna_comprehensive_*'
            }
            
            for glacier_id, pattern in glacier_patterns.items():
                try:
                    logger.info(f"Loading seasonal data for {glacier_id}...")
                    
                    # Find the latest results directory for this glacier
                    matching_dirs = list(outputs_dir.glob(pattern))
                    if not matching_dirs:
                        logger.warning(f"No analysis results found for {glacier_id} with pattern {pattern}")
                        continue
                    
                    # Get the most recent directory (by timestamp in name)
                    latest_dir = max(matching_dirs, key=lambda x: x.name)
                    
                    # Extract the analysis type from directory name (e.g., 'comprehensive', 'best_pixel')
                    dir_parts = latest_dir.name.split('_')
                    # Handle multi-word analysis types like 'best_pixel'
                    if len(dir_parts) >= 3 and dir_parts[1] == 'best' and dir_parts[2] == 'pixel':
                        analysis_type = 'best_pixel'
                    elif len(dir_parts) > 1:
                        analysis_type = dir_parts[1]
                    else:
                        analysis_type = 'merged'
                    merged_data_file = latest_dir / "results" / f"{glacier_id}_{analysis_type}_merged_data.csv"
                    
                    if not merged_data_file.exists():
                        logger.warning(f"Merged data file not found: {merged_data_file}")
                        continue
                    
                    # Load the merged data
                    logger.info(f"Loading data from {merged_data_file}")
                    merged_df = pd.read_csv(merged_data_file)
                    
                    if merged_df.empty:
                        logger.warning(f"Empty merged data for {glacier_id}")
                        continue
                    
                    # Convert date column
                    merged_df['date'] = pd.to_datetime(merged_df['date'])
                    
                    # Extract year and month for filtering
                    merged_df['year'] = merged_df['date'].dt.year
                    merged_df['month'] = merged_df['date'].dt.month
                    
                    # Filter for melt season months (June=6 to September=9)
                    melt_season_data = merged_df[merged_df['month'].isin([6, 7, 8, 9])].copy()
                    
                    if melt_season_data.empty:
                        logger.warning(f"No melt season data for {glacier_id}")
                        continue
                    
                    # Reshape data from wide to long format for methods
                    methods = ['MCD43A3', 'MOD09GA', 'MOD10A1']
                    glacier_seasonal_data = []
                    
                    for method in methods:
                        if method in melt_season_data.columns:
                            # Extract data for this method
                            method_data = melt_season_data[['date', 'year', 'month', method, 'AWS']].copy()
                            method_data = method_data.dropna(subset=[method, 'AWS'])  # Remove rows with missing data
                            
                            if not method_data.empty:
                                method_data['glacier_id'] = glacier_id
                                method_data['method'] = method
                                method_data['albedo'] = method_data[method]
                                method_data['aws_albedo'] = method_data['AWS']
                                
                                # Keep only the columns we need
                                method_data = method_data[['glacier_id', 'method', 'date', 'year', 'month', 'albedo', 'aws_albedo']]
                                glacier_seasonal_data.append(method_data)
                    
                    if glacier_seasonal_data:
                        # Combine all methods for this glacier
                        glacier_combined = pd.concat(glacier_seasonal_data, ignore_index=True)
                        seasonal_data_list.append(glacier_combined)
                        
                        # Log data availability summary
                        year_range = f"{glacier_combined['year'].min()}-{glacier_combined['year'].max()}"
                        months_available = sorted(glacier_combined['month'].unique())
                        month_names = {6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep'}
                        month_str = ', '.join([month_names[m] for m in months_available])
                        
                        logger.info(f"Loaded {len(glacier_combined)} seasonal records for {glacier_id}")
                        logger.info(f"  - Years: {year_range}")
                        logger.info(f"  - Months: {month_str}")
                        logger.info(f"  - Methods: {glacier_combined['method'].unique()}")
                        
                except Exception as e:
                    logger.error(f"Error loading seasonal data for {glacier_id}: {e}")
                    continue
            
            if seasonal_data_list:
                # Combine all glacier data
                combined_data = pd.concat(seasonal_data_list, ignore_index=True)
                
                # Log overall summary
                total_records = len(combined_data)
                glaciers_loaded = combined_data['glacier_id'].nunique()
                year_range = f"{combined_data['year'].min()}-{combined_data['year'].max()}"
                
                logger.info(f"Successfully loaded real seasonal data:")
                logger.info(f"  - Total records: {total_records:,}")
                logger.info(f"  - Glaciers: {glaciers_loaded}")
                logger.info(f"  - Year range: {year_range}")
                logger.info(f"  - Melt season focus: June-September")
                
                return combined_data
            else:
                logger.warning("No seasonal data could be loaded from any glacier")
                return None
                
        except Exception as e:
            logger.error(f"Error loading seasonal data: {e}")
            return None

    def _plot_single_method_seasonal_boxplots(self, ax, method_data: pd.DataFrame, method: str, method_color: str):
        """Plot seasonal boxplots for a single method using hierarchical year-month positioning."""
        try:
            # Create year_month column for chronological sorting
            method_data['year_month'] = method_data['year'].astype(str) + '-' + method_data['month'].astype(str).str.zfill(2)
            unique_year_months = sorted(method_data['year_month'].unique())
            
            if not unique_year_months:
                logger.warning(f"No year-month combinations found for {method}")
                return
            
            # Create hierarchical positioning system 
            position_info = self._create_hierarchical_positions(unique_year_months)
            if not position_info:
                logger.warning(f"Could not create hierarchical positions for {method}")
                return
            
            # Prepare boxplot data and positions
            boxplot_data = []
            boxplot_positions = []
            
            # Plot boxplots for each year-month combination
            for year_month in unique_year_months:
                year, month = year_month.split('-')
                month_int = int(month)
                
                # Get data for this year-month
                month_data = method_data[method_data['year_month'] == year_month]['albedo'].values
                
                if len(month_data) > 0:
                    boxplot_data.append(month_data)
                    # Use center position since we only have one method per subplot
                    if (year, month_int) in position_info['month_positions']:
                        boxplot_positions.append(position_info['month_positions'][(year, month_int)])
            
            # Create boxplots with single method color
            if boxplot_data:
                bp = ax.boxplot(boxplot_data, positions=boxplot_positions, 
                               widths=2.5, patch_artist=True,  # Wider since only one method
                               boxprops=dict(facecolor=method_color, alpha=0.7),
                               medianprops=dict(color='white', linewidth=2),
                               whiskerprops=dict(color='black', linewidth=1.5),
                               capprops=dict(color='black', linewidth=1.5),
                               flierprops=dict(marker='o', markerfacecolor=method_color, 
                                             markersize=4, markeredgecolor='black', alpha=0.5))
                
                # Add hierarchical labels (years and months)
                self._add_hierarchical_labels(ax, position_info, unique_year_months)
                
                # Set axis limits and styling  
                if boxplot_positions:
                    x_margin = (max(boxplot_positions) - min(boxplot_positions)) * 0.05
                    ax.set_xlim(min(boxplot_positions) - x_margin, max(boxplot_positions) + x_margin)
                
        except Exception as e:
            logger.error(f"Error plotting single method seasonal boxplots for {method}: {e}")

    def _plot_glacier_seasonal_boxplots(self, ax, glacier_data: pd.DataFrame, glacier_id: str, method_colors: dict):
        """Plot seasonal boxplots for a single glacier with all 3 methods using hierarchical positioning."""
        try:
            # Get available methods for this glacier
            available_methods = glacier_data['method'].unique()
            
            # Get unique year-month combinations, sorted chronologically
            glacier_data['year_month'] = glacier_data['year'].astype(str) + '-' + glacier_data['month'].astype(str).str.zfill(2)
            unique_year_months = sorted(glacier_data['year_month'].unique())
            
            if not unique_year_months:
                logger.warning(f"No year-month combinations found for {glacier_id}")
                return
            
            # Create hierarchical positioning system
            position_info = self._create_hierarchical_positions(unique_year_months)
            if not position_info:
                logger.warning(f"Could not create hierarchical positions for {glacier_id}")
                return
            
            # Track plotted data for legend
            methods_plotted = []
            
            # Plot each method using hierarchical positions
            for method in ['MCD43A3', 'MOD09GA', 'MOD10A1']:
                if method in available_methods:
                    boxplot_data = []
                    boxplot_positions = []
                    
                    # Get data for each year-month combination
                    for year_month in unique_year_months:
                        year, month = year_month.split('-')
                        month_int = int(month)
                        
                        # Get position for this year-month-method combination
                        if (year, month_int, method) in position_info['method_positions']:
                            method_data = glacier_data[
                                (glacier_data['year_month'] == year_month) & 
                                (glacier_data['method'] == method)
                            ]['albedo'].values
                            
                            if len(method_data) > 0:
                                boxplot_data.append(method_data)
                                boxplot_positions.append(position_info['method_positions'][(year, month_int, method)])
                    
                    # Create boxplots for this method
                    if boxplot_data:
                        bp = ax.boxplot(boxplot_data, positions=boxplot_positions, 
                                       widths=0.6, patch_artist=True,
                                       boxprops=dict(facecolor=method_colors[method], alpha=0.7),
                                       medianprops=dict(color='black', linewidth=2))
                        methods_plotted.append(method)
            
            # Add hierarchical labels (years and months)
            self._add_hierarchical_labels(ax, position_info, unique_year_months)
            
            logger.info(f"Plotted seasonal data for {glacier_id}: {len(unique_year_months)} time periods, methods: {methods_plotted}")
                        
        except Exception as e:
            logger.error(f"Error plotting seasonal boxplots for {glacier_id}: {e}")

    def _create_hierarchical_positions(self, year_months: List[str]) -> Dict[str, Any]:
        """Create hierarchical positioning system for year-month-method layout."""
        try:
            # Group year-months by year for hierarchical positioning
            year_groups = {}
            for year_month in year_months:
                year, month = year_month.split('-')
                if year not in year_groups:
                    year_groups[year] = []
                year_groups[year].append(int(month))
            
            # Sort years and months
            sorted_years = sorted(year_groups.keys())
            for year in sorted_years:
                year_groups[year] = sorted(year_groups[year])
            
            # Calculate hierarchical positions
            year_spacing = 20  # Wide gaps between years
            month_spacing = 4  # Medium gaps between months within a year
            method_spacing = 0.8  # Narrow gaps between methods within a month
            
            # Build position mapping
            position_info = {
                'year_positions': {},  # year -> center position
                'month_positions': {},  # (year, month) -> center position  
                'method_positions': {},  # (year, month, method) -> position
                'year_groups': year_groups,
                'sorted_years': sorted_years
            }
            
            current_position = 0
            
            for year_idx, year in enumerate(sorted_years):
                year_start = current_position
                months = year_groups[year]
                
                # Calculate positions for months within this year
                for month_idx, month in enumerate(months):
                    month_center = current_position + month_idx * month_spacing
                    position_info['month_positions'][(year, month)] = month_center
                    
                    # Calculate positions for methods within this month
                    for method_idx, method in enumerate(['MCD43A3', 'MOD09GA', 'MOD10A1']):
                        method_position = month_center + (method_idx - 1) * method_spacing
                        position_info['method_positions'][(year, month, method)] = method_position
                
                # Calculate year center position
                year_end = current_position + (len(months) - 1) * month_spacing
                year_center = (year_start + year_end) / 2
                position_info['year_positions'][year] = year_center
                
                # Move to start of next year (with gap)
                current_position += len(months) * month_spacing + year_spacing
            
            return position_info
            
        except Exception as e:
            logger.error(f"Error creating hierarchical positions: {e}")
            return {}

    def _add_hierarchical_labels(self, ax, position_info: Dict[str, Any], year_months: List[str]):
        """Add hierarchical year and month labels with proper spacing."""
        try:
            if not position_info:
                return
                
            # Add year labels (top level)
            for year, year_center in position_info['year_positions'].items():
                ax.text(year_center, -0.15, year, ha='center', va='top', 
                       transform=ax.get_xaxis_transform(), fontsize=12, fontweight='bold')
            
            # Add month labels (middle level) - single letter abbreviations
            month_names = {5: 'M', 6: 'J', 7: 'J', 8: 'A', 9: 'S', 10: 'O'}
            for (year, month), month_center in position_info['month_positions'].items():
                month_name = month_names.get(month, f'M{month}')
                ax.text(month_center, -0.08, month_name, ha='center', va='top',
                       transform=ax.get_xaxis_transform(), fontsize=10, rotation=0)
            
            # Add year separators (visual dividers)  
            sorted_years = position_info['sorted_years']
            for i in range(len(sorted_years) - 1):
                # Find separation point between years
                current_year = sorted_years[i]
                next_year = sorted_years[i + 1]
                
                # Get last month of current year and first month of next year
                current_months = position_info['year_groups'][current_year]
                next_months = position_info['year_groups'][next_year]
                
                if current_months and next_months:
                    last_month_pos = position_info['month_positions'][(current_year, max(current_months))]
                    first_month_pos = position_info['month_positions'][(next_year, min(next_months))]
                    sep_position = (last_month_pos + first_month_pos) / 2
                    
                    ax.axvline(x=sep_position, color='gray', linestyle='--', alpha=0.4, linewidth=1)
            
            # Remove default tick labels and ticks
            ax.set_xticks([])
            ax.set_xticklabels([])
                
        except Exception as e:
            logger.error(f"Error adding hierarchical labels: {e}")

    def _plot_aws_seasonal_boxplots(self, ax, seasonal_data: pd.DataFrame, method_colors: dict):
        """Plot AWS seasonal boxplots combined from all glaciers using hierarchical positioning."""
        try:
            # Combine AWS data from all glaciers
            aws_data = seasonal_data.copy()
            
            # Get unique year-month combinations, sorted chronologically
            aws_data['year_month'] = aws_data['year'].astype(str) + '-' + aws_data['month'].astype(str).str.zfill(2)
            unique_year_months = sorted(aws_data['year_month'].unique())
            
            if not unique_year_months:
                logger.warning("No year-month combinations found for AWS data")
                return
            
            # Create hierarchical positioning system
            position_info = self._create_hierarchical_positions(unique_year_months)
            if not position_info:
                logger.warning("Could not create hierarchical positions for AWS data")
                return
            
            # Prepare data for each year-month combination
            boxplot_data = []
            boxplot_positions = []
            
            for year_month in unique_year_months:
                year, month = year_month.split('-')
                month_int = int(month)
                
                # For AWS, we use the month center position (no method offset)
                if (year, month_int) in position_info['month_positions']:
                    aws_month_data = aws_data[aws_data['year_month'] == year_month]['aws_albedo'].values
                    if len(aws_month_data) > 0:
                        boxplot_data.append(aws_month_data)
                        boxplot_positions.append(position_info['month_positions'][(year, month_int)])
            
            # Create AWS boxplots
            if boxplot_data:
                bp = ax.boxplot(boxplot_data, positions=boxplot_positions, 
                               widths=2.0, patch_artist=True,  # Wider since no method grouping
                               boxprops=dict(facecolor=method_colors['AWS'], alpha=0.7),
                               medianprops=dict(color='white', linewidth=2))
                
                # Add hierarchical labels (years and months)
                self._add_hierarchical_labels(ax, position_info, unique_year_months)
                
                logger.info(f"Plotted AWS seasonal data: {len(unique_year_months)} time periods")
                
        except Exception as e:
            logger.error(f"Error plotting AWS seasonal boxplots: {e}")

    def generate_all_plots(self, df: pd.DataFrame, output_dir: Path) -> None:
        """Generate all 8 comparative visualization types (including maps and seasonal analysis)."""
        logger.info("Starting generation of all comparative plots...")
        
        if df.empty:
            logger.error("No data available for plotting")
            return
        
        try:
            # Generate all plots 
            self.plot_method_performance_matrix(df, output_dir)
            self.plot_cross_glacier_scatterplot_matrix(df, output_dir)
            self.plot_regional_comparison_boxplots(df, output_dir)
            self.plot_sample_size_vs_performance(df, output_dir)
            self.plot_bias_comparison_radar(df, output_dir)
            self.plot_statistical_confidence_dashboard(df, output_dir)  # Combined with sample sizes
            self.plot_glacier_maps(df, output_dir)  # Spatial mapping suite
            self.plot_seasonal_analysis_multi_glacier(df, output_dir)  # Seasonal boxplot analysis
            self.plot_seasonal_analysis_time_series(df, output_dir)  # NEW: Seasonal time series analysis
            
            logger.info("Successfully generated all 9 comparative visualization types")
            
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