#!/usr/bin/env python3
"""
Comprehensive plot implementations for albedo analysis.

This module contains comprehensive dashboard-style plots and summary
visualizations that combine multiple analysis types.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional
from .base import BasePlotter

logger = logging.getLogger(__name__)


class ComprehensivePlotter(BasePlotter):
    """Specialized plotter for comprehensive dashboard-style visualizations."""
    
    def create_comprehensive_analysis(self, aws_data: pd.Series, 
                                    modis_methods: Dict[str, pd.Series],
                                    glacier_name: str = "Glacier",
                                    output_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive analysis dashboard."""
        logger.info("Creating comprehensive analysis dashboard")
        
        if not modis_methods:
            logger.error("No MODIS methods provided for comprehensive analysis")
            return None
        
        # Create figure with 3x2 grid
        fig, axes = plt.subplots(3, 2, figsize=(16, 18))
        fig.suptitle(f'{glacier_name.upper()} Glacier - Comprehensive Analysis Dashboard', 
                    fontsize=18, fontweight='bold')
        
        methods = list(modis_methods.keys())
        
        # Panel 1: Method Performance Comparison (top-left)
        self._create_performance_summary_panel(axes[0, 0], aws_data, modis_methods)
        
        # Panel 2: Correlation Matrix (top-right)
        correlation_data = {'AWS': aws_data}
        correlation_data.update(modis_methods)
        self._create_correlation_heatmap_panel(axes[0, 1], correlation_data)
        
        # Panel 3: Scatter Plot Matrix (middle-left)
        self._create_scatter_matrix_panel(axes[1, 0], aws_data, modis_methods)
        
        # Panel 4: Distribution Comparison (middle-right)
        all_data = {'AWS': aws_data}
        all_data.update(modis_methods)
        self._create_distribution_panel(axes[1, 1], all_data)
        
        # Panel 5: Bias Analysis (bottom-left)
        self._create_bias_analysis_panel(axes[2, 0], aws_data, modis_methods)
        
        # Panel 6: Summary Statistics Table (bottom-right)
        self._create_statistics_table_panel(axes[2, 1], aws_data, modis_methods)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.94)
        
        # Save figure
        self._save_figure(fig, output_path, "Comprehensive analysis dashboard")
        
        return fig
    
    def _create_performance_summary_panel(self, ax, aws_data, modis_methods):
        """Create performance summary panel."""
        methods = list(modis_methods.keys())
        metrics = {'correlation': [], 'rmse': [], 'bias': [], 'mae': []}
        valid_methods = []
        
        for method in methods:
            modis_data = modis_methods[method]
            aws_clean, modis_clean = self._clean_and_align_data(aws_data, modis_data)
            
            if len(aws_clean) > 0:
                # Calculate metrics
                r = np.corrcoef(modis_clean, aws_clean)[0, 1] if len(aws_clean) > 1 else np.nan
                rmse = np.sqrt(np.mean((modis_clean - aws_clean) ** 2))
                bias = np.mean(modis_clean - aws_clean)
                mae = np.mean(np.abs(modis_clean - aws_clean))
                
                metrics['correlation'].append(r)
                metrics['rmse'].append(rmse)
                metrics['bias'].append(abs(bias))  # Use absolute bias for comparison
                metrics['mae'].append(mae)
                valid_methods.append(method)
        
        if not valid_methods:
            ax.text(0.5, 0.5, 'No valid data for performance summary', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Performance Summary')
            return
        
        # Create grouped bar chart
        x = np.arange(len(valid_methods))
        width = 0.2
        
        ax.bar(x - 1.5*width, metrics['correlation'], width, label='Correlation', alpha=0.8, color='#1f77b4')
        ax.bar(x - 0.5*width, metrics['rmse'], width, label='RMSE', alpha=0.8, color='#ff7f0e')
        ax.bar(x + 0.5*width, metrics['bias'], width, label='|Bias|', alpha=0.8, color='#2ca02c')
        ax.bar(x + 1.5*width, metrics['mae'], width, label='MAE', alpha=0.8, color='#d62728')
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Metric Value')
        ax.set_title('Performance Metrics Summary')
        ax.set_xticks(x)
        ax.set_xticklabels(valid_methods)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_correlation_heatmap_panel(self, ax, correlation_data):
        """Create correlation matrix heatmap panel."""
        try:
            # Create correlation matrix
            corr_df = pd.DataFrame(correlation_data).corr()
            
            # Create heatmap
            im = ax.imshow(corr_df.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
            
            # Add correlation values as text
            for i in range(len(corr_df)):
                for j in range(len(corr_df.columns)):
                    text = ax.text(j, i, f'{corr_df.iloc[i, j]:.3f}',
                                  ha="center", va="center", color="black", 
                                  fontsize=10, fontweight='bold')
            
            ax.set_xticks(range(len(corr_df.columns)))
            ax.set_yticks(range(len(corr_df)))
            ax.set_xticklabels(corr_df.columns)
            ax.set_yticklabels(corr_df.index)
            ax.set_title('Inter-Method Correlation Matrix')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.8)
            cbar.set_label('Correlation Coefficient')
            
        except Exception as e:
            logger.warning(f"Could not create correlation matrix: {e}")
            ax.text(0.5, 0.5, 'Correlation matrix unavailable', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Inter-Method Correlation Matrix')
    
    def _create_scatter_matrix_panel(self, ax, aws_data, modis_methods):
        """Create scatter plot matrix panel (simplified version)."""
        methods = list(modis_methods.keys())
        
        if len(methods) == 0:
            ax.text(0.5, 0.5, 'No methods for scatter matrix', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Method Scatter Comparison')
            return
        
        # For simplicity, show scatter plot of first method vs AWS
        first_method = methods[0]
        modis_data = modis_methods[first_method]
        
        aws_clean, modis_clean = self._clean_and_align_data(aws_data, modis_data)
        
        if len(aws_clean) > 0:
            # Scatter plot
            color = self._get_method_color(first_method, 0)
            ax.scatter(aws_clean, modis_clean, alpha=0.6, color=color, s=20)
            
            # Add 1:1 line
            min_val = min(aws_clean.min(), modis_clean.min())
            max_val = max(aws_clean.max(), modis_clean.max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            # Calculate and display R²
            r = np.corrcoef(aws_clean, modis_clean)[0, 1]
            ax.text(0.05, 0.95, f'R² = {r**2:.3f}', transform=ax.transAxes, 
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            ax.set_xlabel('AWS Albedo')
            ax.set_ylabel(f'{first_method} Albedo')
            ax.set_title(f'AWS vs {first_method} Scatter')
        else:
            ax.text(0.5, 0.5, 'No valid data for scatter plot', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Method Scatter Comparison')
        
        ax.grid(True, alpha=0.3)
    
    def _create_distribution_panel(self, ax, all_data):
        """Create distribution comparison panel."""
        plot_data = []
        labels = []
        
        for method_name, values in all_data.items():
            clean_values = values.dropna()
            if len(clean_values) > 0:
                plot_data.append(clean_values)
                labels.append(f"{method_name}\n(n={len(clean_values)})")
        
        if plot_data:
            box_plot = ax.boxplot(plot_data, labels=labels, patch_artist=True)
            
            # Color boxes
            for i, (method_name, _) in enumerate(all_data.items()):
                if i < len(box_plot['boxes']):
                    color = self._get_method_color(method_name, i)
                    box_plot['boxes'][i].set_facecolor(color)
                    box_plot['boxes'][i].set_alpha(0.7)
        
        ax.set_title('Albedo Distribution Comparison')
        ax.set_ylabel('Albedo')
        ax.grid(True, alpha=0.3)
        
        # Rotate labels if needed
        if max(len(label.split('\n')[0]) for label in labels) > 8:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _create_bias_analysis_panel(self, ax, aws_data, modis_methods):
        """Create bias analysis panel."""
        methods = list(modis_methods.keys())
        biases = []
        valid_methods = []
        
        for method in methods:
            modis_data = modis_methods[method]
            aws_clean, modis_clean = self._clean_and_align_data(aws_data, modis_data)
            
            if len(aws_clean) > 0:
                bias = np.mean(modis_clean - aws_clean)
                biases.append(bias)
                valid_methods.append(method)
        
        if biases:
            colors = [self._get_method_color(method, i) for i, method in enumerate(valid_methods)]
            bars = ax.bar(valid_methods, biases, color=colors, alpha=0.7)
            
            # Add zero line
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            
            # Add value labels on bars
            for bar, bias in zip(bars, biases):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                y_pos = height + (0.001 if height >= 0 else -0.001)
                ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                       f'{bias:.3f}', ha='center', va=va, fontsize=9)
        
        ax.set_title('Method Bias Analysis')
        ax.set_ylabel('Bias (MODIS - AWS)')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _create_statistics_table_panel(self, ax, aws_data, modis_methods):
        """Create summary statistics table panel."""
        ax.axis('off')
        
        # Prepare data for table
        all_data = {'AWS': aws_data}
        all_data.update(modis_methods)
        
        table_data = []
        for method_name, values in all_data.items():
            clean_values = values.dropna()
            if len(clean_values) > 0:
                table_data.append([
                    method_name,
                    f"{clean_values.mean():.4f}",
                    f"{clean_values.std():.4f}",
                    f"{clean_values.median():.4f}",
                    f"{clean_values.min():.4f}",
                    f"{clean_values.max():.4f}",
                    f"{len(clean_values)}"
                ])
        
        if table_data:
            table = ax.table(cellText=table_data,
                           colLabels=['Method', 'Mean', 'Std', 'Median', 'Min', 'Max', 'Count'],
                           cellLoc='center',
                           loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 1.8)
            
            # Style header
            for i in range(len(table_data[0])):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    def create_method_comparison_dashboard(self, aws_data: pd.Series, 
                                         modis_methods: Dict[str, pd.Series],
                                         glacier_name: str = "Glacier",
                                         output_path: Optional[str] = None) -> plt.Figure:
        """Create method comparison dashboard with advanced visualizations."""
        logger.info("Creating method comparison dashboard")
        
        if not modis_methods:
            logger.error("No MODIS methods provided for dashboard")
            return None
        
        # Create figure with 2x3 grid
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{glacier_name.upper()} Glacier - Method Comparison Dashboard', 
                    fontsize=18, fontweight='bold')
        
        methods = list(modis_methods.keys())
        
        # Panel 1: Performance radar chart (top-left)
        self._create_performance_radar_panel(axes[0, 0], aws_data, modis_methods)
        
        # Panel 2: Error distribution (top-middle)
        self._create_error_distribution_panel(axes[0, 1], aws_data, modis_methods)
        
        # Panel 3: Temporal performance (top-right)
        self._create_temporal_performance_panel(axes[0, 2], aws_data, modis_methods)
        
        # Panel 4: Method ranking (bottom-left)
        self._create_method_ranking_panel(axes[1, 0], aws_data, modis_methods)
        
        # Panel 5: Uncertainty analysis (bottom-middle)
        self._create_uncertainty_panel(axes[1, 1], aws_data, modis_methods)
        
        # Panel 6: Data availability (bottom-right)
        self._create_data_availability_panel(axes[1, 2], aws_data, modis_methods)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.93)
        
        # Save figure
        self._save_figure(fig, output_path, "Method comparison dashboard")
        
        return fig
    
    def _create_performance_radar_panel(self, ax, aws_data, modis_methods):
        """Create performance radar chart panel."""
        try:
            import math
            
            # Calculate normalized metrics for radar chart
            method_stats = {}
            for method_name, modis_data in modis_methods.items():
                aws_clean, modis_clean = self._clean_and_align_data(aws_data, modis_data)
                if len(aws_clean) > 0:
                    metrics = self._calculate_basic_metrics(aws_clean, modis_clean)
                    method_stats[method_name] = metrics
            
            if not method_stats:
                ax.text(0.5, 0.5, 'No data for radar chart', transform=ax.transAxes, 
                       ha='center', va='center')
                ax.set_title('Performance Radar')
                return
            
            # Define metrics for radar (normalized 0-1, higher is better)
            metrics = ['correlation', 'rmse_inv', 'bias_abs_inv', 'n_samples_norm']
            metric_labels = ['Correlation', '1/RMSE', '1/|Bias|', 'Sample Size']
            
            # Normalize metrics
            all_values = {metric: [] for metric in metrics}
            for method_name, stats in method_stats.items():
                all_values['correlation'].append(abs(stats['correlation']))
                all_values['rmse_inv'].append(1 / max(stats['rmse'], 0.001))
                all_values['bias_abs_inv'].append(1 / max(abs(stats['bias']), 0.001))
                all_values['n_samples_norm'].append(stats['n_samples'])
            
            # Normalize to 0-1 scale
            normalized_stats = {}
            for method_name, stats in method_stats.items():
                normalized_stats[method_name] = {
                    'correlation': abs(stats['correlation']),
                    'rmse_inv': (1 / max(stats['rmse'], 0.001)) / max(all_values['rmse_inv']),
                    'bias_abs_inv': (1 / max(abs(stats['bias']), 0.001)) / max(all_values['bias_abs_inv']),
                    'n_samples_norm': stats['n_samples'] / max(all_values['n_samples_norm'])
                }
            
            # Convert to polar coordinates
            angles = [n / len(metrics) * 2 * math.pi for n in range(len(metrics))]
            angles += angles[:1]
            
            # Create polar subplot
            ax.remove()
            ax = plt.subplot(2, 3, 1, projection='polar')
            
            ax.set_theta_offset(math.pi / 2)
            ax.set_theta_direction(-1)
            ax.set_thetagrids([a * 180/math.pi for a in angles[:-1]], metric_labels)
            
            # Plot each method
            for i, (method_name, stats) in enumerate(normalized_stats.items()):
                values = [stats[metric] for metric in metrics]
                values += values[:1]
                
                color = self._get_method_color(method_name, i)
                ax.plot(angles, values, color=color, linewidth=2, label=method_name)
                ax.fill(angles, values, color=color, alpha=0.2)
            
            ax.set_ylim(0, 1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            ax.set_title('Performance Radar', pad=20)
            
        except Exception as e:
            logger.warning(f"Could not create radar chart: {e}")
            ax.text(0.5, 0.5, 'Radar chart unavailable', transform=ax.transAxes, 
                   ha='center', va='center')
            ax.set_title('Performance Radar')
    
    def _create_error_distribution_panel(self, ax, aws_data, modis_methods):
        """Create error distribution panel."""
        for i, (method_name, modis_data) in enumerate(modis_methods.items()):
            aws_clean, modis_clean = self._clean_and_align_data(aws_data, modis_data)
            
            if len(aws_clean) > 0:
                errors = modis_clean - aws_clean
                color = self._get_method_color(method_name, i)
                ax.hist(errors, alpha=0.6, bins=20, label=method_name, 
                       color=color, density=True)
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8)
        ax.set_xlabel('Error (MODIS - AWS)')
        ax.set_ylabel('Density')
        ax.set_title('Error Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_temporal_performance_panel(self, ax, aws_data, modis_methods):
        """Create temporal performance panel (simplified)."""
        # This is a placeholder for temporal analysis
        ax.text(0.5, 0.5, 'Temporal Performance\n(Implementation needed)', 
               transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Temporal Performance')
    
    def _create_method_ranking_panel(self, ax, aws_data, modis_methods):
        """Create method ranking panel."""
        method_scores = {}
        
        for method_name, modis_data in modis_methods.items():
            aws_clean, modis_clean = self._clean_and_align_data(aws_data, modis_data)
            
            if len(aws_clean) > 0:
                # Simple scoring: correlation - normalized(rmse + |bias|)
                r = abs(np.corrcoef(aws_clean, modis_clean)[0, 1])
                rmse = np.sqrt(np.mean((modis_clean - aws_clean) ** 2))
                bias = abs(np.mean(modis_clean - aws_clean))
                
                # Normalize RMSE and bias (assuming max reasonable values)
                rmse_norm = min(rmse / 0.2, 1)  # Normalize to 0.2 max
                bias_norm = min(bias / 0.1, 1)  # Normalize to 0.1 max
                
                score = r - 0.5 * (rmse_norm + bias_norm)
                method_scores[method_name] = max(score, 0)  # Ensure non-negative
        
        if method_scores:
            methods = list(method_scores.keys())
            scores = list(method_scores.values())
            
            colors = [self._get_method_color(method, i) for i, method in enumerate(methods)]
            bars = ax.barh(methods, scores, color=colors, alpha=0.8)
            
            # Add score labels
            for bar, score in zip(bars, scores):
                width = bar.get_width()
                ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                       f'{score:.3f}', ha='left', va='center', fontsize=9)
        
        ax.set_xlabel('Performance Score')
        ax.set_title('Method Ranking')
        ax.grid(True, alpha=0.3, axis='x')
    
    def _create_uncertainty_panel(self, ax, aws_data, modis_methods):
        """Create uncertainty analysis panel."""
        # This is a placeholder for uncertainty analysis
        ax.text(0.5, 0.5, 'Uncertainty Analysis\n(Implementation needed)', 
               transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Uncertainty Analysis')
    
    def _create_data_availability_panel(self, ax, aws_data, modis_methods):
        """Create data availability panel."""
        methods = ['AWS'] + list(modis_methods.keys())
        availability = []
        
        # AWS availability
        availability.append(len(aws_data.dropna()))
        
        # MODIS methods availability
        for method_name, modis_data in modis_methods.items():
            availability.append(len(modis_data.dropna()))
        
        colors = [self._get_method_color(method, i) for i, method in enumerate(methods)]
        bars = ax.bar(methods, availability, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, count in zip(bars, availability):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                   f'{count}', ha='center', va='bottom', fontsize=9)
        
        ax.set_ylabel('Number of Observations')
        ax.set_title('Data Availability')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Rotate labels if needed
        if max(len(method) for method in methods) > 8:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')