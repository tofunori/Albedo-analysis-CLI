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


class PlotGenerator:
    """Generate standardized plots for albedo analysis."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.viz_config = config.get('visualization', {})
        self.colors = self.viz_config.get('colors', {
            'MOD09GA': '#1f77b4',
            'MOD10A1': '#ff7f0e', 
            'MCD43A3': '#2ca02c',
            'AWS': '#d62728'
        })
        self.figure_size = self.viz_config.get('figure_size', [10, 8])
        self.dpi = self.viz_config.get('dpi', 300)
        
        # Set plotting style
        style = self.viz_config.get('style', 'seaborn-v0_8')
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not available, using default")
    
    def create_scatterplot(self, x_data: pd.Series, y_data: pd.Series,
                          x_label: str = "AWS Albedo", y_label: str = "MODIS Albedo",
                          title: str = "MODIS vs AWS Albedo Comparison",
                          method_name: str = "MODIS",
                          show_stats: bool = True,
                          add_regression_line: bool = True,
                          add_identity_line: bool = True,
                          output_path: Optional[str] = None) -> plt.Figure:
        """Create scatterplot comparing two albedo datasets."""
        
        # Remove NaN values
        mask = ~(x_data.isna() | y_data.isna())
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
        if len(x_clean) == 0:
            logger.error("No valid data points for scatterplot")
            return None
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Get color for method
        color = self.colors.get(method_name, '#1f77b4')
        
        # Create scatter plot
        scatter = ax.scatter(x_clean, y_clean, alpha=0.6, s=50, 
                           c=color, edgecolors='black', linewidth=0.5,
                           label=f'{method_name} (n={len(x_clean)})')
        
        # Add identity line (1:1)
        if add_identity_line:
            min_val = min(x_clean.min(), y_clean.min())
            max_val = max(x_clean.max(), y_clean.max())
            ax.plot([min_val, max_val], [min_val, max_val], 
                   'k--', alpha=0.8, linewidth=2, label='1:1 line')
        
        # Add regression line
        if add_regression_line and len(x_clean) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_clean, y_clean)
            line_x = np.array([x_clean.min(), x_clean.max()])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, 'r-', alpha=0.8, linewidth=2, 
                   label=f'Regression (R²={r_value**2:.3f})')
        
        # Calculate and display statistics
        if show_stats and len(x_clean) > 1:
            from ..analysis.statistical_analysis import StatisticalAnalyzer
            analyzer = StatisticalAnalyzer(self.config)
            metrics = analyzer.calculate_basic_metrics(x_clean, y_clean)
            
            # Create statistics text box
            stats_text = f'RMSE: {metrics["rmse"]:.3f}\n'
            stats_text += f'Bias: {metrics["bias"]:.3f}\n'
            stats_text += f'R²: {metrics["r2"]:.3f}\n'
            stats_text += f'n: {metrics["n_samples"]}'
            
            # Position text box
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round',
                   facecolor='white', alpha=0.8), fontsize=10)
        
        # Customize plot
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel(y_label, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set equal aspect ratio and limits
        ax.set_aspect('equal', adjustable='box')
        all_values = np.concatenate([x_clean, y_clean])
        margin = (all_values.max() - all_values.min()) * 0.05
        ax.set_xlim(all_values.min() - margin, all_values.max() + margin)
        ax.set_ylim(all_values.min() - margin, all_values.max() + margin)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Scatterplot saved to {output_path}")
        
        return fig
    
    def create_multi_method_scatterplot(self, aws_data: pd.Series, 
                                      modis_methods: Dict[str, pd.Series],
                                      title: str = "Multi-Method MODIS vs AWS Comparison",
                                      output_path: Optional[str] = None) -> plt.Figure:
        """Create scatterplot comparing multiple MODIS methods against AWS."""
        
        fig, axes = plt.subplots(1, len(modis_methods), 
                                figsize=(self.figure_size[0] * len(modis_methods), self.figure_size[1]))
        
        if len(modis_methods) == 1:
            axes = [axes]
        
        for i, (method_name, modis_data) in enumerate(modis_methods.items()):
            self._create_single_scatterplot_axis(
                axes[i], aws_data, modis_data, method_name,
                f"AWS vs {method_name}", "AWS Albedo", f"{method_name} Albedo"
            )
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Multi-method scatterplot saved to {output_path}")
        
        return fig
    
    def _create_single_scatterplot_axis(self, ax, x_data, y_data, method_name, 
                                      title, x_label, y_label):
        """Helper function to create scatterplot on given axis."""
        # Remove NaN values
        mask = ~(x_data.isna() | y_data.isna())
        x_clean = x_data[mask]
        y_clean = y_data[mask]
        
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
        from ..analysis.statistical_analysis import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer(self.config)
        metrics = analyzer.calculate_basic_metrics(x_clean, y_clean)
        
        stats_text = f'RMSE: {metrics["rmse"]:.3f}\nR²: {metrics["r2"]:.3f}\nn: {metrics["n_samples"]}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)
        
        # Formatting
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
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
                color = self.colors.get(method_name, f'C{i}')
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
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Boxplot saved to {output_path}")
        
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
            output_path=output_path
        )
        
        # Add zero line
        if fig is not None:
            ax = fig.get_axes()[0]
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1)
        
        return fig
    
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
                        color = self.colors.get(method, f'C{j}')
                        box_plot['boxes'][j].set_facecolor(color)
                        box_plot['boxes'][j].set_alpha(0.7)
            
            axes[i].set_title(season)
            axes[i].set_ylabel('Albedo' if i % 2 == 0 else '')
            axes[i].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Seasonal comparison plot saved to {output_path}")
        
        return fig
    
    def create_time_series_plot(self, data: pd.DataFrame,
                              date_column: str = 'date',
                              value_column: str = 'albedo',
                              method_column: str = 'method',
                              title: str = "Albedo Time Series",
                              output_path: Optional[str] = None) -> plt.Figure:
        """Create time series plot of albedo data."""
        
        if any(col not in data.columns for col in [date_column, value_column, method_column]):
            logger.error("Required columns missing for time series plot")
            return None
        
        # Prepare data
        data_copy = data.copy()
        data_copy[date_column] = pd.to_datetime(data_copy[date_column])
        data_copy = data_copy.sort_values(date_column)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(self.figure_size[0] * 1.2, self.figure_size[1]))
        
        # Plot each method
        methods = data_copy[method_column].unique()
        
        for method in methods:
            method_data = data_copy[data_copy[method_column] == method]
            color = self.colors.get(method, None)
            
            ax.plot(method_data[date_column], method_data[value_column],
                   label=method, color=color, alpha=0.8, linewidth=2, marker='o', markersize=4)
        
        # Customize plot
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Albedo', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        fig.autofmt_xdate()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Time series plot saved to {output_path}")
        
        return fig
    
    def create_correlation_matrix(self, data: pd.DataFrame,
                                title: str = "Method Correlation Matrix",
                                output_path: Optional[str] = None) -> plt.Figure:
        """Create correlation matrix heatmap."""
        
        # Calculate correlation matrix
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figure_size)
        
        # Create heatmap
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))  # Mask upper triangle
        
        sns.heatmap(correlation_matrix, annot=True, cmap='RdBu_r', center=0,
                   square=True, mask=mask, cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {output_path}")
        
        return fig
    
    def create_summary_figure(self, aws_data: pd.Series, 
                            modis_methods: Dict[str, pd.Series],
                            glacier_name: str = "Glacier",
                            output_path: Optional[str] = None) -> plt.Figure:
        """Create comprehensive summary figure with multiple plot types."""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(self.figure_size[0] * 2, self.figure_size[1] * 2))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Scatterplots (top row)
        method_names = list(modis_methods.keys())
        for i, method_name in enumerate(method_names[:3]):  # Limit to 3 methods
            ax = fig.add_subplot(gs[0, i])
            self._create_single_scatterplot_axis(
                ax, aws_data, modis_methods[method_name], method_name,
                f"AWS vs {method_name}", "AWS", method_name
            )
        
        # 2. Distribution comparison (middle left)
        ax_box = fig.add_subplot(gs[1, :2])
        all_data = {'AWS': aws_data}
        all_data.update(modis_methods)
        
        plot_data = []
        labels = []
        colors_list = []
        
        for method_name, values in all_data.items():
            clean_values = values.dropna()
            if len(clean_values) > 0:
                plot_data.append(clean_values)
                labels.append(f"{method_name}\n(n={len(clean_values)})")
                colors_list.append(self.colors.get(method_name, 'gray'))
        
        if plot_data:
            box_plot = ax_box.boxplot(plot_data, labels=labels, patch_artist=True)
            for box, color in zip(box_plot['boxes'], colors_list):
                box.set_facecolor(color)
                box.set_alpha(0.7)
        
        ax_box.set_ylabel('Albedo')
        ax_box.set_title('Distribution Comparison')
        ax_box.grid(True, alpha=0.3)
        
        # 3. Statistics table (middle right)
        ax_stats = fig.add_subplot(gs[1, 2])
        ax_stats.axis('off')
        
        # Calculate statistics for table
        from ..analysis.statistical_analysis import StatisticalAnalyzer
        analyzer = StatisticalAnalyzer(self.config)
        
        stats_data = []
        for method_name, modis_data in modis_methods.items():
            metrics = analyzer.calculate_basic_metrics(aws_data, modis_data)
            stats_data.append([
                method_name,
                f"{metrics['rmse']:.3f}",
                f"{metrics['bias']:.3f}",
                f"{metrics['r2']:.3f}",
                f"{metrics['n_samples']}"
            ])
        
        if stats_data:
            table = ax_stats.table(cellText=stats_data,
                                 colLabels=['Method', 'RMSE', 'Bias', 'R²', 'n'],
                                 cellLoc='center',
                                 loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2)
        
        ax_stats.set_title('Performance Metrics')
        
        # 4. Difference plot (bottom)
        ax_diff = fig.add_subplot(gs[2, :])
        
        differences = {}
        for method_name, method_data in modis_methods.items():
            common_idx = aws_data.index.intersection(method_data.index)
            if len(common_idx) > 0:
                diff = method_data.loc[common_idx] - aws_data.loc[common_idx]
                differences[method_name] = diff.dropna()
        
        if differences:
            diff_plot_data = []
            diff_labels = []
            diff_colors = []
            
            for method_name, diff_values in differences.items():
                if len(diff_values) > 0:
                    diff_plot_data.append(diff_values)
                    diff_labels.append(f"{method_name}\n(n={len(diff_values)})")
                    diff_colors.append(self.colors.get(method_name, 'gray'))
            
            if diff_plot_data:
                diff_box_plot = ax_diff.boxplot(diff_plot_data, labels=diff_labels, patch_artist=True)
                for box, color in zip(diff_box_plot['boxes'], diff_colors):
                    box.set_facecolor(color)
                    box.set_alpha(0.7)
                
                ax_diff.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        
        ax_diff.set_ylabel('Albedo Difference (MODIS - AWS)')
        ax_diff.set_title('Bias Distribution')
        ax_diff.grid(True, alpha=0.3)
        
        # Overall title
        fig.suptitle(f'{glacier_name} - Albedo Analysis Summary', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Summary figure saved to {output_path}")
        
        return fig