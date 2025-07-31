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
        self.plot_config = self.viz_config.get('plot_output', {})
        self.colors = self.viz_config.get('colors', {
            'MOD09GA': '#1f77b4',
            'MOD10A1': '#ff7f0e', 
            'MCD43A3': '#2ca02c',
            'AWS': '#d62728'
        })
        self.figure_size = self.viz_config.get('figure_size', [10, 8])
        self.dpi = self.viz_config.get('dpi', 300)
        
        # Plot generation flags
        self.eliminate_redundancy = self.plot_config.get('eliminate_redundancy', True)
        self.generate_individual = self.plot_config.get('individual_plots', True)
        self.generate_dashboard = self.plot_config.get('dashboard_plot', True)
        
        # Track which box plots have been generated to avoid redundancy
        self._generated_boxplots = set()
        
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
            # Use local statistical calculation instead of importing analyzer
            def calculate_basic_metrics(observed, predicted):
                """Calculate basic statistical metrics."""
                rmse = np.sqrt(np.mean((predicted - observed) ** 2))
                bias = np.mean(predicted - observed)
                correlation_matrix = np.corrcoef(observed, predicted)
                r_value = correlation_matrix[0, 1] if not np.isnan(correlation_matrix).any() else np.nan
                r_squared = r_value ** 2 if not np.isnan(r_value) else np.nan
                n_samples = len(observed)
                return {'rmse': rmse, 'bias': bias, 'r2': r_squared, 'n_samples': n_samples}
            
            metrics = calculate_basic_metrics(x_clean, y_clean)
            
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
        # Align data on common indices and remove NaN values
        if hasattr(x_data, 'index') and hasattr(y_data, 'index'):
            # Both are pandas Series - align on common index
            aligned_data = pd.DataFrame({'x': x_data, 'y': y_data}).dropna()
            x_clean = aligned_data['x']
            y_clean = aligned_data['y']
        else:
            # Handle arrays or other data types
            import numpy as np
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
        def calculate_basic_metrics(observed, predicted):
            """Calculate basic statistical metrics."""
            rmse = np.sqrt(np.mean((predicted - observed) ** 2))
            bias = np.mean(predicted - observed)
            correlation_matrix = np.corrcoef(observed, predicted)
            r_value = correlation_matrix[0, 1] if not np.isnan(correlation_matrix).any() else np.nan
            r_squared = r_value ** 2 if not np.isnan(r_value) else np.nan
            n_samples = len(observed)
            return {'rmse': rmse, 'bias': bias, 'r2': r_squared, 'n_samples': n_samples}
            
        metrics = calculate_basic_metrics(x_clean, y_clean)
        
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
                colors.extend([self.colors.get(method_name, f'C{i}')] * len(diff_values))
        
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
                color = self.colors.get(method_name, f'C{i}')
                box.set_facecolor(color)
                box.set_alpha(0.3)
        
        # Add zero line
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1)
        
        # Customize plot
        ax.set_ylabel(f"Albedo Difference (Method - {reference_name})", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Difference scatterplot saved to {output_path}")
        
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
                    color = self.colors.get(method, f'C{list(methods).index(method)}')
                    
                    # Create date within year for x-axis (using day of year)
                    ax.plot(method_data['day_of_year'], method_data[value_column],
                           label=method, color=color, alpha=0.8, linewidth=1.5, 
                           marker='o', markersize=2)
            
            # Customize each subplot
            ax.set_title(str(year), fontweight='bold', fontsize=12)
            ax.set_ylim(y_min_plot, y_max_plot)
            ax.grid(True, alpha=0.3)
            
            # Dynamic x-axis: only show months with actual data
            if not year_data.empty:
                # Find actual data range for this year
                min_day = year_data['day_of_year'].min()
                max_day = year_data['day_of_year'].max()
                
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
            else:
                # Fallback for empty data
                ax.set_xlim(150, 270)  # Roughly Jun-Sep
            
            # Only show y-axis label on leftmost plots
            if i % cols == 0:
                ax.set_ylabel('Albedo', fontsize=10)
            
            # Only show x-axis label on bottom plots
            if i >= (rows - 1) * cols:
                ax.set_xlabel('Month', fontsize=10)
        
        # Hide unused subplots
        for j in range(len(years), len(axes)):
            axes[j].set_visible(False)
        
        # Add legend to the figure (not individual subplots to save space)
        if methods:
            # Create legend handles
            legend_handles = []
            for method in methods:
                color = self.colors.get(method, f'C{list(methods).index(method)}')
                legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=2, 
                                               marker='o', markersize=4, label=method))
            
            fig.legend(legend_handles, methods, loc='upper right', bbox_to_anchor=(0.98, 0.95))
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92, right=0.92)
        
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
        from analysis.core.statistical_analyzer import StatisticalAnalyzer
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
    
    def should_include_boxplot(self, plot_type: str, content_type: str) -> bool:
        """Check if a box plot should be included based on redundancy elimination settings."""
        if not self.eliminate_redundancy:
            return True
        
        boxplot_key = f"{plot_type}_{content_type}"
        
        # If seasonal analysis with monthly box plots is included, 
        # don't include general method box plots in other analyses
        if content_type == "method_boxplots" and "seasonal_monthly_boxplots" in self._generated_boxplots:
            logger.info(f"Skipping {boxplot_key} - redundant with seasonal analysis")
            return False
        
        # If this specific box plot type hasn't been generated yet, allow it
        if boxplot_key not in self._generated_boxplots:
            self._generated_boxplots.add(boxplot_key)
            return True
        
        # Already generated this type of box plot
        logger.info(f"Skipping duplicate box plot: {boxplot_key}")
        return False
    
    def create_refined_distribution_analysis(self, data: Dict[str, pd.Series],
                                           glacier_name: str = "Glacier",
                                           output_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Create distribution analysis focusing on histograms and avoiding redundant box plots."""
        if not self.plot_config.get('include_distribution_analysis', True):
            return None
        
        logger.info("Creating refined distribution analysis (histograms only)")
        
        # Create figure with histogram focus
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{glacier_name} - Distribution Analysis', fontsize=16, fontweight='bold')
        
        methods = list(data.keys())
        colors = [self.colors.get(method, 'gray') for method in methods]
        
        # Create histograms for each method
        for i, (method, series) in enumerate(data.items()):
            ax = axes[i // 2, i % 2]
            clean_data = series.dropna()
            
            if len(clean_data) > 0:
                # Histogram with statistics
                ax.hist(clean_data, bins=30, alpha=0.7, color=colors[i], edgecolor='black')
                ax.set_title(f'{method} Distribution (n={len(clean_data)})')
                ax.set_xlabel('Albedo')
                ax.set_ylabel('Frequency')
                
                # Add statistical annotations
                mean_val = clean_data.mean()
                std_val = clean_data.std()
                median_val = clean_data.median()
                
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
                ax.axvline(median_val, color='orange', linestyle=':', alpha=0.8, label=f'Median: {median_val:.3f}')
                
                ax.text(0.05, 0.95, f'σ = {std_val:.3f}', transform=ax.transAxes, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                       verticalalignment='top')
                
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplot if needed
        if len(methods) < 4:
            for i in range(len(methods), 4):
                axes[i // 2, i % 2].set_visible(False)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Refined distribution analysis saved to {output_path}")
        
        return fig
    
    def create_refined_method_comparison(self, aws_data: pd.Series, 
                                       modis_methods: Dict[str, pd.Series],
                                       glacier_name: str = "Glacier",
                                       output_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Create method comparison focusing on scatter plots without redundant box plots."""
        if not self.plot_config.get('include_method_comparison', True):
            return None
        
        logger.info("Creating refined method comparison (scatter plots only)")
        
        # Use existing scatter plot functionality but avoid box plots
        return self.create_multi_method_scatterplot(
            aws_data, modis_methods, 
            title=f"{glacier_name} - MODIS vs AWS Comparison",
            output_path=output_path
        )
    
    def create_comprehensive_dashboard(self, aws_data: pd.Series, 
                                     modis_methods: Dict[str, pd.Series],
                                     statistics: Dict[str, Any],
                                     glacier_name: str = "Glacier",
                                     output_path: Optional[str] = None) -> Optional[plt.Figure]:
        """Create comprehensive dashboard with 6 panels maintaining exact visual style."""
        if not self.generate_dashboard:
            return None
        
        logger.info("Creating comprehensive dashboard")
        
        # Dashboard configuration
        dashboard_size = self.plot_config.get('dashboard_size', [18, 12])
        
        # Create figure with 2x3 grid
        fig = plt.figure(figsize=dashboard_size)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Panel 1: Best method scatter plot (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        best_method = self._get_best_method(statistics)
        if best_method and best_method in modis_methods:
            self._create_single_scatterplot_axis(
                ax1, modis_methods[best_method], aws_data, best_method,
                f"{best_method} vs AWS", "MODIS Albedo", "AWS Albedo"
            )
        
        # Panel 2: Time series overview (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        self._create_dashboard_time_series(ax2, aws_data, modis_methods)
        
        # Panel 3: Seasonal summary (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        self._create_dashboard_seasonal_summary(ax3, aws_data, modis_methods)
        
        # Panel 4: Statistical summary (bottom-left)
        ax4 = fig.add_subplot(gs[1, 0])
        self._create_dashboard_statistics_table(ax4, statistics)
        
        # Panel 5: Distribution overview (bottom-middle)
        ax5 = fig.add_subplot(gs[1, 1])
        self._create_dashboard_distribution_summary(ax5, modis_methods)
        
        # Panel 6: Performance metrics (bottom-right)
        ax6 = fig.add_subplot(gs[1, 2])
        self._create_dashboard_performance_summary(ax6, statistics)
        
        # Overall title
        fig.suptitle(f'{glacier_name} - Comprehensive Analysis Dashboard', 
                    fontsize=18, fontweight='bold', y=0.95)
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Comprehensive dashboard saved to {output_path}")
        
        return fig
    
    def create_original_outlier_analysis(self, aws_data: pd.Series, 
                                       modis_methods: Dict[str, pd.Series],
                                       glacier_name: str = "Glacier",
                                       outlier_threshold: float = 2.5,
                                       output_path: Optional[str] = None) -> plt.Figure:
        """Create original 6-panel outlier analysis (before/after outlier removal)."""
        logger.info("Creating original outlier analysis (6-panel)")
        
        # Create figure with 2x3 grid (2 rows, 3 columns)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{glacier_name.upper()} Glacier - Outlier Analysis ({outlier_threshold}σ threshold)', 
                    fontsize=16, fontweight='bold')
        
        methods = list(modis_methods.keys())[:3]  # Limit to 3 methods
        
        for i, method in enumerate(methods):
            modis_data = modis_methods[method]
            
            # Find common indices
            common_idx = aws_data.index.intersection(modis_data.index)
            if len(common_idx) == 0:
                continue
                
            aws_aligned = aws_data.loc[common_idx]
            modis_aligned = modis_data.loc[common_idx]
            
            # Remove initial NaN pairs
            mask = ~(aws_aligned.isna() | modis_aligned.isna())
            aws_clean = aws_aligned[mask]
            modis_clean = modis_aligned[mask]
            
            if len(aws_clean) == 0:
                continue
            
            # Calculate outliers using residuals
            residuals = modis_clean - aws_clean
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            threshold = outlier_threshold * std_residual
            
            # Identify outliers
            outlier_mask = np.abs(residuals - mean_residual) > threshold
            outliers_aws = aws_clean[outlier_mask]
            outliers_modis = modis_clean[outlier_mask]
            
            # Clean data (after outlier removal)
            clean_mask = ~outlier_mask
            aws_final = aws_clean[clean_mask]
            modis_final = modis_clean[clean_mask]
            
            # Calculate statistics before outlier removal
            def calc_metrics(obs, pred):
                r = np.corrcoef(obs, pred)[0, 1] if len(obs) > 1 else np.nan
                rmse = np.sqrt(np.mean((pred - obs) ** 2))
                bias = np.mean(pred - obs)
                mae = np.mean(np.abs(pred - obs))
                return r, rmse, bias, mae, len(obs)
            
            r_before, rmse_before, bias_before, mae_before, n_before = calc_metrics(aws_clean, modis_clean)
            r_after, rmse_after, bias_after, mae_after, n_after = calc_metrics(aws_final, modis_final)
            
            # Top row: Before outlier removal
            ax_before = axes[0, i]
            color = self.colors.get(method, f'C{i}')
            
            # Scatter plot before
            ax_before.scatter(aws_clean, modis_clean, alpha=0.6, s=20, color=color, 
                            edgecolors='black', linewidth=0.3, label='Data points')
            
            # Highlight outliers
            if len(outliers_aws) > 0:
                ax_before.scatter(outliers_aws, outliers_modis, color='red', marker='x', 
                                s=50, linewidth=2, label=f'Outliers (n={len(outliers_aws)})')
            
            # 1:1 line
            min_val = min(aws_clean.min(), modis_clean.min())
            max_val = max(aws_clean.max(), modis_clean.max())
            ax_before.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.7, linewidth=1)
            
            # Regression line
            if len(aws_clean) > 1:
                z = np.polyfit(aws_clean, modis_clean, 1)
                p = np.poly1d(z)
                ax_before.plot(aws_clean, p(aws_clean), 'black', alpha=0.8, linewidth=1.5)
            
            # Statistics box before
            stats_text_before = (f'r={r_before:.3f}\n'
                               f'RMSE={rmse_before:.3f}\n'
                               f'n={n_before}\n'
                               f'MAE={mae_before:.3f}\n'
                               f'Bias={bias_before:.3f}')
            
            ax_before.text(0.05, 0.95, stats_text_before, transform=ax_before.transAxes,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                         fontsize=9)
            
            ax_before.set_title(f'{method} - Before Outlier Removal')
            ax_before.set_ylabel(f'{method} Albedo')
            ax_before.grid(True, alpha=0.3)
            ax_before.set_xlim(0, 1)
            ax_before.set_ylim(0, 1)
            
            # Bottom row: After outlier removal
            ax_after = axes[1, i]
            
            # Scatter plot after
            ax_after.scatter(aws_final, modis_final, alpha=0.6, s=20, color=color,
                           edgecolors='black', linewidth=0.3)
            
            # 1:1 line
            ax_after.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=1)
            
            # Regression line
            if len(aws_final) > 1:
                z = np.polyfit(aws_final, modis_final, 1)
                p = np.poly1d(z)
                ax_after.plot(aws_final, p(aws_final), 'black', alpha=0.8, linewidth=1.5)
            
            # Calculate improvement percentage
            outlier_percent = (len(outliers_aws) / len(aws_clean)) * 100 if len(aws_clean) > 0 else 0
            
            # Statistics box after
            stats_text_after = (f'r={r_after:.3f}\n'
                              f'RMSE={rmse_after:.3f}\n'
                              f'n={n_after}\n'
                              f'MAE={mae_after:.3f}\n'
                              f'Bias={bias_after:.3f}\n'
                              f'Δr={outlier_percent:.1f}%')
            
            ax_after.text(0.05, 0.95, stats_text_after, transform=ax_after.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8),
                        fontsize=9)
            
            ax_after.set_title(f'{method} - After Outlier Removal')
            ax_after.set_xlabel('AWS Albedo')
            ax_after.set_ylabel(f'{method} Albedo')
            ax_after.grid(True, alpha=0.3)
            ax_after.set_xlim(0, 1)
            ax_after.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Original outlier analysis saved to {output_path}")
        
        return fig
    
    def create_original_comprehensive_summary(self, aws_data: pd.Series, 
                                            modis_methods: Dict[str, pd.Series],
                                            glacier_name: str = "Glacier",
                                            output_path: Optional[str] = None) -> plt.Figure:
        """Create original comprehensive summary figure (4-panel + performance table + 2 boxplots)."""
        logger.info("Creating original comprehensive summary figure")
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 2, 1.2], hspace=0.3, wspace=0.3)
        
        fig.suptitle(f'{glacier_name.upper()} Glacier - Comprehensive Albedo Analysis Summary', 
                    fontsize=16, fontweight='bold')
        
        methods = list(modis_methods.keys())[:3]  # Limit to 3 methods
        
        # Top row: 3 scatter plots + performance table
        for i, method in enumerate(methods):
            ax = fig.add_subplot(gs[0, i])
            modis_data = modis_methods[method]
            
            # Find common indices and clean data
            common_idx = aws_data.index.intersection(modis_data.index)
            if len(common_idx) > 0:
                aws_aligned = aws_data.loc[common_idx].dropna()
                modis_aligned = modis_data.loc[common_idx].dropna()
                
                # Further align on common indices after dropna
                final_common = aws_aligned.index.intersection(modis_aligned.index)
                if len(final_common) > 0:
                    aws_clean = aws_aligned.loc[final_common]
                    modis_clean = modis_aligned.loc[final_common]
                    
                    color = self.colors.get(method, f'C{i}')
                    
                    # Scatter plot
                    ax.scatter(modis_clean, aws_clean, alpha=0.6, s=15, color=color)
                    
                    # 1:1 line
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=1)
                    
                    # Regression line
                    if len(aws_clean) > 1:
                        z = np.polyfit(modis_clean, aws_clean, 1)
                        p = np.poly1d(z)
                        ax.plot(modis_clean, p(modis_clean), 'red', alpha=0.8, linewidth=1.5)
                    
                    # Calculate metrics
                    r = np.corrcoef(modis_clean, aws_clean)[0, 1] if len(aws_clean) > 1 else np.nan
                    rmse = np.sqrt(np.mean((modis_clean - aws_clean) ** 2))
                    bias = np.mean(modis_clean - aws_clean)
                    
                    # Statistics text
                    stats_text = (f'r²={r**2:.3f}\n'
                                f'RMSE={rmse:.3f}\n'
                                f'n={len(aws_clean)}')
                    
                    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                           fontsize=10)
            
            ax.set_title(f'{method} vs AWS')
            ax.set_xlabel(f'{method} Albedo')
            ax.set_ylabel('AWS Albedo')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        
        # Performance Summary Table (top right)
        ax_table = fig.add_subplot(gs[0, 3])
        ax_table.axis('off')
        
        # Calculate performance metrics for table
        table_data = []
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
                    
                    r = np.corrcoef(modis_clean, aws_clean)[0, 1] if len(aws_clean) > 1 else np.nan
                    rmse = np.sqrt(np.mean((modis_clean - aws_clean) ** 2))
                    bias = np.mean(modis_clean - aws_clean)
                    
                    table_data.append([method, f'{r:.3f}', f'{rmse:.3f}', f'{bias:.3f}', str(len(aws_clean))])
        
        if table_data:
            table = ax_table.table(cellText=table_data,
                                 colLabels=['Method', 'R', 'RMSE', 'Bias', 'n'],
                                 cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
        
        ax_table.set_title('Performance Summary', fontweight='bold')
        
        # Middle row: 4 metric boxplots (Correlation, Bias, MAE, RMSE)
        # Collect all metrics for each method
        metrics_data = {
            'correlation': [],
            'bias': [],
            'mae': [],
            'rmse': []
        }
        method_labels = []
        
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
                    
                    metrics_data['correlation'].append(r)
                    metrics_data['bias'].append(bias)
                    metrics_data['mae'].append(mae)
                    metrics_data['rmse'].append(rmse)
                    method_labels.append(method)
        
        # Create 4 boxplots in middle row
        metric_titles = ['Correlation (R)', 'Bias', 'MAE', 'RMSE']
        metric_keys = ['correlation', 'bias', 'mae', 'rmse']
        y_labels = ['Correlation', 'Albedo Bias (MODIS - AWS)', 'Mean Absolute Error', 'Root Mean Square Error']
        
        for i, (metric_key, metric_title, y_label) in enumerate(zip(metric_keys, metric_titles, y_labels)):
            ax = fig.add_subplot(gs[1, i])
            
            if metrics_data[metric_key]:
                # Create individual metric values for each method
                plot_data = []
                plot_labels = []
                colors = []
                
                for j, method in enumerate(method_labels):
                    if j < len(metrics_data[metric_key]):
                        # Create a list with single value for boxplot
                        plot_data.append([metrics_data[metric_key][j]])
                        plot_labels.append(method)
                        colors.append(self.colors.get(method, f'C{j}'))
                
                if plot_data:
                    box_plot = ax.boxplot(plot_data, labels=plot_labels, patch_artist=True)
                    for box, color in zip(box_plot['boxes'], colors):
                        box.set_facecolor(color)
                        box.set_alpha(0.7)
                    
                    # Add zero reference line for bias
                    if metric_key == 'bias':
                        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8)
            
            ax.set_title(metric_title)
            ax.set_ylabel(y_label)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Original comprehensive summary saved to {output_path}")
        
        return fig
    
    def create_four_metrics_boxplot_summary(self, aws_data: pd.Series, 
                                           modis_methods: Dict[str, pd.Series],
                                           glacier_name: str = "Glacier",
                                           output_path: Optional[str] = None) -> plt.Figure:
        """Create publication-ready single grouped bar chart showing performance metrics for all methods."""
        logger.info("Creating publication-ready grouped bar chart for method performance")
        
        # Set publication-quality matplotlib parameters
        plt.rcParams.update({
            'font.family': 'Arial',
            'font.size': 11,
            'axes.linewidth': 1.2,
            'grid.linewidth': 0.8,
            'lines.linewidth': 1.5
        })
        
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
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            logger.info(f"Publication-ready grouped bar chart saved to {output_path}")
        
        # Reset matplotlib parameters
        plt.rcParams.update(plt.rcParamsDefault)
        
        return fig
    
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
        ax1 = axes[0, 0]
        
        # Create correlation matrix
        corr_df = pd.DataFrame(correlation_data).corr()
        
        # Create heatmap
        im = ax1.imshow(corr_df.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        
        # Add correlation values as text
        for i in range(len(corr_df)):
            for j in range(len(corr_df.columns)):
                text = ax1.text(j, i, f'{corr_df.iloc[i, j]:.3f}',
                              ha="center", va="center", color="black", fontsize=10, fontweight='bold')
        
        ax1.set_xticks(range(len(corr_df.columns)))
        ax1.set_yticks(range(len(corr_df)))
        ax1.set_xticklabels(corr_df.columns)
        ax1.set_yticklabels(corr_df.index)
        ax1.set_title('Correlation Matrix')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax1, shrink=0.8)
        cbar.set_label('Correlation Coefficient')
        
        # Panel 2: Method Performance Comparison (top-right)
        ax2 = axes[0, 1]
        
        methods_list = performance_data['Method']
        x_pos = np.arange(len(methods_list))
        
        # Create grouped bar chart
        width = 0.25
        r_bars = ax2.bar(x_pos - width, performance_data['R'], width, label='R', color='lightblue', alpha=0.8)
        rmse_bars = ax2.bar(x_pos, performance_data['RMSE'], width, label='RMSE', color='lightcoral', alpha=0.8)
        bias_bars = ax2.bar(x_pos + width, [abs(b) for b in performance_data['BIAS']], width, label='|BIAS|', color='lightgreen', alpha=0.8)
        
        ax2.set_xlabel('Methods')
        ax2.set_ylabel('Metric Value')
        ax2.set_title('Method Performance Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(methods_list)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Panel 3: Bias vs. AWS Albedo (bottom-left)
        ax3 = axes[1, 0]
        
        method_colors = [self.colors.get(method, f'C{i}') for i, method in enumerate(methods)]
        unique_methods = list(set(bias_scatter_data['method']))
        
        for i, method in enumerate(unique_methods):
            method_mask = [m == method for m in bias_scatter_data['method']]
            aws_vals = [bias_scatter_data['aws'][j] for j in range(len(method_mask)) if method_mask[j]]
            bias_vals = [bias_scatter_data['bias'][j] for j in range(len(method_mask)) if method_mask[j]]
            
            color = self.colors.get(method, f'C{i}')
            ax3.scatter(aws_vals, bias_vals, alpha=0.6, s=15, color=color, label=method)
        
        ax3.axhline(y=0, color='red', linestyle='--', alpha=0.8)
        ax3.set_xlabel('AWS Albedo')
        ax3.set_ylabel('Bias (MODIS - AWS)')
        ax3.set_title('Bias vs. AWS Albedo')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: RMSE vs. Sample Size (bottom-right)
        ax4 = axes[1, 1]
        
        for i, method in enumerate(rmse_sample_data['method']):
            color = self.colors.get(method, f'C{i}')
            ax4.scatter(rmse_sample_data['n_samples'][i], rmse_sample_data['rmse'][i], 
                       s=100, alpha=0.7, color=color, label=method)
            
            # Add method label next to point
            ax4.annotate(method, (rmse_sample_data['n_samples'][i], rmse_sample_data['rmse'][i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax4.set_xlabel('Sample Size')
        ax4.set_ylabel('RMSE')
        ax4.set_title('RMSE vs. Sample Size')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Original correlation and bias analysis saved to {output_path}")
        
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
                        color = self.colors.get(method, f'C{j}')
                        box_plot['boxes'][j].set_facecolor(color)
                        box_plot['boxes'][j].set_alpha(0.7)
            
            ax.set_title(month_name, fontweight='bold')
            ax.set_ylabel('Albedo')
            ax.grid(True, alpha=0.3)
            
            # Set consistent y-axis limits
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Original seasonal analysis saved to {output_path}")
        
        return fig
    
    def _get_best_method(self, statistics: Dict[str, Any]) -> Optional[str]:
        """Get the best performing method based on statistics."""
        if 'method_comparison' not in statistics:
            return None
        
        best_method = None
        best_r2 = -1
        
        for method, metrics in statistics['method_comparison'].items():
            if metrics.get('r_squared', 0) > best_r2:
                best_r2 = metrics['r_squared']
                best_method = method
        
        return best_method
    
    def _create_dashboard_time_series(self, ax, aws_data: pd.Series, modis_methods: Dict[str, pd.Series]):
        """Create condensed time series for dashboard."""
        # Simple time series plot
        if hasattr(aws_data, 'index') and hasattr(aws_data.index, 'to_series'):
            dates = pd.to_datetime(aws_data.index)
            ax.scatter(dates, aws_data.values, alpha=0.6, color=self.colors.get('AWS', 'red'), 
                      s=10, label='AWS')
            
            for method, data in modis_methods.items():
                if len(data) > 0:
                    method_dates = pd.to_datetime(data.index)
                    ax.scatter(method_dates, data.values, alpha=0.4, 
                             color=self.colors.get(method, 'gray'), s=8, label=method)
        
        ax.set_title('Temporal Overview')
        ax.set_xlabel('Date')
        ax.set_ylabel('Albedo')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _create_dashboard_seasonal_summary(self, ax, aws_data: pd.Series, modis_methods: Dict[str, pd.Series]):
        """Create condensed seasonal summary for dashboard."""
        # Simple seasonal box plot (condensed version)
        try:
            # Create monthly averages
            monthly_data = []
            labels = []
            
            # Combine AWS and best method for seasonal comparison
            all_data = {'AWS': aws_data}
            all_data.update(modis_methods)
            
            for method, data in all_data.items():
                if len(data) > 0:
                    monthly_data.append(data.values)
                    labels.append(method)
            
            if monthly_data:
                box_plot = ax.boxplot(monthly_data, labels=labels, patch_artist=True)
                for i, box in enumerate(box_plot['boxes']):
                    method = labels[i]
                    box.set_facecolor(self.colors.get(method, 'gray'))
                    box.set_alpha(0.7)
        
        except Exception as e:
            ax.text(0.5, 0.5, 'Seasonal data\nnot available', 
                   transform=ax.transAxes, ha='center', va='center')
        
        ax.set_title('Method Comparison')
        ax.set_ylabel('Albedo')
        ax.grid(True, alpha=0.3)
    
    def _create_dashboard_statistics_table(self, ax, statistics: Dict[str, Any]):
        """Create statistics table for dashboard."""
        ax.axis('off')
        
        if 'method_comparison' in statistics:
            table_data = []
            for method, metrics in statistics['method_comparison'].items():
                table_data.append([
                    method,
                    f"{metrics.get('r', 0):.3f}",
                    f"{metrics.get('rmse', 0):.3f}",
                    f"{metrics.get('n_samples', 0)}"
                ])
            
            if table_data:
                table = ax.table(cellText=table_data,
                               colLabels=['Method', 'R', 'RMSE', 'n'],
                               cellLoc='center',
                               loc='center')
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1, 1.5)
        
        ax.set_title('Statistical Summary')
    
    def _create_dashboard_distribution_summary(self, ax, modis_methods: Dict[str, pd.Series]):
        """Create distribution summary for dashboard."""
        # Overlapping histograms
        for method, data in modis_methods.items():
            clean_data = data.dropna()
            if len(clean_data) > 0:
                ax.hist(clean_data, bins=20, alpha=0.5, 
                       color=self.colors.get(method, 'gray'), 
                       label=f'{method} (n={len(clean_data)})')
        
        ax.set_title('Distribution Overview')
        ax.set_xlabel('Albedo')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    def _create_dashboard_performance_summary(self, ax, statistics: Dict[str, Any]):
        """Create performance summary for dashboard."""
        if 'method_comparison' not in statistics:
            ax.text(0.5, 0.5, 'Performance\ndata not available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title('Performance Summary')
            return
        
        methods = list(statistics['method_comparison'].keys())
        r_values = [statistics['method_comparison'][m].get('r', 0) for m in methods]
        colors = [self.colors.get(method, 'gray') for method in methods]
        
        bars = ax.bar(methods, r_values, color=colors, alpha=0.7)
        ax.set_title('Correlation Performance')
        ax.set_ylabel('Correlation (R)')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, r_val in zip(bars, r_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{r_val:.3f}', ha='center', va='bottom', fontsize=8)