import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.visualization.plots import PlotGenerator


class TestPlotGenerator:
    """Test plot generation functionality."""
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing."""
        return {
            'visualization': {
                'style': 'default',
                'figure_size': [8, 6],
                'dpi': 100,
                'colors': {
                    'MOD09GA': '#1f77b4',
                    'MOD10A1': '#ff7f0e',
                    'MCD43A3': '#2ca02c',
                    'AWS': '#d62728'
                }
            }
        }
    
    @pytest.fixture
    def sample_aws_data(self):
        """Sample AWS albedo data."""
        np.random.seed(42)
        return pd.Series(np.random.uniform(0.3, 0.8, 100), name='aws_albedo')
    
    @pytest.fixture
    def sample_modis_data(self, sample_aws_data):
        """Sample MODIS albedo data correlated with AWS."""
        np.random.seed(43)
        # Add some correlation and noise
        modis_values = sample_aws_data.values + np.random.normal(0.02, 0.05, len(sample_aws_data))
        return pd.Series(np.clip(modis_values, 0, 1), name='modis_albedo')
    
    @pytest.fixture
    def multiple_modis_methods(self, sample_aws_data):
        """Multiple MODIS methods for testing."""
        np.random.seed(44)
        methods = {}
        
        # Method 1: Good correlation, small bias
        methods['MOD09GA'] = pd.Series(
            sample_aws_data.values + np.random.normal(0.01, 0.03, len(sample_aws_data)),
            name='MOD09GA'
        )
        
        # Method 2: Lower correlation, larger bias
        methods['MOD10A1'] = pd.Series(
            sample_aws_data.values + np.random.normal(0.05, 0.06, len(sample_aws_data)),
            name='MOD10A1'
        )
        
        # Method 3: Negative bias
        methods['MCD43A3'] = pd.Series(
            sample_aws_data.values + np.random.normal(-0.02, 0.04, len(sample_aws_data)),
            name='MCD43A3'
        )
        
        # Clip to valid range
        for method in methods:
            methods[method] = pd.Series(np.clip(methods[method].values, 0, 1))
        
        return methods
    
    def test_plot_generator_initialization(self, config):
        """Test plot generator initialization."""
        generator = PlotGenerator(config)
        
        assert generator.config == config
        assert generator.colors == config['visualization']['colors']
        assert generator.figure_size == config['visualization']['figure_size']
        assert generator.dpi == config['visualization']['dpi']
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_scatterplot(self, mock_savefig, config, sample_aws_data, sample_modis_data):
        """Test scatterplot creation."""
        generator = PlotGenerator(config)
        
        fig = generator.create_scatterplot(
            sample_aws_data, sample_modis_data,
            x_label="AWS Albedo",
            y_label="MODIS Albedo",
            title="Test Scatterplot",
            method_name="MOD09GA",
            show_stats=True,
            add_regression_line=True,
            add_identity_line=True
        )
        
        # Check that a figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has axes
        axes = fig.get_axes()
        assert len(axes) >= 1
        
        ax = axes[0]
        
        # Check axis labels
        assert ax.get_xlabel() == "AWS Albedo"
        assert ax.get_ylabel() == "MODIS Albedo"
        assert "Test Scatterplot" in ax.get_title()
        
        # Clean up
        plt.close(fig)
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_multi_method_scatterplot(self, mock_savefig, config, sample_aws_data, multiple_modis_methods):
        """Test multi-method scatterplot creation."""
        generator = PlotGenerator(config)
        
        fig = generator.create_multi_method_scatterplot(
            sample_aws_data, multiple_modis_methods,
            title="Multi-Method Test"
        )
        
        # Check that a figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have multiple subplots (one for each method)
        axes = fig.get_axes()
        assert len(axes) == len(multiple_modis_methods)
        
        # Clean up
        plt.close(fig)
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_boxplot(self, mock_savefig, config, sample_aws_data, multiple_modis_methods):
        """Test boxplot creation."""
        generator = PlotGenerator(config)
        
        # Combine AWS and MODIS data
        all_data = {'AWS': sample_aws_data}
        all_data.update(multiple_modis_methods)
        
        fig = generator.create_boxplot(
            all_data,
            title="Distribution Test",
            y_label="Albedo"
        )
        
        # Check that a figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Check axes
        axes = fig.get_axes()
        assert len(axes) >= 1
        
        ax = axes[0]
        assert ax.get_ylabel() == "Albedo"
        assert "Distribution Test" in ax.get_title()
        
        # Clean up
        plt.close(fig)
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_difference_boxplot(self, mock_savefig, config, sample_aws_data, multiple_modis_methods):
        """Test difference boxplot creation."""
        generator = PlotGenerator(config)
        
        fig = generator.create_difference_boxplot(
            sample_aws_data, multiple_modis_methods,
            title="Difference Test",
            reference_name="AWS"
        )
        
        # Check that a figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have zero line for reference
        axes = fig.get_axes()
        assert len(axes) >= 1
        
        # Clean up
        plt.close(fig)
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_seasonal_comparison(self, mock_savefig, config):
        """Test seasonal comparison plot creation."""
        generator = PlotGenerator(config)
        
        # Create seasonal data
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        
        # Simulate different methods with seasonal patterns
        data_list = []
        
        for method in ['AWS', 'MOD09GA', 'MOD10A1']:
            # Create seasonal pattern
            day_of_year = dates.dayofyear
            seasonal_base = 0.5 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
            
            # Add method-specific bias and noise
            if method == 'AWS':
                values = seasonal_base + np.random.normal(0, 0.02, len(dates))
            else:
                bias = np.random.normal(0.01, 0.01)
                values = seasonal_base + bias + np.random.normal(0, 0.03, len(dates))
            
            method_data = pd.DataFrame({
                'date': dates,
                'albedo': np.clip(values, 0, 1),
                'method': method
            })
            
            data_list.append(method_data)
        
        data = pd.concat(data_list, ignore_index=True)
        
        fig = generator.create_seasonal_comparison(
            data,
            title="Seasonal Test"
        )
        
        # Check that a figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have 4 subplots (one for each season)
        axes = fig.get_axes()
        assert len(axes) == 4
        
        # Clean up
        plt.close(fig)
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_time_series_plot(self, mock_savefig, config):
        """Test time series plot creation."""
        generator = PlotGenerator(config)
        
        # Create time series data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        
        data_list = []
        for method in ['AWS', 'MOD09GA']:
            values = 0.5 + 0.1 * np.sin(2 * np.pi * np.arange(100) / 30) + np.random.normal(0, 0.02, 100)
            
            method_data = pd.DataFrame({
                'date': dates,
                'albedo': np.clip(values, 0, 1),
                'method': method
            })
            
            data_list.append(method_data)
        
        data = pd.concat(data_list, ignore_index=True)
        
        fig = generator.create_time_series_plot(
            data,
            title="Time Series Test"
        )
        
        # Check that a figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        axes = fig.get_axes()
        assert len(axes) >= 1
        
        ax = axes[0]
        assert ax.get_xlabel() == "Date"
        assert ax.get_ylabel() == "Albedo"
        
        # Clean up
        plt.close(fig)
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_correlation_matrix(self, mock_savefig, config, sample_aws_data, multiple_modis_methods):
        """Test correlation matrix creation."""
        generator = PlotGenerator(config)
        
        # Create DataFrame with multiple methods
        data_dict = {'AWS': sample_aws_data}
        data_dict.update(multiple_modis_methods)
        data = pd.DataFrame(data_dict)
        
        fig = generator.create_correlation_matrix(
            data,
            title="Correlation Test"
        )
        
        # Check that a figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Clean up
        plt.close(fig)
    
    @patch('matplotlib.pyplot.savefig')
    def test_create_summary_figure(self, mock_savefig, config, sample_aws_data, multiple_modis_methods):
        """Test comprehensive summary figure creation."""
        generator = PlotGenerator(config)
        
        fig = generator.create_summary_figure(
            sample_aws_data, multiple_modis_methods,
            glacier_name="Test Glacier"
        )
        
        # Check that a figure was created
        assert fig is not None
        assert isinstance(fig, plt.Figure)
        
        # Should have multiple subplots
        axes = fig.get_axes()
        assert len(axes) >= 4  # Expect multiple subplots in summary
        
        # Clean up
        plt.close(fig)
    
    def test_plot_with_empty_data(self, config):
        """Test plot generation with empty data."""
        generator = PlotGenerator(config)
        
        empty_series = pd.Series([], dtype=float)
        valid_series = pd.Series([0.5, 0.6, 0.7])
        
        # Should handle empty data gracefully
        fig = generator.create_scatterplot(empty_series, valid_series)
        
        # Should return None or handle gracefully
        if fig is not None:
            plt.close(fig)
    
    def test_plot_with_nan_data(self, config):
        """Test plot generation with NaN data."""
        generator = PlotGenerator(config)
        
        nan_series = pd.Series([0.5, np.nan, 0.7, np.nan])
        valid_series = pd.Series([0.4, 0.6, 0.8, 0.5])
        
        # Should handle NaN data by removing it
        fig = generator.create_scatterplot(nan_series, valid_series)
        
        if fig is not None:
            # Should still create a plot with valid data points
            axes = fig.get_axes()
            assert len(axes) >= 1
            plt.close(fig)
    
    def test_output_path_functionality(self, config, sample_aws_data, sample_modis_data):
        """Test saving plots to specified paths."""
        generator = PlotGenerator(config)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = os.path.join(temp_dir, 'test_plot.png')
            
            fig = generator.create_scatterplot(
                sample_aws_data, sample_modis_data,
                output_path=output_path
            )
            
            # Check that file was created
            assert os.path.exists(output_path)
            
            # Clean up
            if fig is not None:
                plt.close(fig)
    
    def test_color_configuration(self, config, sample_aws_data, multiple_modis_methods):
        """Test that plot colors match configuration."""
        generator = PlotGenerator(config)
        
        # Test that colors from config are used
        assert generator.colors['MOD09GA'] == '#1f77b4'
        assert generator.colors['AWS'] == '#d62728'
        
        # This is more of a visual check - in practice, you'd need to inspect
        # the actual plot objects to verify colors are applied correctly
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Set up and tear down for each test."""
        # Setup: Use non-interactive backend for testing
        plt.switch_backend('Agg')
        
        yield
        
        # Teardown: Close all figures
        plt.close('all')


if __name__ == '__main__':
    # Run tests if script is executed directly
    pytest.main([__file__])