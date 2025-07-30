import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.analysis.albedo_calculator import AlbedoCalculator
from src.analysis.statistical_analysis import StatisticalAnalyzer


class TestAlbedoCalculator:
    """Test albedo calculation methods."""
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing."""
        return {
            'analysis': {
                'quality_filters': {
                    'snow_threshold': 10
                }
            }
        }
    
    @pytest.fixture
    def sample_mod09ga_data(self):
        """Sample MOD09GA reflectance data."""
        return pd.DataFrame({
            'red_reflectance': np.random.uniform(0.1, 0.5, 20),
            'nir_reflectance': np.random.uniform(0.2, 0.7, 20),
            'blue_reflectance': np.random.uniform(0.05, 0.3, 20),
            'green_reflectance': np.random.uniform(0.08, 0.4, 20)
        })
    
    @pytest.fixture
    def sample_mod10a1_data(self):
        """Sample MOD10A1 snow data."""
        return pd.DataFrame({
            'snow_albedo': np.random.uniform(30, 90, 20),  # Percentage
            'snow_cover': np.random.uniform(20, 100, 20)
        })
    
    @pytest.fixture
    def sample_mcd43a3_data(self):
        """Sample MCD43A3 BRDF data."""
        return pd.DataFrame({
            'white_sky_albedo': np.random.uniform(0.4, 0.8, 20),
            'black_sky_albedo': np.random.uniform(0.3, 0.7, 20)
        })
    
    def test_calculator_initialization(self, config):
        """Test calculator initialization."""
        calculator = AlbedoCalculator(config)
        assert calculator.config == config
    
    def test_mod09ga_broadband_albedo(self, config, sample_mod09ga_data):
        """Test MOD09GA broadband albedo calculation."""
        calculator = AlbedoCalculator(config)
        
        albedo = calculator._broadband_albedo_mod09ga(sample_mod09ga_data)
        
        # Check output properties
        assert len(albedo) == len(sample_mod09ga_data)
        assert (albedo >= 0).all()
        assert (albedo <= 1).all()
        assert not albedo.isna().any()
    
    def test_mod09ga_narrowband_albedo(self, config, sample_mod09ga_data):
        """Test MOD09GA narrowband albedo calculation."""
        calculator = AlbedoCalculator(config)
        
        albedo = calculator._narrowband_albedo_mod09ga(sample_mod09ga_data)
        
        # Check output properties
        assert len(albedo) == len(sample_mod09ga_data)
        assert (albedo >= 0).all()
        assert (albedo <= 1).all()
    
    def test_mod09ga_liang_albedo(self, config, sample_mod09ga_data):
        """Test MOD09GA Liang algorithm albedo calculation."""
        calculator = AlbedoCalculator(config)
        
        albedo = calculator._liang_albedo_mod09ga(sample_mod09ga_data)
        
        # Check output properties
        assert len(albedo) == len(sample_mod09ga_data)
        assert (albedo >= 0).all()
        assert (albedo <= 1).all()
    
    def test_mod10a1_albedo_calculation(self, config, sample_mod10a1_data):
        """Test MOD10A1 albedo calculation."""
        calculator = AlbedoCalculator(config)
        
        albedo = calculator.calculate_mod10a1_albedo(sample_mod10a1_data)
        
        # Check output properties
        assert len(albedo) == len(sample_mod10a1_data)
        assert (albedo >= 0).all()
        assert (albedo <= 1).all()
    
    def test_mcd43a3_albedo_calculation(self, config, sample_mcd43a3_data):
        """Test MCD43A3 blue-sky albedo calculation."""
        calculator = AlbedoCalculator(config)
        
        albedo = calculator.calculate_mcd43a3_albedo(sample_mcd43a3_data)
        
        # Check output properties
        assert len(albedo) == len(sample_mcd43a3_data)
        assert (albedo >= 0).all()
        assert (albedo <= 1).all()
    
    def test_mcd43a3_with_solar_zenith(self, config, sample_mcd43a3_data):
        """Test MCD43A3 albedo calculation with solar zenith angle."""
        calculator = AlbedoCalculator(config)
        
        # Solar zenith angles in degrees
        solar_zenith = pd.Series(np.random.uniform(20, 70, len(sample_mcd43a3_data)))
        
        albedo = calculator.calculate_mcd43a3_albedo(sample_mcd43a3_data, solar_zenith)
        
        # Check output properties
        assert len(albedo) == len(sample_mcd43a3_data)
        assert (albedo >= 0).all()
        assert (albedo <= 1).all()
    
    def test_temporal_smoothing(self, config):
        """Test temporal smoothing of albedo time series."""
        calculator = AlbedoCalculator(config)
        
        # Create time series with noise
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        albedo_series = pd.Series(
            0.5 + 0.1 * np.sin(np.arange(30) * 2 * np.pi / 15) + np.random.normal(0, 0.05, 30),
            index=dates
        )
        
        # Test rolling mean
        smoothed_mean = calculator.temporal_smoothing(albedo_series, window_days=7, method='rolling_mean')
        assert len(smoothed_mean) == len(albedo_series)
        
        # Test rolling median
        smoothed_median = calculator.temporal_smoothing(albedo_series, window_days=7, method='rolling_median')
        assert len(smoothed_median) == len(albedo_series)
    
    def test_gap_filling(self, config):
        """Test gap filling in albedo time series."""
        calculator = AlbedoCalculator(config)
        
        # Create time series with gaps
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        values = np.random.uniform(0.3, 0.7, 30)
        values[10:13] = np.nan  # Create gaps
        values[20:22] = np.nan
        
        albedo_series = pd.Series(values, index=dates)
        
        # Test interpolation gap filling
        filled = calculator.gap_filling(albedo_series, method='interpolation', max_gap_days=5)
        
        # Should have fewer NaN values
        assert filled.isna().sum() <= albedo_series.isna().sum()
    
    def test_quality_assessment(self, config):
        """Test quality assessment of albedo values."""
        calculator = AlbedoCalculator(config)
        
        # Create albedo series with various quality issues
        albedo_values = np.concatenate([
            np.random.uniform(0.3, 0.7, 80),  # Normal values
            [0.01, 0.02],  # Very low values
            [0.96, 0.98],  # Very high values
            [np.nan] * 10  # Missing values
        ])
        
        albedo_series = pd.Series(albedo_values)
        
        assessment = calculator.quality_assessment(albedo_series)
        
        # Check assessment structure
        assert 'total_values' in assessment
        assert 'valid_values' in assessment
        assert 'data_completeness' in assessment
        assert 'quality_flags' in assessment
        
        # Check values
        assert assessment['total_values'] == len(albedo_series)
        assert assessment['valid_values'] == (~albedo_series.isna()).sum()
        assert assessment['quality_flags']['very_low_values'] >= 2
        assert assessment['quality_flags']['very_high_values'] >= 2


class TestStatisticalAnalyzer:
    """Test statistical analysis methods."""
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing."""
        return {
            'analysis': {
                'statistics': {
                    'confidence_level': 0.95
                }
            }
        }
    
    @pytest.fixture
    def sample_observed_data(self):
        """Sample observed (reference) data."""
        np.random.seed(42)  # For reproducible tests
        return pd.Series(np.random.uniform(0.3, 0.8, 100))
    
    @pytest.fixture
    def sample_predicted_data(self, sample_observed_data):
        """Sample predicted data (correlated with observed)."""
        np.random.seed(43)
        # Add some bias and noise to observed data
        noise = np.random.normal(0, 0.05, len(sample_observed_data))
        bias = 0.02
        return sample_observed_data + bias + noise
    
    def test_analyzer_initialization(self, config):
        """Test analyzer initialization."""
        analyzer = StatisticalAnalyzer(config)
        assert analyzer.config == config
        assert analyzer.confidence_level == 0.95
    
    def test_basic_metrics_calculation(self, config, sample_observed_data, sample_predicted_data):
        """Test basic statistical metrics calculation."""
        analyzer = StatisticalAnalyzer(config)
        
        metrics = analyzer.calculate_basic_metrics(sample_observed_data, sample_predicted_data)
        
        # Check that all expected metrics are present
        expected_metrics = [
            'n_samples', 'rmse', 'mae', 'bias', 'relative_bias', 
            'r2', 'correlation', 'std_obs', 'std_pred', 'mean_obs', 'mean_pred',
            'nse', 'ioa', 'pbias', 'kge'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics
            assert not np.isnan(metrics[metric]) or metric in ['r2']  # R² can be NaN in edge cases
        
        # Check reasonable values
        assert metrics['n_samples'] == len(sample_observed_data)
        assert metrics['rmse'] >= 0
        assert metrics['mae'] >= 0
        assert -1 <= metrics['correlation'] <= 1
    
    def test_confidence_intervals(self, config, sample_observed_data, sample_predicted_data):
        """Test confidence interval calculation."""
        analyzer = StatisticalAnalyzer(config)
        
        ci = analyzer.calculate_confidence_intervals(sample_observed_data, sample_predicted_data)
        
        # Check structure
        for metric in ['rmse', 'bias', 'correlation']:
            if metric in ci:
                assert 'lower' in ci[metric]
                assert 'upper' in ci[metric]
                assert 'width' in ci[metric]
                
                # Upper bound should be greater than lower bound
                assert ci[metric]['upper'] >= ci[metric]['lower']
    
    def test_significance_tests(self, config, sample_observed_data, sample_predicted_data):
        """Test statistical significance tests."""
        analyzer = StatisticalAnalyzer(config)
        
        # Create multiple datasets
        datasets = {
            'method1': sample_observed_data,
            'method2': sample_predicted_data,
            'method3': sample_observed_data + np.random.normal(0, 0.03, len(sample_observed_data))
        }
        
        results = analyzer.perform_significance_tests(datasets)
        
        # Check that pairwise comparisons were performed
        expected_pairs = ['method1_vs_method2', 'method1_vs_method3', 'method2_vs_method3']
        
        for pair in expected_pairs:
            if pair in results:
                assert 't_test' in results[pair]
                assert 'mann_whitney' in results[pair]
                assert 'ks_test' in results[pair]
                
                # Check test results structure
                for test in ['t_test', 'mann_whitney', 'ks_test']:
                    assert 'statistic' in results[pair][test]
                    assert 'p_value' in results[pair][test]
    
    def test_seasonal_statistics(self, config):
        """Test seasonal statistics calculation."""
        analyzer = StatisticalAnalyzer(config)
        
        # Create seasonal data
        dates = pd.date_range('2023-01-01', periods=365, freq='D')
        # Simulate seasonal pattern
        day_of_year = dates.dayofyear
        seasonal_pattern = 0.5 + 0.2 * np.sin(2 * np.pi * day_of_year / 365)
        noise = np.random.normal(0, 0.05, len(dates))
        albedo_values = seasonal_pattern + noise
        
        data = pd.DataFrame({
            'date': dates,
            'albedo': albedo_values
        })
        
        seasonal_stats = analyzer.calculate_seasonal_statistics(data)
        
        # Check structure
        assert 'seasonal' in seasonal_stats
        assert 'monthly' in seasonal_stats
        
        # Check that all seasons are present (if data exists)
        seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
        for season in seasons:
            if season in seasonal_stats['seasonal']:
                season_data = seasonal_stats['seasonal'][season]
                assert 'count' in season_data
                assert 'mean' in season_data
                assert 'std' in season_data
    
    def test_trend_analysis(self, config):
        """Test trend analysis."""
        analyzer = StatisticalAnalyzer(config)
        
        # Create data with trend
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        trend = 0.0001 * np.arange(1000)  # Small positive trend
        seasonal = 0.1 * np.sin(2 * np.pi * np.arange(1000) / 365)
        noise = np.random.normal(0, 0.02, 1000)
        albedo_values = 0.5 + trend + seasonal + noise
        
        data = pd.DataFrame({
            'date': dates,
            'albedo': albedo_values
        })
        
        trend_results = analyzer.trend_analysis(data)
        
        # Check structure
        assert 'linear_trend' in trend_results
        assert 'mann_kendall' in trend_results
        assert 'data_period' in trend_results
        
        # Check linear trend
        linear_trend = trend_results['linear_trend']
        assert 'slope' in linear_trend
        assert 'slope_per_year' in linear_trend
        assert 'p_value' in linear_trend
        assert 'r_squared' in linear_trend
        
        # Should detect positive trend (though may not be statistically significant due to noise)
        assert linear_trend['slope'] > 0
    
    def test_multi_method_comparison(self, config, sample_observed_data):
        """Test multiple method comparison."""
        analyzer = StatisticalAnalyzer(config)
        
        # Create multiple methods with different performance
        method_data = {
            'method1': sample_observed_data + np.random.normal(0.01, 0.02, len(sample_observed_data)),  # Small bias, low noise
            'method2': sample_observed_data + np.random.normal(0.05, 0.05, len(sample_observed_data)),  # Large bias, high noise
            'method3': sample_observed_data + np.random.normal(-0.02, 0.03, len(sample_observed_data))  # Negative bias, medium noise
        }
        
        comparison_df = analyzer.compare_multiple_methods(sample_observed_data, method_data)
        
        # Check structure
        assert 'method' in comparison_df.columns
        assert 'rmse' in comparison_df.columns
        assert 'bias' in comparison_df.columns
        assert 'overall_rank' in comparison_df.columns
        
        # Should be sorted by overall rank
        assert comparison_df['overall_rank'].is_monotonic_increasing
        
        # Best method should have lowest RMSE (generally)
        best_method = comparison_df.iloc[0]
        assert best_method['rmse'] <= comparison_df['rmse'].median()
    
    def test_empty_data_handling(self, config):
        """Test handling of empty data."""
        analyzer = StatisticalAnalyzer(config)
        
        empty_series = pd.Series([], dtype=float)
        valid_series = pd.Series([0.5, 0.6, 0.7])
        
        # Should return empty metrics
        metrics = analyzer.calculate_basic_metrics(empty_series, valid_series)
        assert metrics['n_samples'] == 0
        assert np.isnan(metrics['rmse'])
        
        # Should handle NaN values
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        metrics = analyzer.calculate_basic_metrics(nan_series, valid_series)
        assert metrics['n_samples'] == 0


if __name__ == '__main__':
    # Run tests if script is executed directly
    pytest.main([__file__])