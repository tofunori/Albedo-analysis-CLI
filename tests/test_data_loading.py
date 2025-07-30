import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

# Add src to path for testing
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.modis_loader import MOD09GALoader, MOD10A1Loader, MCD43A3Loader, create_modis_loader
from src.data.aws_loader import AWSAlbedoLoader
from src.data.data_processor import DataValidator, DataProcessor


class TestMODISLoaders:
    """Test MODIS data loaders."""
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing."""
        return {
            'analysis': {
                'quality_filters': {
                    'cloud_threshold': 0.2,
                    'snow_threshold': 0.1
                }
            }
        }
    
    @pytest.fixture
    def sample_mod09ga_data(self):
        """Sample MOD09GA data for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'lat': np.random.uniform(64, 67, 10),
            'lon': np.random.uniform(-20, -17, 10),
            'red_reflectance': np.random.uniform(0.1, 0.6, 10),
            'nir_reflectance': np.random.uniform(0.2, 0.8, 10),
            'quality_flag': np.random.uniform(0, 0.3, 10)
        })
    
    @pytest.fixture
    def sample_mod10a1_data(self):
        """Sample MOD10A1 data for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'lat': np.random.uniform(64, 67, 10),
            'lon': np.random.uniform(-20, -17, 10),
            'snow_albedo': np.random.uniform(30, 90, 10),  # Percentage
            'snow_cover': np.random.uniform(50, 100, 10),
            'quality_flag': np.random.uniform(0, 0.2, 10)
        })
    
    def test_mod09ga_loader_initialization(self, config):
        """Test MOD09GA loader initialization."""
        loader = MOD09GALoader(config)
        assert loader.product_name == "MOD09GA"
        assert loader.config == config
    
    def test_mod09ga_required_columns(self, config):
        """Test MOD09GA required columns."""
        loader = MOD09GALoader(config)
        required = loader.get_required_columns()
        expected = ['date', 'lat', 'lon', 'red_reflectance', 'nir_reflectance', 'quality_flag']
        assert required == expected
    
    def test_mod09ga_albedo_calculation(self, config, sample_mod09ga_data):
        """Test MOD09GA albedo calculation."""
        loader = MOD09GALoader(config)
        albedo = loader.calculate_albedo_mod09ga(sample_mod09ga_data)
        
        # Check that albedo is in valid range
        assert (albedo >= 0).all()
        assert (albedo <= 1).all()
        assert len(albedo) == len(sample_mod09ga_data)
    
    def test_mod09ga_validation(self, config, sample_mod09ga_data):
        """Test MOD09GA data validation."""
        loader = MOD09GALoader(config)
        
        # Valid data should pass
        assert loader.validate_data(sample_mod09ga_data) == True
        
        # Invalid coordinates should fail
        invalid_data = sample_mod09ga_data.copy()
        invalid_data.loc[0, 'lat'] = 95  # Invalid latitude
        assert loader.validate_data(invalid_data) == False
        
        # Invalid reflectance should fail
        invalid_data = sample_mod09ga_data.copy()
        invalid_data.loc[0, 'red_reflectance'] = 1.5  # Invalid reflectance
        assert loader.validate_data(invalid_data) == False
    
    def test_mod10a1_loader(self, config, sample_mod10a1_data):
        """Test MOD10A1 loader."""
        loader = MOD10A1Loader(config)
        
        # Test required columns
        required = loader.get_required_columns()
        expected = ['date', 'lat', 'lon', 'snow_albedo', 'snow_cover', 'quality_flag']
        assert required == expected
        
        # Test validation
        assert loader.validate_data(sample_mod10a1_data) == True
    
    def test_create_modis_loader(self, config):
        """Test MODIS loader factory function."""
        # Test valid products
        loader1 = create_modis_loader('MOD09GA', config)
        assert isinstance(loader1, MOD09GALoader)
        
        loader2 = create_modis_loader('MOD10A1', config)
        assert isinstance(loader2, MOD10A1Loader)
        
        loader3 = create_modis_loader('MCD43A3', config)
        assert isinstance(loader3, MCD43A3Loader)
        
        # Test invalid product
        with pytest.raises(ValueError):
            create_modis_loader('INVALID', config)


class TestAWSLoader:
    """Test AWS data loader."""
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing."""
        return {
            'analysis': {
                'quality_filters': {
                    'cloud_threshold': 0.2
                }
            }
        }
    
    @pytest.fixture
    def sample_aws_data(self):
        """Sample AWS data for testing."""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50, freq='H'),
            'albedo': np.random.uniform(0.2, 0.8, 50),
            'station_id': ['AWS_01'] * 50,
            'temperature': np.random.uniform(-10, 5, 50),
            'wind_speed': np.random.uniform(0, 15, 50)
        })
    
    @pytest.fixture
    def glacier_config(self):
        """Sample glacier configuration."""
        return {
            'aws_stations': {
                'AWS_01': {
                    'lat': 65.0,
                    'lon': -18.0,
                    'elevation': 1200
                }
            }
        }
    
    def test_aws_loader_initialization(self, config):
        """Test AWS loader initialization."""
        loader = AWSAlbedoLoader(config)
        assert loader.config == config
    
    def test_aws_required_columns(self, config):
        """Test AWS required columns."""
        loader = AWSAlbedoLoader(config)
        required = loader.get_required_columns()
        expected = ['date', 'albedo', 'station_id']
        assert required == expected
    
    def test_aws_validation(self, config, sample_aws_data):
        """Test AWS data validation."""
        loader = AWSAlbedoLoader(config)
        
        # Valid data should pass
        assert loader.validate_data(sample_aws_data) == True
        
        # Invalid albedo should fail
        invalid_data = sample_aws_data.copy()
        invalid_data.loc[0, 'albedo'] = 1.5  # Invalid albedo
        assert loader.validate_data(invalid_data) == False
    
    def test_aws_quality_filtering(self, config, sample_aws_data):
        """Test AWS quality filtering.""" 
        loader = AWSAlbedoLoader(config)
        
        # Add extreme values that should be filtered
        sample_aws_data.loc[0, 'albedo'] = 0.01  # Very low
        sample_aws_data.loc[1, 'albedo'] = 0.99  # Very high
        sample_aws_data.loc[2, 'temperature'] = -50  # Very cold
        sample_aws_data.loc[3, 'wind_speed'] = 35  # Very windy
        
        filtered = loader.quality_filter(sample_aws_data)
        
        # Should have fewer records after filtering
        assert len(filtered) < len(sample_aws_data)
    
    def test_aws_coordinate_addition(self, config, sample_aws_data, glacier_config):
        """Test adding station coordinates to AWS data."""
        loader = AWSAlbedoLoader(config)
        
        data_with_coords = loader.add_station_coordinates(sample_aws_data, glacier_config)
        
        # Check that coordinates were added
        assert 'lat' in data_with_coords.columns
        assert 'lon' in data_with_coords.columns
        assert 'elevation' in data_with_coords.columns
        
        # Check values
        aws_01_data = data_with_coords[data_with_coords['station_id'] == 'AWS_01']
        assert (aws_01_data['lat'] == 65.0).all()
        assert (aws_01_data['lon'] == -18.0).all()
        assert (aws_01_data['elevation'] == 1200).all()
    
    def test_aws_daily_resampling(self, config, sample_aws_data):
        """Test AWS daily resampling."""
        loader = AWSAlbedoLoader(config)
        
        daily_data = loader.resample_to_daily(sample_aws_data)
        
        # Should have fewer records (daily instead of hourly)
        assert len(daily_data) <= len(sample_aws_data)
        
        # Should have daily frequency
        if not daily_data.empty:
            date_diffs = daily_data['date'].diff().dropna()
            # Most differences should be 1 day (allowing for missing days)
            assert date_diffs.mode()[0].days == 1


class TestDataValidator:
    """Test data validation utilities."""
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing."""
        return {}
    
    @pytest.fixture
    def sample_temporal_data(self):
        """Sample data with temporal structure."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        # Add some gaps
        dates = dates.delete([10, 11, 12, 20])  # Create gaps
        
        return pd.DataFrame({
            'date': dates,
            'albedo': np.random.uniform(0.2, 0.8, len(dates)),
            'lat': np.full(len(dates), 65.0),
            'lon': np.full(len(dates), -18.0)
        })
    
    def test_validator_initialization(self, config):
        """Test validator initialization."""
        validator = DataValidator(config)
        assert validator.config == config
    
    def test_temporal_consistency_validation(self, config, sample_temporal_data):
        """Test temporal consistency validation."""
        validator = DataValidator(config)
        
        results = validator.validate_temporal_consistency(sample_temporal_data)
        
        assert 'valid' in results
        assert 'errors' in results
        assert 'warnings' in results
        assert 'stats' in results
        
        # Should detect gaps
        assert len(results['warnings']) > 0
    
    def test_spatial_consistency_validation(self, config, sample_temporal_data):
        """Test spatial consistency validation."""
        validator = DataValidator(config)
        
        results = validator.validate_spatial_consistency(sample_temporal_data)
        
        assert 'valid' in results
        assert results['valid'] == True  # Valid coordinates
        assert 'stats' in results
        
        # Test invalid coordinates
        invalid_data = sample_temporal_data.copy()
        invalid_data.loc[0, 'lat'] = 95  # Invalid latitude
        
        results_invalid = validator.validate_spatial_consistency(invalid_data)
        assert results_invalid['valid'] == False
    
    def test_outlier_detection(self, config, sample_temporal_data):
        """Test outlier detection."""
        validator = DataValidator(config)
        
        # Add outliers
        sample_temporal_data.loc[0, 'albedo'] = 2.0  # Extreme outlier
        sample_temporal_data.loc[1, 'albedo'] = -0.5  # Another outlier
        
        results = validator.detect_outliers(sample_temporal_data, 'albedo', method='iqr')
        
        assert 'outliers' in results
        assert 'count' in results
        assert results['count'] >= 2  # Should detect the outliers we added
    
    def test_albedo_validation(self, config, sample_temporal_data):
        """Test albedo-specific validation."""
        validator = DataValidator(config)
        
        results = validator.validate_albedo_values(sample_temporal_data)
        
        assert 'valid' in results
        assert 'stats' in results
        
        # Test invalid albedo
        invalid_data = sample_temporal_data.copy()
        invalid_data.loc[0, 'albedo'] = 1.5  # Invalid albedo
        
        results_invalid = validator.validate_albedo_values(invalid_data)
        assert results_invalid['valid'] == False


class TestDataProcessor:
    """Test data processing utilities."""
    
    @pytest.fixture
    def config(self):
        """Basic configuration for testing."""
        return {}
    
    @pytest.fixture
    def sample_modis_data(self):
        """Sample MODIS data."""
        return pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=10, freq='D'),
            'lat': np.full(10, 65.0),
            'lon': np.full(10, -18.0),
            'albedo': np.random.uniform(0.3, 0.7, 10)
        })
    
    @pytest.fixture
    def sample_aws_data(self):
        """Sample AWS data."""
        # Overlap with MODIS dates
        return pd.DataFrame({
            'date': pd.date_range('2023-01-03', periods=8, freq='D'),
            'albedo': np.random.uniform(0.4, 0.8, 8),
            'station_id': ['AWS_01'] * 8
        })
    
    def test_processor_initialization(self, config):
        """Test processor initialization."""
        processor = DataProcessor(config)
        assert processor.config == config
        assert isinstance(processor.validator, DataValidator)
    
    def test_temporal_alignment(self, config, sample_modis_data, sample_aws_data):
        """Test temporal data alignment."""
        processor = DataProcessor(config)
        
        aligned = processor.align_temporal_data(sample_modis_data, sample_aws_data)
        
        # Should have some aligned records
        assert not aligned.empty
        assert 'modis_albedo' in aligned.columns
        assert 'aws_albedo' in aligned.columns
        assert 'time_diff_hours' in aligned.columns
        
        # Time differences should be reasonable (within tolerance)
        assert (aligned['time_diff_hours'].abs() <= 12).all()
    
    def test_data_report_generation(self, config, sample_modis_data):
        """Test data quality report generation."""
        processor = DataProcessor(config)
        
        report = processor.generate_data_report(sample_modis_data, 'MODIS')
        
        assert 'data_type' in report
        assert report['data_type'] == 'MODIS'
        assert 'basic_stats' in report
        assert 'validation_results' in report
        assert 'recommendations' in report
        
        # Check basic stats
        assert report['basic_stats']['record_count'] == len(sample_modis_data)


if __name__ == '__main__':
    # Run tests if script is executed directly
    pytest.main([__file__])