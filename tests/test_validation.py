#!/usr/bin/env python3
"""
Test suite for validation utilities.

This test file validates the new validation functions without modifying
existing framework functionality. All tests are designed to be non-breaking.
"""

import sys
import os
import numpy as np
import pandas as pd
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.validation import (
    validate_file_exists,
    validate_dataframe_structure,
    validate_albedo_values,
    validate_correlation_data,
    validate_glacier_config,
    validate_analysis_results
)


class TestValidationUtilities(unittest.TestCase):
    """Test cases for validation utility functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data_dir = Path(__file__).parent / "test_data"
        self.test_data_dir.mkdir(exist_ok=True)
        
        # Create a test file
        self.test_file = self.test_data_dir / "test.csv"
        pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}).to_csv(self.test_file, index=False)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.test_file.exists():
            self.test_file.unlink()
        if self.test_data_dir.exists():
            self.test_data_dir.rmdir()
    
    def test_validate_file_exists(self):
        """Test file existence validation."""
        # Test existing file
        self.assertTrue(validate_file_exists(self.test_file))
        
        # Test non-existing file
        self.assertFalse(validate_file_exists("non_existent_file.csv"))
        
        # Test with description
        self.assertTrue(validate_file_exists(self.test_file, "Test CSV file"))
    
    def test_validate_dataframe_structure(self):
        """Test DataFrame structure validation."""
        # Create test DataFrame
        df = pd.DataFrame({'date': [1, 2, 3], 'albedo': [0.1, 0.2, 0.3], 'method': ['A', 'B', 'C']})
        
        # Test valid structure
        is_valid, missing = validate_dataframe_structure(df, ['date', 'albedo'])
        self.assertTrue(is_valid)
        self.assertEqual(missing, [])
        
        # Test missing columns
        is_valid, missing = validate_dataframe_structure(df, ['date', 'albedo', 'missing_col'])
        self.assertFalse(is_valid)
        self.assertEqual(missing, ['missing_col'])
        
        # Test empty DataFrame
        empty_df = pd.DataFrame()
        is_valid, missing = validate_dataframe_structure(empty_df, ['date'])
        self.assertFalse(is_valid)
    
    def test_validate_albedo_values(self):
        """Test albedo value validation."""
        # Test valid albedo values
        valid_values = np.array([0.0, 0.5, 1.0, 0.2, 0.8])
        result = validate_albedo_values(valid_values)
        self.assertTrue(result['valid'])
        self.assertEqual(result['n_invalid'], 0)
        
        # Test invalid albedo values
        invalid_values = np.array([0.5, 1.5, -0.1, 0.8])
        result = validate_albedo_values(invalid_values)
        self.assertFalse(result['valid'])
        self.assertEqual(result['n_invalid'], 2)
        
        # Test with NaN values
        nan_values = np.array([0.5, np.nan, 0.8, np.nan])
        result = validate_albedo_values(nan_values)
        self.assertTrue(result['valid'])  # NaN values are excluded from validation
        self.assertEqual(result['n_nan'], 2)
    
    def test_validate_correlation_data(self):
        """Test correlation data validation."""
        # Test valid correlation data
        aws_values = np.array([0.8, 0.7, 0.6, 0.9, 0.5, 0.4, 0.3, 0.2, 0.1, 0.8, 0.7])
        modis_values = np.array([0.75, 0.72, 0.58, 0.85, 0.48, 0.42, 0.35, 0.25, 0.15, 0.82, 0.68])
        
        result = validate_correlation_data(aws_values, modis_values, min_samples=10)
        self.assertTrue(result['valid'])
        self.assertEqual(result['n_valid_pairs'], 11)
        
        # Test insufficient data
        short_aws = np.array([0.8, 0.7])
        short_modis = np.array([0.75, 0.72])
        result = validate_correlation_data(short_aws, short_modis, min_samples=10)
        self.assertFalse(result['valid'])
        self.assertFalse(result['sufficient_data'])
        
        # Test length mismatch
        result = validate_correlation_data(aws_values, short_modis)
        self.assertFalse(result['valid'])
        self.assertIn('error', result)
    
    def test_validate_glacier_config(self):
        """Test glacier configuration validation."""
        # Test valid config
        valid_config = {
            'name': 'Test Glacier',
            'coordinates': {'lat': 50.0, 'lon': -115.0},
            'elevation': 2000
        }
        is_valid, errors = validate_glacier_config(valid_config)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])
        
        # Test missing required field
        invalid_config = {'coordinates': {'lat': 50.0, 'lon': -115.0}}
        is_valid, errors = validate_glacier_config(invalid_config)
        self.assertFalse(is_valid)
        self.assertIn('Missing required field: name', errors)
        
        # Test invalid coordinates
        invalid_coords_config = {
            'name': 'Test Glacier',
            'coordinates': {'lat': 100.0, 'lon': -200.0}  # Invalid lat/lon
        }
        is_valid, errors = validate_glacier_config(invalid_coords_config)
        self.assertFalse(is_valid)
        self.assertTrue(any('latitude' in error for error in errors))
        self.assertTrue(any('longitude' in error for error in errors))
    
    def test_validate_analysis_results(self):
        """Test analysis results validation."""
        # Test valid results
        valid_results = pd.DataFrame({
            'r': [0.8, 0.9, 0.7],
            'rmse': [0.1, 0.2, 0.15],
            'n_samples': [100, 150, 200]
        })
        result = validate_analysis_results(valid_results)
        self.assertTrue(result['valid'])
        
        # Test invalid correlation values
        invalid_results = pd.DataFrame({
            'r': [1.5, 0.9, -1.2],  # Invalid correlation values
            'rmse': [0.1, 0.2, 0.15],
            'n_samples': [100, 150, 200]
        })
        result = validate_analysis_results(invalid_results)
        self.assertFalse(result['valid'])
        self.assertFalse(result['checks']['correlation_range']['valid'])
        
        # Test negative RMSE values
        negative_rmse_results = pd.DataFrame({
            'r': [0.8, 0.9, 0.7],
            'rmse': [-0.1, 0.2, 0.15],  # Negative RMSE
            'n_samples': [100, 150, 200]
        })
        result = validate_analysis_results(negative_rmse_results)
        self.assertFalse(result['valid'])
        self.assertFalse(result['checks']['rmse_positive']['valid'])


class TestFrameworkIntegration(unittest.TestCase):
    """Test integration with existing framework components."""
    
    def test_import_existing_modules(self):
        """Test that existing modules can still be imported."""
        try:
            # Test importing existing analysis modules
            from src.analysis.comparative_analysis import MultiGlacierComparativeAnalysis
            from src.analysis.comparative_interface import ComparativeAnalysisInterface
            
            # Test that classes can be instantiated (without running analysis)
            analyzer = MultiGlacierComparativeAnalysis()
            interface = ComparativeAnalysisInterface()
            
            # Verify that our improvements don't break initialization
            self.assertIsNotNone(analyzer.outputs_dir)
            self.assertIsNotNone(interface.outputs_dir)
            
        except ImportError as e:
            self.fail(f"Failed to import existing modules: {e}")
    
    def test_enhanced_documentation(self):
        """Test that enhanced documentation is accessible."""
        from src.analysis.comparative_analysis import MultiGlacierComparativeAnalysis
        
        # Test that class has enhanced docstring
        analyzer = MultiGlacierComparativeAnalysis()
        self.assertIsNotNone(analyzer.__class__.__doc__)
        self.assertIn("Athabasca Glacier", analyzer.__class__.__doc__)
        self.assertIn("pixel selection", analyzer.__class__.__doc__)
        
        # Test that methods have enhanced documentation
        self.assertIsNotNone(analyzer.aggregate_glacier_data.__doc__)
        self.assertIn("outlier-filtered", analyzer.aggregate_glacier_data.__doc__)


if __name__ == '__main__':
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise during testing
    
    # Run tests
    unittest.main(verbosity=2)