# Code Structure Reference Guide

## Overview

This document provides comprehensive guidelines for developing new glacier albedo analysis scripts following the established patterns in this codebase. All analysis scripts follow a standardized modular architecture for consistency, maintainability, and professional quality.

## File Organization Standards

### Required Structure

Every analysis script must follow this exact structure:

```python
#!/usr/bin/env python3
"""
[Analysis Name] Generator

[Detailed description]
[Features list]

Author: [Name]
Date: [Date]
"""

# ============================================================================
# IMPORTS
# ============================================================================

# ============================================================================
# LOGGING SETUP
# ============================================================================

# ============================================================================
# CONFIGURATION
# ============================================================================

# ============================================================================
# [MODULE] MODULE
# ============================================================================

# ============================================================================
# SUMMARY AND DOCUMENTATION FUNCTIONS
# ============================================================================

# ============================================================================
# MAIN EXECUTION FUNCTIONS
# ============================================================================
```

### Section Headers

- Use exactly **76 equal signs** for section dividers
- Always include section comment headers for clear organization
- Group related functions/classes under appropriate sections

## Import Standards

### Required Import Categories

Organize imports in this exact order:

```python
# Standard library imports
import logging
from datetime import datetime
from pathlib import Path

# Scientific computing and data manipulation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

# Statistical analysis
from scipy import stats

# Type hints for better code documentation
from typing import Any, Dict, List, Optional, Tuple

# Local utilities
from output_manager import OutputManager
```

### Import Guidelines

- **Standard library first** - logging, datetime, pathlib, etc.
- **Scientific packages** - matplotlib, numpy, pandas
- **Specialized packages** - scipy, seaborn, sklearn, etc.
- **Type hints** - from typing import ...
- **Local imports last** - output_manager, custom modules

## Configuration Dictionary (CONFIG)

### Required CONFIG Structure

```python
CONFIG = {
    'data_paths': {
        'athabasca': {'modis': '...', 'aws': '...'},
        'haig': {'modis': '...', 'aws': '...'},
        'coropuna': {'modis': '...', 'aws': '...'}
    },
    'aws_stations': {
        'athabasca': {'lat': float, 'lon': float, 'name': str},
        'haig': {'lat': float, 'lon': float, 'name': str},
        'coropuna': {'lat': float, 'lon': float, 'name': str}
    },
    'colors': {
        # Glacier and method specific colors
    },
    'methods': ['MCD43A3', 'MOD09GA', 'MOD10A1'],
    'method_mapping': {
        # Case-insensitive standardization
    },
    'outlier_threshold': 2.5,
    'quality_filters': {
        'min_glacier_fraction': 0.1,
        'min_observations': 10
    },
    'visualization': {
        'figsize': (width, height),
        'dpi': 300,
        'style': 'seaborn-v0_8'
    },
    'output': {
        'analysis_name': 'lowercase_name',
        'base_dir': 'outputs',
        'plot_filename': 'descriptive_name.png',
        'summary_template': {
            'analysis_type': 'Descriptive Title',
            'description': 'Detailed description'
        }
    }
}
```

### CONFIG Guidelines

- **Always include all standard sections** - Even if not all are used
- **Use consistent coordinate formats** - lat/lon as floats
- **Standardize method names** - Use uppercase for MODIS products
- **Include comprehensive method mapping** - Handle case variations
- **Set reasonable defaults** - Outlier threshold 2.5σ, min observations 10
- **Use descriptive filenames** - Include analysis type in plot filename

## Class Architecture

### Standard Class Pattern

Every analysis follows this modular architecture:

1. **DataLoader** - Handles file loading and preprocessing
2. **PixelSelector** - Implements intelligent pixel selection
3. **DataProcessor** - Merges data and calculates statistics
4. **[AnalysisType]Visualizer** - Creates analysis-specific plots

### Class Implementation Standards

#### DataLoader Class

```python
class DataLoader:
    """Handles loading and preprocessing of MODIS and AWS data for all glaciers."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_glacier_data(self, glacier_id: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and preprocess MODIS and AWS data for a specific glacier."""
        # Standard validation
        if glacier_id not in self.config['data_paths']:
            raise ValueError(f"Unknown glacier ID: {glacier_id}")
        
        # Implementation...
        
    def _load_modis_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        """Load MODIS data with glacier-specific parsing."""
        # Handle file existence
        if not Path(file_path).exists():
            raise FileNotFoundError(f"MODIS data file not found: {file_path}")
        
        # Implementation...
        
    def _load_aws_data(self, file_path: str, glacier_id: str) -> pd.DataFrame:
        """Load AWS data with glacier-specific parsing."""
        # Implementation...
        
    def _convert_to_long_format(self, data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Convert wide format MODIS data to long format."""
        # Implementation...
```

#### PixelSelector Class

```python
class PixelSelector:
    """Implements intelligent pixel selection based on distance to AWS stations."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def select_best_pixels(self, modis_data: pd.DataFrame, glacier_id: str) -> pd.DataFrame:
        """Select best pixels for analysis based on AWS distance and glacier fraction."""
        # Implementation...
        
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance using Haversine formula."""
        R = 6371  # Earth's radius in km
        # Standard Haversine implementation...
```

#### DataProcessor Class

```python
class DataProcessor:
    """Handles AWS-MODIS data merging and statistical processing."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def merge_and_process(self, modis_data: pd.DataFrame, aws_data: pd.DataFrame, 
                         glacier_id: str) -> pd.DataFrame:
        """Merge AWS and MODIS data and calculate statistics for each method."""
        # Implementation...
        
    def _apply_outlier_filtering(self, aws_vals: np.ndarray, modis_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply 2.5σ outlier filtering to AWS-MODIS pairs."""
        # Standard outlier filtering implementation...
        
    def _calculate_statistics(self, aws_vals: np.ndarray, modis_vals: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive statistics between AWS and MODIS values."""
        # Standard statistical calculations...
```

### Class Guidelines

- **Always include type hints** - For all parameters and return values
- **Use descriptive docstrings** - Explain purpose, parameters, and returns
- **Implement error handling** - Check file existence, validate inputs
- **Follow naming conventions** - Private methods start with underscore
- **Keep methods focused** - Single responsibility principle

## Naming Conventions

### Variable Names

- **snake_case** for all variables and function names
- **Descriptive names** - `modis_data` not `data1`
- **Consistent naming** - Always use `glacier_id`, `aws_data`, `modis_data`
- **Method consistency** - Always use `method` column for MODIS products

### Function Names

- **Verbs for actions** - `load_glacier_data`, `calculate_statistics`
- **Private methods** - Start with underscore: `_apply_outlier_filtering`
- **Descriptive names** - Explain what the function does

### Class Names

- **PascalCase** for class names
- **Descriptive suffixes** - `DataLoader`, `PixelSelector`, `Visualizer`
- **Analysis-specific names** - `ScatterplotVisualizer`, `BarChartVisualizer`

## Error Handling Standards

### Required Error Checks

```python
# File existence
if not Path(file_path).exists():
    raise FileNotFoundError(f"Data file not found: {file_path}")

# Glacier ID validation
if glacier_id not in self.config['data_paths']:
    raise ValueError(f"Unknown glacier ID: {glacier_id}")

# Minimum data requirements
if len(merged) < 3:
    logger.warning(f"Insufficient data for {method}: {len(merged)} points")
    continue

# Empty DataFrame checks
if data.empty:
    logger.warning(f"No data available for {glacier_id}")
    return pd.DataFrame()
```

### Error Handling Guidelines

- **Use appropriate exception types** - FileNotFoundError, ValueError, etc.
- **Include descriptive messages** - Help debugging with context
- **Log warnings for data issues** - Use logger.warning() for non-fatal issues
- **Continue processing when possible** - Don't stop entire analysis for one glacier
- **Return empty DataFrames** - When no data available, return empty not None

## Logging Standards

### Required Logging Setup

```python
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
```

### Logging Guidelines

- **Use INFO for progress** - Major steps and successful operations
- **Use WARNING for issues** - Data problems that don't stop processing
- **Use ERROR for failures** - Critical problems that affect results
- **Include context** - Glacier names, method names, data counts
- **Log file operations** - When files are loaded or saved

### Logging Examples

```python
# Progress logging
logger.info(f"Loading data for {glacier_id} glacier...")
logger.info(f"Loaded {len(modis_data):,} MODIS and {len(aws_data):,} AWS records")

# Warning logging
logger.warning(f"No {method} data found for {glacier_id}")
logger.warning(f"Insufficient data after outlier filtering: {len(clean_data)} points")

# Error logging
logger.error(f"Error processing {glacier_id}: {e}")
```

## Visualization Standards

### Standard Matplotlib Setup

```python
# Set style
try:
    plt.style.use(self.config['visualization']['style'])
except:
    logger.warning("Could not set plotting style, using default")

# Create figure
fig, axes = plt.subplots(rows, cols, figsize=self.config['visualization']['figsize'])

# Add main title
fig.suptitle('Analysis Title', fontsize=16, fontweight='bold')
```

### Color Standards

Use the standardized color scheme from CONFIG:

```python
'colors': {
    # Glacier-specific colors
    'athabasca': '#1f77b4',    # Blue
    'haig': '#ff7f0e',         # Orange  
    'coropuna': '#2ca02c',     # Green
    
    # Method-specific colors
    'MOD09GA': '#9467bd',      # Purple
    'MCD43A3': '#d62728',      # Red
    'MOD10A1': '#8c564b',      # Brown
    'AWS': '#000000'           # Black
}
```

### Plot Guidelines

- **Use consistent colors** - Follow the established color scheme
- **Include proper legends** - Label all plot elements
- **Add axis labels** - Clear, descriptive axis labels
- **Set appropriate limits** - Use consistent scales across subplots
- **Include sample sizes** - Show n= for each data group
- **Save high quality** - Use dpi=300 for publication quality

## Output Management

### OutputManager Integration

Always use OutputManager for consistent file organization:

```python
# Initialize OutputManager
output_manager = OutputManager(
    CONFIG['output']['analysis_name'],
    CONFIG['output']['base_dir']
)

# Use for plot paths
plot_path = output_manager.get_plot_path(CONFIG['output']['plot_filename'])

# Log file operations
output_manager.log_file_saved(plot_path, "plot")
```

### Summary Generation

Required summary function:

```python
def generate_summary_and_readme(output_manager: OutputManager, processed_data: Any):
    """Generate summary file and README with analysis results."""
    try:
        # Collect statistics
        # Prepare summary data
        # Save summary and README
        output_manager.save_summary(summary_data)
        output_manager.save_readme(...)
        
    except Exception as e:
        logger.error(f"Error generating summary and README: {e}")
```

## Main Execution Pattern

### Standard main() Function

```python
def main():
    """Main execution function."""
    logger.info("Starting [Analysis Name] Generation")
    
    # Initialize OutputManager
    output_manager = OutputManager(...)
    
    # Initialize components
    data_loader = DataLoader(CONFIG)
    pixel_selector = PixelSelector(CONFIG)
    data_processor = DataProcessor(CONFIG)
    visualizer = AnalysisVisualizer(CONFIG)
    
    # Process each glacier
    all_processed_data = []
    
    for glacier_id in ['athabasca', 'haig', 'coropuna']:
        try:
            # Processing pipeline
            modis_data, aws_data = data_loader.load_glacier_data(glacier_id)
            selected_modis = pixel_selector.select_best_pixels(modis_data, glacier_id)
            processed = data_processor.merge_and_process(selected_modis, aws_data, glacier_id)
            
            if not processed.empty:
                all_processed_data.append(processed)
                
        except Exception as e:
            logger.error(f"Error processing {glacier_id}: {e}")
            continue
    
    # Create visualization and documentation
    if all_processed_data:
        # Create plots
        # Generate summary
        # Success logging
    else:
        logger.error("No data could be processed for any glacier")

if __name__ == "__main__":
    main()
```

## Data Standards

### DataFrame Column Standards

- **date** - pandas datetime, consistent across all data
- **albedo** - MODIS albedo values (0-1 range)
- **Albedo** - AWS albedo values (0-1 range, note capitalization)
- **method** - Standardized MODIS product names (uppercase)
- **glacier_id** - lowercase glacier identifier
- **pixel_id** - unique pixel identifier
- **latitude**, **longitude** - spatial coordinates
- **glacier_fraction** - fraction of pixel covered by glacier (0-1)

### Data Processing Standards

- **Outlier filtering** - Always apply 2.5σ threshold
- **Date merging** - Use inner joins on date column
- **Missing data** - Remove NaN values before analysis
- **Quality filters** - Apply minimum glacier fraction and observation counts

## Best Practices

### Development Workflow

1. **Copy template** - Start with `analysis_template.py`
2. **Customize CONFIG** - Update analysis name, plot filename, description
3. **Implement visualizer** - Focus on the `create_visualization` method
4. **Customize data processor** - Modify `merge_and_process` for analysis needs
5. **Update summary generation** - Add analysis-specific statistics
6. **Test thoroughly** - Run with all three glaciers
7. **Document changes** - Update docstrings and comments

### Code Quality Guidelines

- **Type hints everywhere** - Use typing for all function signatures
- **Comprehensive docstrings** - Explain purpose, parameters, returns, raises
- **Error handling** - Check for edge cases and invalid inputs
- **Consistent formatting** - Follow established indentation and spacing
- **Clear comments** - Explain complex logic and calculations
- **Logging throughout** - Track progress and issues

### Common Pitfalls to Avoid

- **Hard-coded paths** - Always use CONFIG for file paths
- **Missing error handling** - Check file existence and data validity
- **Inconsistent naming** - Follow established variable naming conventions
- **Poor logging** - Include context in log messages
- **Manual output paths** - Always use OutputManager for file operations
- **Missing type hints** - Include typing for all function parameters
- **Inadequate docstrings** - Document all classes and methods

## Testing Guidelines

### Manual Testing Steps

1. **Run with each glacier individually** - Test data loading for each
2. **Check output directory creation** - Verify plots and results folders
3. **Validate plot generation** - Ensure plots are created and saved
4. **Review summary files** - Check summary.txt and README.md content
5. **Test error conditions** - Try with missing files or invalid data

### Data Validation

- **Check data ranges** - Albedo values should be 0-1
- **Verify merging** - Ensure AWS and MODIS data align on dates
- **Validate statistics** - Check correlation values and error metrics
- **Confirm sample sizes** - Ensure adequate data for each analysis

This reference guide ensures consistency and quality across all analysis scripts in the glacier albedo research codebase.