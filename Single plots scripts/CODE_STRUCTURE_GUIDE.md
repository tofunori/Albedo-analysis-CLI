# AWS vs MODIS Scatterplot Generator - Code Structure Guide

## Overview

The AWS vs MODIS scatterplot generator has been completely restructured into a clean, modular architecture that separates concerns and makes it easy to modify individual components while maintaining the same functionality.

## Architecture

```
PIPELINE: Data Loading → Pixel Selection → Data Processing → Visualization

┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   DataLoader    │ -> │ PixelSelector   │ -> │ DataProcessor   │ -> │ Visualizer      │
│                 │    │                 │    │                 │    │                 │
│ • Load MODIS    │    │ • Quality       │    │ • Merge AWS/    │    │ • Create 3x3    │
│ • Load AWS      │    │   filters       │    │   MODIS data    │    │   scatterplots  │
│ • Format        │    │ • Distance      │    │ • Outlier       │    │ • Add stats     │
│   standardize   │    │   calculation   │    │   filtering     │    │ • Save plots    │
│ • Validation    │    │ • Best pixel    │    │ • Statistics    │    │                 │
│                 │    │   selection     │    │   calculation   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Classes and Their Responsibilities

### 1. DataLoader
**Purpose**: Handles all data loading and preprocessing
**Key Features**:
- Glacier-specific data format handling
- Automatic format detection (wide vs long format)
- Data validation and cleaning
- Consistent output format across glaciers

**Main Methods**:
- `load_glacier_data()`: Main entry point for loading data
- `_load_modis_data()`: MODIS-specific loading with format conversion
- `_load_aws_data()`: AWS-specific loading with format standardization
- `_convert_to_long_format()`: Converts wide format data to long format

### 2. PixelSelector
**Purpose**: Implements intelligent pixel selection based on quality metrics
**Key Features**:
- Distance calculation to AWS stations using Haversine formula
- Quality filtering (glacier fraction, observation count)
- Glacier-specific selection strategies
- Composite ranking system

**Main Methods**:
- `select_best_pixels()`: Main pixel selection pipeline
- `_calculate_pixel_metrics()`: Quality metric calculation
- `_apply_quality_filters()`: Filter low-quality pixels
- `_calculate_aws_distances()`: Distance calculations

### 3. DataProcessor
**Purpose**: Handles data merging, outlier filtering, and statistical analysis
**Key Features**:
- Temporal alignment of AWS and MODIS data
- Configurable outlier filtering (2.5σ threshold)
- Comprehensive statistical analysis
- Method-specific processing

**Main Methods**:
- `merge_and_process()`: Main processing pipeline
- `_process_single_method()`: Process individual MODIS methods
- `_apply_outlier_filtering()`: Statistical outlier removal
- `_calculate_comprehensive_statistics()`: Full statistical suite

### 4. ScatterplotVisualizer
**Purpose**: Creates publication-ready visualizations
**Key Features**:
- 3×3 scatterplot matrix layout
- Consistent styling and color schemes
- Statistical annotations
- High-resolution output

**Main Methods**:
- `create_scatterplot_matrix()`: Main visualization pipeline
- `_create_data_scatterplot()`: Individual subplot creation
- `_add_statistics_textbox()`: Statistical annotations
- `_save_figure()`: High-quality figure output

### 5. AnalysisPipeline
**Purpose**: Orchestrates the complete analysis workflow
**Key Features**:
- End-to-end pipeline coordination
- Error handling and logging
- Results summary and validation
- Component integration

**Main Methods**:
- `run_complete_analysis()`: Execute full pipeline
- `_process_single_glacier()`: Process individual glaciers
- `_generate_visualization()`: Create final plots
- `_log_final_summary()`: Analysis summary

## Configuration System

All settings are centralized in the `CONFIG` dictionary:

```python
CONFIG = {
    'data_paths': {...},          # File paths for each glacier
    'aws_stations': {...},        # AWS station coordinates
    'colors': {...},              # Visualization color scheme
    'methods': [...],             # MODIS methods to analyze
    'method_mapping': {...},      # Standardize method names
    'outlier_threshold': 2.5,     # Statistical filtering threshold
    'quality_filters': {...},     # Pixel quality requirements
    'visualization': {...}        # Plot settings and styling
}
```

## Usage Examples

### Simple Usage (Recommended)
```python
from aws_vs_modis_scatterplot_generator import main

# Run complete analysis with default settings
success = main()
```

### Advanced Usage with Custom Configuration
```python
from aws_vs_modis_scatterplot_generator import AnalysisPipeline, CONFIG

# Modify configuration
CONFIG['outlier_threshold'] = 3.0  # More lenient outlier filtering
CONFIG['visualization']['figsize'] = (20, 16)  # Larger plots

# Run with custom settings
pipeline = AnalysisPipeline(CONFIG)
success = pipeline.run_complete_analysis("custom_output.png")
```

### Component-Level Usage
```python
# Use individual components for custom workflows
data_loader = DataLoader(CONFIG)
pixel_selector = PixelSelector(CONFIG)

# Load and process specific glacier
modis_data, aws_data = data_loader.load_glacier_data('athabasca')
selected_modis = pixel_selector.select_best_pixels(modis_data, 'athabasca')
```

## Modification Guide

### Adding a New Glacier
1. Add data paths to `CONFIG['data_paths']`
2. Add AWS coordinates to `CONFIG['aws_stations']`
3. Add color scheme to `CONFIG['colors']`
4. Update glacier list in main functions

### Modifying Pixel Selection Strategy
- Edit `PixelSelector._apply_selection_strategy()`
- Adjust `CONFIG['quality_filters']` for different thresholds
- Modify `_rank_pixels_by_quality()` for different ranking criteria

### Changing Statistical Analysis
- Edit `DataProcessor._calculate_comprehensive_statistics()`
- Modify outlier filtering in `_apply_outlier_filtering()`
- Adjust `CONFIG['outlier_threshold']`

### Customizing Visualizations
- Edit `ScatterplotVisualizer._create_data_scatterplot()`
- Modify `CONFIG['visualization']` settings
- Adjust `_format_statistics_text()` for different metrics

## Benefits of New Structure

### ✅ Maintainability
- Clear separation of concerns
- Modular design enables independent testing
- Easy to locate and modify specific functionality

### ✅ Extensibility
- Easy to add new glaciers or MODIS methods
- Component-based architecture supports new features
- Configuration-driven behavior

### ✅ Reliability
- Comprehensive error handling
- Detailed logging throughout pipeline
- Input validation and data quality checks

### ✅ Usability
- Simple main function for basic usage
- Advanced pipeline class for custom workflows
- Clear documentation and examples

### ✅ Performance
- Efficient data processing with pandas vectorization
- Minimal memory usage with streaming approach
- Optimized visualization rendering

## Legacy Compatibility

The original functionality is preserved through `main_legacy()` function, ensuring existing workflows continue to work while providing access to the new structured approach.
