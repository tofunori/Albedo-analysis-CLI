# Usage Examples

This document provides practical examples of how to use the MODIS Albedo Analysis Framework for different scenarios.

## Quick Start Example

### 1. Basic Single Glacier Analysis

Process one glacier with default settings:

```bash
# Process glacier_1 with default configuration
python main.py --glacier glacier_1

# Process with custom configuration
python main.py --glacier glacier_1 --config config/custom_config.yaml
```

Expected output:
```
2024-01-28 14:30:22 - INFO - Starting analysis for glacier: glacier_1
2024-01-28 14:30:23 - INFO - Loading data...
2024-01-28 14:30:24 - INFO - Loaded MOD09GA: 1250 records
2024-01-28 14:30:24 - INFO - Loaded MOD10A1: 890 records
2024-01-28 14:30:25 - INFO - Loaded MCD43A3: 1100 records
2024-01-28 14:30:25 - INFO - Loaded AWS data: 2500 records
2024-01-28 14:30:26 - INFO - Processing spatial data...
2024-01-28 14:30:27 - INFO - Aligning temporal data...
2024-01-28 14:30:28 - INFO - Aligned MOD09GA: 145 common dates
2024-01-28 14:30:28 - INFO - Aligned MOD10A1: 98 common dates  
2024-01-28 14:30:28 - INFO - Aligned MCD43A3: 132 common dates
2024-01-28 14:30:29 - INFO - Performing statistical analysis...
2024-01-28 14:30:30 - INFO - Generating plots...
2024-01-28 14:30:35 - INFO - Generating maps...
2024-01-28 14:30:37 - INFO - Exporting results...
2024-01-28 14:30:38 - INFO - Analysis completed for glacier: glacier_1
Analysis completed for 1 glacier(s)
```

### 2. Batch Processing All Glaciers

Process all glaciers defined in configuration:

```bash
# Process all glaciers and generate summary
python main.py --all-glaciers --output-summary batch_results_summary.csv
```

This creates individual output directories for each glacier plus a summary CSV:

```csv
glacier_id,analysis_timestamp,output_directory
glacier_1,20240128_143022,outputs/glacier_1_20240128_143022
glacier_2,20240128_143045,outputs/glacier_2_20240128_143045
glacier_3,20240128_143108,outputs/glacier_3_20240128_143108
glacier_4,20240128_143131,outputs/glacier_4_20240128_143131
```

## Configuration Examples

### 3. Custom Analysis Configuration

Create a custom configuration file `config/high_quality_config.yaml`:

```yaml
# High-quality analysis configuration
analysis:
  albedo:
    modis_products:
      - MOD09GA
      - MOD10A1
      - MCD43A3
    quality_filters:
      cloud_threshold: 0.1  # Stricter cloud filtering
      snow_threshold: 0.05   # Stricter snow filtering
      
  statistics:
    metrics:
      - rmse
      - bias
      - correlation
      - mae
      - nse         # Nash-Sutcliffe Efficiency
      - kge         # Kling-Gupta Efficiency
    confidence_level: 0.99  # Higher confidence level
    
  spatial:
    pixel_size: 250          # Higher resolution
    buffer_distance: 500     # Smaller buffer around AWS

visualization:
  style: "seaborn-v0_8-paper"  # Publication style
  dpi: 600                     # High DPI for publications
  figure_size: [12, 9]         # Larger figures
  
processing:
  parallel: true
  n_cores: 4
  memory_limit: "16GB"

logging:
  level: "DEBUG"  # Detailed logging
```

Use the custom configuration:

```bash
python main.py --glacier glacier_1 --config config/high_quality_config.yaml
```

### 4. Glacier Site Configuration Example

Add a new glacier to `config/glacier_sites.yaml`:

```yaml
glaciers:
  # ... existing glaciers ...
  
  new_glacier:
    name: "Vatnajökull Outlet"
    region: "Iceland"
    coordinates:
      lat: 64.0
      lon: -16.8
    data_files:
      modis:
        MOD09GA: "vatnajokull_mod09ga_2020_2023.csv"
        MOD10A1: "vatnajokull_mod10a1_2020_2023.csv"
        MCD43A3: "vatnajokull_mcd43a3_2020_2023.csv"
      aws: "vatnajokull_aws_2020_2023.csv"
      mask: "vatnajokull_outline.shp"
    aws_stations:
      vat_aws_01:
        name: "Vatnajökull AWS 1"
        lat: 64.01
        lon: -16.79
        elevation: 1500
      vat_aws_02:
        name: "Vatnajökull AWS 2"  
        lat: 63.99
        lon: -16.81
        elevation: 1800
```

## Data Preparation Examples

### 5. MODIS Data Format Examples

**MOD09GA Surface Reflectance CSV:**
```csv
date,lat,lon,red_reflectance,nir_reflectance,blue_reflectance,green_reflectance,quality_flag
2023-01-01,65.123,-18.456,0.234,0.567,0.123,0.345,0.05
2023-01-02,65.125,-18.454,0.198,0.623,0.098,0.289,0.02
2023-01-03,65.121,-18.458,0.267,0.534,0.145,0.378,0.08
```

**MOD10A1 Snow Cover CSV:**
```csv
date,lat,lon,snow_albedo,snow_cover,quality_flag
2023-01-01,65.123,-18.456,75.5,95.2,0.03
2023-01-02,65.125,-18.454,82.1,98.7,0.01
2023-01-03,65.121,-18.458,68.9,89.4,0.06
```

**MCD43A3 BRDF/Albedo CSV:**
```csv
date,lat,lon,white_sky_albedo,black_sky_albedo,quality_flag
2023-01-01,65.123,-18.456,0.785,0.723,0.02
2023-01-02,65.125,-18.454,0.821,0.756,0.01
2023-01-03,65.121,-18.458,0.689,0.634,0.04
```

### 6. AWS Data Format Example

**AWS Albedo Measurements CSV:**
```csv
date,albedo,station_id,temperature,wind_speed,solar_radiation
2023-01-01 12:00:00,0.78,AWS_01,-2.3,5.6,234.5
2023-01-01 13:00:00,0.81,AWS_01,-1.9,4.8,287.2
2023-01-01 14:00:00,0.76,AWS_01,-1.2,6.2,312.8
2023-01-01 12:30:00,0.82,AWS_02,-3.1,7.1,225.6
2023-01-01 13:30:00,0.85,AWS_02,-2.7,6.9,275.4
```

### 7. Data Validation Example

Check data quality before analysis:

```python
from src.data.data_processor import DataProcessor
from src.utils.helpers import load_config

# Load configuration
config = load_config('config/config.yaml')
processor = DataProcessor(config)

# Load and validate MODIS data
from src.data.modis_loader import create_modis_loader
mod09ga_loader = create_modis_loader('MOD09GA', config)
modis_data = mod09ga_loader.load_data('data/modis/glacier_1_mod09ga.csv')

# Generate data quality report
report = processor.generate_data_report(modis_data, 'MOD09GA')

print("Data Quality Report:")
print(f"Total records: {report['basic_stats']['record_count']}")
print(f"Missing values: {report['basic_stats']['missing_values']}")
print(f"Recommendations: {report['recommendations']}")
```

## Analysis Examples

### 8. Custom Statistical Analysis

Perform detailed statistical analysis with custom metrics:

```python
from src.analysis.statistical_analysis import StatisticalAnalyzer
import pandas as pd
import numpy as np

# Initialize analyzer
config = load_config('config/config.yaml')
analyzer = StatisticalAnalyzer(config)

# Load aligned data (example)
aws_data = pd.Series([0.78, 0.82, 0.76, 0.81, 0.79], name='AWS')
mod09ga_data = pd.Series([0.76, 0.84, 0.73, 0.83, 0.77], name='MOD09GA')

# Calculate comprehensive metrics
metrics = analyzer.calculate_basic_metrics(aws_data, mod09ga_data)

print("Statistical Metrics:")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"Bias: {metrics['bias']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
print(f"Nash-Sutcliffe Efficiency: {metrics['nse']:.4f}")
print(f"Kling-Gupta Efficiency: {metrics['kge']:.4f}")

# Calculate confidence intervals
ci = analyzer.calculate_confidence_intervals(aws_data, mod09ga_data)
print(f"RMSE 95% CI: [{ci['rmse']['lower']:.4f}, {ci['rmse']['upper']:.4f}]")
```

### 9. Seasonal Analysis Example

Analyze seasonal patterns in albedo data:

```python
from src.analysis.statistical_analysis import StatisticalAnalyzer

# Create seasonal data
dates = pd.date_range('2023-01-01', periods=365, freq='D')
# Simulate seasonal albedo pattern
day_of_year = dates.dayofyear
seasonal_albedo = 0.5 + 0.3 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
noise = np.random.normal(0, 0.05, len(dates))

data = pd.DataFrame({
    'date': dates,
    'albedo': seasonal_albedo + noise
})

# Analyze seasonal patterns
analyzer = StatisticalAnalyzer(config)
seasonal_stats = analyzer.calculate_seasonal_statistics(data)

print("Seasonal Statistics:")
for season, stats in seasonal_stats['seasonal'].items():
    print(f"{season}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
```

## Visualization Examples

### 10. Custom Plot Generation

Generate specific plots programmatically:

```python
from src.visualization.plots import PlotGenerator
import matplotlib.pyplot as plt

# Initialize plot generator
config = load_config('config/config.yaml')
plot_gen = PlotGenerator(config)

# Create custom scatterplot
fig = plot_gen.create_scatterplot(
    aws_data, mod09ga_data,
    x_label="AWS Albedo",
    y_label="MOD09GA Albedo", 
    title="Glacier 1: MOD09GA Validation",
    method_name="MOD09GA",
    show_stats=True,
    add_regression_line=True,
    output_path="custom_scatterplot.png"
)

# Create multi-method comparison
modis_methods = {
    'MOD09GA': mod09ga_data,
    'MOD10A1': mod10a1_data,
    'MCD43A3': mcd43a3_data
}

fig = plot_gen.create_multi_method_scatterplot(
    aws_data, modis_methods,
    title="Multi-Method Comparison",
    output_path="multi_method_comparison.png"
)

plt.show()
```

### 11. Custom Color Schemes

Use custom colors for different glacier types or regions:

```yaml
# In config/config.yaml
visualization:
  colors:
    MOD09GA: "#2E86AB"      # Blue for surface reflectance
    MOD10A1: "#A23B72"      # Purple for snow products  
    MCD43A3: "#F18F01"      # Orange for BRDF products
    AWS: "#C73E1D"          # Red for ground truth
    
    # Region-specific colors
    arctic_glaciers: "#1B4F72"
    alpine_glaciers: "#27AE60" 
    antarctic_glaciers: "#8E44AD"
```

## Spatial Analysis Examples

### 12. Working with Glacier Masks

Process different types of glacier masks:

```python
from src.mapping.glacier_masks import GlacierMaskProcessor

# Initialize processor
mask_processor = GlacierMaskProcessor(config)

# Load shapefile mask
shapefile_mask = mask_processor.load_and_validate_mask('data/glacier_masks/glacier_1.shp')

# Load raster mask
raster_mask = mask_processor.load_and_validate_mask('data/glacier_masks/glacier_1.tif')

# Calculate glacier properties
properties = mask_processor.calculate_glacier_properties(shapefile_mask)
print(f"Glacier area: {properties['total_area_km2']:.2f} km²")
print(f"Glacier centroid: {properties['centroid_lat']:.4f}°N, {properties['centroid_lon']:.4f}°W")

# Create elevation zones (if DEM available)
elevation_zones = mask_processor.create_elevation_zones(
    shapefile_mask, 
    dem_path='data/elevation/glacier_1_dem.tif',
    elevation_bands=[0, 500, 1000, 1500, 2000, 3000]
)
```

### 13. Spatial Data Export

Export spatial results in different formats:

```python
from src.mapping.spatial_utils import SpatialProcessor

spatial_processor = SpatialProcessor(config)

# Export glacier mask with MODIS pixels
# (Assuming you have processed spatial data)
spatial_processor.export_spatial_data(
    processed_modis_gdf, 
    'outputs/glacier_1_modis_pixels.shp',
    format='shapefile'
)

# Export as GeoJSON for web applications
spatial_processor.export_spatial_data(
    processed_modis_gdf,
    'outputs/glacier_1_modis_pixels.geojson', 
    format='geojson'
)
```

## Advanced Usage Examples

### 14. Parallel Processing Configuration

Configure for high-performance computing:

```yaml
# config/hpc_config.yaml
processing:
  parallel: true
  n_cores: 32              # Use all available cores
  chunk_size: 5000         # Larger chunks for HPC
  memory_limit: "128GB"    # High memory limit

# Enable distributed processing (if using Dask)
distributed:
  scheduler_address: "tcp://scheduler:8786"
  dashboard_address: ":8787"
```

### 15. Custom Quality Control

Implement custom quality filtering:

```python
from src.data.modis_loader import MOD09GALoader

class CustomMOD09GALoader(MOD09GALoader):
    def quality_filter(self, data):
        """Custom quality filtering for specific region/conditions."""
        # Apply base quality filtering
        filtered_data = super().quality_filter(data)
        
        # Add custom filters
        # Example: Remove data with extreme view angles
        if 'view_angle' in filtered_data.columns:
            filtered_data = filtered_data[filtered_data['view_angle'] < 60]
        
        # Example: Seasonal filtering for Arctic regions
        if self.is_arctic_region():
            # Only use data from April to September
            filtered_data = filtered_data[
                filtered_data['date'].dt.month.between(4, 9)
            ]
        
        return filtered_data
    
    def is_arctic_region(self):
        # Check if this is an Arctic glacier based on coordinates
        return self.glacier_lat > 66.5  # Arctic Circle
```

### 16. Integration with External Data

Integrate with meteorological data:

```python
def load_weather_data(glacier_config, date_range):
    """Load weather data for enhanced analysis."""
    # Example: Load ERA5 reanalysis data
    weather_data = pd.read_csv('data/weather/era5_glacier_1.csv')
    weather_data['date'] = pd.to_datetime(weather_data['date'])
    
    # Filter to date range
    mask = weather_data['date'].between(date_range[0], date_range[1])
    return weather_data[mask]

def enhanced_analysis(glacier_id):
    """Perform analysis with weather context."""
    # Load standard data
    results = pipeline.process_glacier(glacier_id)
    
    # Load weather data
    weather = load_weather_data(glacier_config, date_range)
    
    # Analyze relationships
    # Example: Albedo vs temperature
    correlation = results['aligned_data']['AWS'].corr(weather['temperature'])
    print(f"Albedo-temperature correlation: {correlation:.3f}")
    
    return results
```

## Troubleshooting Examples

### 17. Debugging Data Issues

Common data problems and solutions:

```python
# Check for common data issues
def diagnose_data_problems(data, data_type):
    """Diagnose common data quality issues."""
    issues = []
    
    # Check for missing dates
    if 'date' in data.columns:
        date_gaps = data['date'].diff().dt.days
        large_gaps = date_gaps[date_gaps > 10].count()
        if large_gaps > 0:
            issues.append(f"{large_gaps} gaps > 10 days in {data_type} data")
    
    # Check coordinate consistency
    if 'lat' in data.columns and 'lon' in data.columns:
        lat_range = data['lat'].max() - data['lat'].min()
        lon_range = data['lon'].max() - data['lon'].min()
        if lat_range > 1 or lon_range > 1:  # > ~100 km span
            issues.append(f"Large spatial extent in {data_type}: {lat_range:.3f}° lat, {lon_range:.3f}° lon")
    
    # Check albedo values
    if 'albedo' in data.columns:
        invalid_albedo = ((data['albedo'] < 0) | (data['albedo'] > 1)).sum()
        if invalid_albedo > 0:
            issues.append(f"{invalid_albedo} invalid albedo values in {data_type}")
    
    return issues

# Usage
modis_issues = diagnose_data_problems(modis_data, 'MODIS')
aws_issues = diagnose_data_problems(aws_data, 'AWS')

for issue in modis_issues + aws_issues:
    print(f"WARNING: {issue}")
```

### 18. Memory Optimization

Handle large datasets efficiently:

```python
def process_large_glacier(glacier_id, chunk_size=1000):
    """Process large glacier dataset in chunks."""
    
    # Load data in chunks
    modis_chunks = []
    for chunk in pd.read_csv('large_modis_file.csv', chunksize=chunk_size):
        # Process chunk
        processed_chunk = preprocess_modis_chunk(chunk)
        modis_chunks.append(processed_chunk)
    
    # Combine results
    full_modis_data = pd.concat(modis_chunks, ignore_index=True)
    
    # Continue with normal processing
    return process_glacier_data(full_modis_data)

def preprocess_modis_chunk(chunk):
    """Preprocess individual chunk to reduce memory usage."""
    # Remove unnecessary columns
    keep_columns = ['date', 'lat', 'lon', 'albedo', 'quality_flag']
    chunk = chunk[keep_columns]
    
    # Convert to appropriate data types
    chunk['date'] = pd.to_datetime(chunk['date'])
    chunk['albedo'] = chunk['albedo'].astype('float32')  # Use less memory
    chunk['quality_flag'] = chunk['quality_flag'].astype('uint8')
    
    return chunk
```

These examples demonstrate the flexibility and power of the MODIS Albedo Analysis Framework. Adapt them to your specific research needs and data characteristics.