# Multi-Glacier Seasonal Analysis Generator

A comprehensive standalone tool that creates **3×4 seasonal boxplot analysis** showing AWS vs MODIS method comparisons across summer months (June-September) for all three glaciers. This tool extends the single-glacier seasonal analysis format to provide comprehensive multi-glacier seasonal patterns in a unified visualization.

![Seasonal Analysis Layout](https://via.placeholder.com/800x600/FFFFFF/000000?text=3x4+Seasonal+Analysis+Grid)

## Features

### 🗓️ **Comprehensive Seasonal Analysis**
- **3×4 grid layout**: 3 glaciers (rows) × 4 months (columns) = 12 subplots
- **Summer focus**: June, July, August, September analysis
- **Consistent format**: Each subplot matches existing Haig Glacier analysis style
- **Cross-glacier comparison**: Easy seasonal pattern comparison between glaciers

### 📊 **Statistical Rigor**
- **Complete data pipeline**: Uses same processing as other analysis tools
- **Outlier filtering**: 2.5σ threshold for robust statistics  
- **Quality filtering**: Minimum glacier fraction and observation requirements
- **Sample size reporting**: (n=XX) display for each method boxplot

### 🎨 **Professional Visualization**
- **Method-specific colors**: AWS (red), MCD43A3 (green), MOD09GA (blue), MOD10A1 (orange)
- **Publication quality**: 300 DPI output with proper titles and legends
- **Consistent styling**: Matches existing analysis framework appearance
- **Large format**: 20×15 inch layout for detailed visualization

### 🔍 **Intelligent Processing**
- **Pixel selection**: Same algorithm as other generators (distance + glacier fraction)
- **Seasonal filtering**: Automatic extraction of summer months only
- **Data validation**: Comprehensive error handling and quality checks
- **Method mapping**: Handles case variations and Terra/Aqua combinations

## Output Format

### 📋 **Layout Structure**
```
┌─── ATHABASCA GLACIER ─────────────────────────────┐
│ June      │ July      │ August    │ September   │
│ AWS+MODIS │ AWS+MODIS │ AWS+MODIS │ AWS+MODIS   │
│ (n=XX)    │ (n=XX)    │ (n=XX)    │ (n=XX)      │
└───────────┴───────────┴───────────┴─────────────┘

┌─── HAIG GLACIER ──────────────────────────────────┐
│ June      │ July      │ August    │ September   │
│ AWS+MODIS │ AWS+MODIS │ AWS+MODIS │ AWS+MODIS   │
│ (n=XX)    │ (n=XX)    │ (n=XX)    │ (n=XX)      │
└───────────┴───────────┴───────────┴─────────────┘

┌─── COROPUNA GLACIER ──────────────────────────────┐
│ June      │ July      │ August    │ September   │
│ AWS+MODIS │ AWS+MODIS │ AWS+MODIS │ AWS+MODIS   │
│ (n=XX)    │ (n=XX)    │ (n=XX)    │ (n=XX)      │
└───────────┴───────────┴───────────┴─────────────┘
```

### 📊 **Each Subplot Contains**
- **AWS reference**: Red boxplot with sample size (n=XX)
- **MCD43A3**: Green boxplot with sample size (n=XX)
- **MOD09GA**: Blue boxplot with sample size (n=XX)  
- **MOD10A1**: Orange boxplot with sample size (n=XX)
- **Y-axis**: Albedo values (0.0 to 1.0)
- **Grid lines**: Light grid for easy value reading

## Installation & Setup

### 📋 **Requirements**
```python
pandas >= 1.3.0
numpy >= 1.20.0
matplotlib >= 3.4.0
scipy >= 1.7.0
pathlib (built-in)
logging (built-in)
```

### 💾 **Data Dependencies**

#### MODIS Data Files
- **Athabasca**: `D:/Documents/Projects/athabasca_analysis/data/csv/Athabasca_Terra_Aqua_MultiProduct_2014-01-01_to_2021-01-01.csv`
- **Haig**: `D:/Documents/Projects/Haig_analysis/data/csv/Haig_MODIS_Pixel_Analysis_MultiProduct_2002_to_2016_fraction.csv`
- **Coropuna**: `D:/Documents/Projects/Coropuna_glacier/data/csv/coropuna_glacier_2014-01-01_to_2025-01-01.csv`

#### AWS Data Files
- **Athabasca**: `D:/Documents/Projects/athabasca_analysis/data/csv/iceAWS_Atha_albedo_daily_20152020_filled_clean.csv`
- **Haig**: `D:/Documents/Projects/Haig_analysis/data/csv/HaigAWS_daily_2002_2015_gapfilled.csv`
- **Coropuna**: `D:/Documents/Projects/Coropuna_glacier/data/csv/COROPUNA_simple.csv`

## Usage

### 🚀 **Quick Start**
```bash
cd /path/to/project
python multi_glacier_seasonal_analysis_generator.py
```

### ⚙️ **Configuration**

Update the `CONFIG` dictionary to match your system:

```python
CONFIG = {
    'data_paths': {
        'glacier_name': {
            'modis': 'path/to/modis/data.csv',
            'aws': 'path/to/aws/data.csv'
        }
    },
    'seasonal_months': [6, 7, 8, 9],  # June, July, August, September
    'colors': {
        'AWS': '#d62728',        # Red
        'MCD43A3': '#2ca02c',    # Green
        'MOD09GA': '#1f77b4',    # Blue
        'MOD10A1': '#ff7f0e'     # Orange
    },
    'visualization': {
        'figsize': (20, 15),     # Large format
        'dpi': 300               # Publication quality
    }
}
```

## Data Format Requirements

### 📊 **MODIS Data Format**

The tool supports the same formats as other generators:

#### Long Format (Preferred - Coropuna)
```csv
pixel_id,date,method,albedo,latitude,longitude,glacier_fraction
1,2020-06-01,MCD43A3,0.85,-15.5181,-72.6617,0.95
```

#### Wide Format (Athabasca, Haig)
```csv
pixel_id,date,latitude,longitude,glacier_fraction,albedo_MCD43A3,albedo_MOD09GA
1,2020-06-01,52.1949,-117.2431,0.95,0.85,0.82
```

### 🌡️ **AWS Data Format**

#### Standard Format (Athabasca, Coropuna)
```csv
Time,Albedo          # or Timestamp,Albedo
2020-06-01,0.82
```

#### Haig Special Format
```csv
Year;Day;albedo       # Semicolon separated, European decimal (,)
2020;153;0,82        # Day 153 = June 1st
```

## Algorithm Details

### 🗓️ **Seasonal Filtering Process**

1. **Month Extraction**
   - Extract month from date columns in both AWS and MODIS data
   - Filter for summer months: June (6), July (7), August (8), September (9)
   - Preserve all years but focus on seasonal patterns

2. **Data Alignment**
   - Merge AWS and MODIS data on exact date matches
   - Apply same outlier filtering as other analysis tools (2.5σ threshold)
   - Maintain data quality standards with minimum observation requirements

3. **Method Processing**
   - Process each MODIS method separately (MCD43A3, MOD09GA, MOD10A1)
   - Combine Terra/Aqua variants (MOD/MYD) for consistency
   - Add AWS reference data for each glacier-month combination

### 📊 **Boxplot Generation**

1. **Data Grouping**
   - Group by glacier → month → method
   - Calculate sample sizes for each group
   - Handle missing data gracefully with "No data available" labels

2. **Statistical Processing**
   - Apply same outlier filtering as existing tools
   - Calculate quartiles, medians, and outliers for boxplots
   - Ensure minimum sample sizes for reliable statistics

3. **Visualization**
   - Create boxplots with method-specific colors
   - Add sample size labels: (n=XX) format
   - Include grid lines and proper axis scaling (0.0 to 1.0)

## Output Examples

### 📊 **Seasonal Patterns You Can Observe**

1. **Cross-Glacier Comparisons**
   - Compare June patterns: Athabasca vs Haig vs Coropuna
   - Identify glacier-specific seasonal behaviors
   - Observe method consistency across different geographic regions

2. **Temporal Trends**
   - Track albedo changes from June through September
   - Identify peak albedo months for each glacier
   - Compare seasonal timing between glaciers

3. **Method Performance**
   - Compare MODIS method accuracy across seasons
   - Identify months with best/worst method performance  
   - Observe seasonal bias patterns in satellite retrievals

### 📈 **Expected Results**

- **Summer progression**: Generally decreasing albedo from June to September
- **Glacier differences**: Northern glaciers (Athabasca, Haig) vs tropical (Coropuna)
- **Method variations**: Different MODIS products show varying seasonal accuracy
- **Sample size patterns**: More data in peak summer months (July, August)

## Troubleshooting

### ❌ **Common Issues**

#### No Data for Specific Months
```
"No data available" displayed in subplot
```
**Solutions**:
- Check if glacier has data for that specific month
- Verify seasonal month filter settings in CONFIG
- Confirm temporal overlap between AWS and MODIS data

#### Empty Plots After Processing
```
All subplots show "No data available"
```
**Solutions**:
- Verify file paths in CONFIG are correct
- Check date format compatibility
- Reduce quality filter thresholds
- Ensure seasonal months contain actual data

#### Method Not Appearing
```
Only AWS data visible, MODIS methods missing
```
**Solutions**:
- Check method name mapping in CONFIG
- Debug data loading to verify method columns exist
- Confirm method data exists for seasonal time periods

#### Sample Size Issues
```
Very low sample sizes (n=1, n=2)
```
**Solutions**:
- Check temporal resolution of input data
- Verify pixel selection isn't too restrictive
- Consider expanding seasonal month range
- Review outlier filtering threshold

### 🔧 **Debugging Tips**

1. **Enable detailed logging**:
   ```python
   logging.basicConfig(level=logging.DEBUG)
   ```

2. **Check seasonal data availability**:
   ```python
   # Add this after data loading to verify seasonal coverage
   print("Available months in data:")
   print(modis_data['date'].dt.month.value_counts().sort_index())
   ```

3. **Verify method availability by month**:
   ```python
   # Check method distribution across months
   seasonal_summary = modis_data.groupby(['month', 'method']).size()
   print(seasonal_summary)
   ```

## Customization

### 🗓️ **Seasonal Period Adjustment**
```python
CONFIG['seasonal_months'] = [5, 6, 7, 8, 9]  # May through September
CONFIG['month_names'] = {5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September'}
```

### 🎨 **Visual Customization**
```python
CONFIG['colors'] = {
    'AWS': '#your_color',
    'MCD43A3': '#your_color',
    'MOD09GA': '#your_color',  
    'MOD10A1': '#your_color'
}

CONFIG['visualization'] = {
    'figsize': (width, height),  # Adjust size
    'dpi': 300,                  # Resolution
    'style': 'your_style'        # Matplotlib style
}
```

### 📊 **Statistical Parameters**
```python
CONFIG['outlier_threshold'] = 3.0      # Change outlier detection
CONFIG['quality_filters'] = {
    'min_glacier_fraction': 0.05,       # Lower threshold
    'min_observations': 5               # Fewer required observations
}
```

### 🌍 **Adding New Glaciers**
```python
CONFIG['data_paths']['new_glacier'] = {
    'modis': 'path/to/new/modis.csv',
    'aws': 'path/to/new/aws.csv'
}
CONFIG['aws_stations']['new_glacier'] = {
    'lat': latitude, 'lon': longitude, 'name': 'Station Name'
}
```

## Comparison with Other Generators

This seasonal analysis tool complements the existing analysis suite:

| Feature | Seasonal Generator | Statistical Generators | Map Generator |
|---------|-------------------|----------------------|---------------|
| **Purpose** | Seasonal patterns & trends | Performance metrics | Spatial relationships |
| **Layout** | 3×4 temporal grid | Statistical matrices | Geographic maps |
| **Focus** | Monthly comparisons | Overall performance | Pixel selection |
| **Data Pipeline** | Identical (same classes) | Identical (same classes) | Identical (same classes) |
| **Time Dimension** | Seasonal (June-Sep) | All available data | Static (location only) |
| **Use Case** | Temporal analysis | Quantitative assessment | Spatial validation |

## Scientific Applications

### 📈 **Research Questions This Tool Addresses**

1. **Seasonal Albedo Patterns**
   - How does glacier albedo change through the summer melt season?
   - Are seasonal patterns consistent across different glaciers?
   - Which months show the highest/lowest albedo values?

2. **Method Seasonal Performance**
   - Do MODIS methods perform better in certain months?
   - Are there seasonal biases in satellite retrievals?
   - Which method provides most consistent seasonal tracking?

3. **Climate/Geographic Effects**
   - How do northern vs tropical glaciers compare seasonally?
   - Do latitude effects influence seasonal albedo patterns?
   - Are there elevation-related seasonal differences?

### 📊 **Publication Use Cases**

- **Seasonal trend analysis**: Compare multi-year seasonal patterns
- **Method validation papers**: Demonstrate seasonal consistency
- **Climate change studies**: Document changing seasonal patterns
- **Comparative glaciology**: Cross-regional seasonal behavior

## Citation

If you use this tool in your research, please cite:

```
Multi-Glacier Seasonal Analysis Generator
Generated from Multi-Glacier Albedo Analysis Framework
[Your Institution/Project Name]
2025
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Verify your data files match the expected seasonal coverage
3. Review the console output for specific error messages
4. Test with the provided sample configuration first

---

**Generated from**: Multi-Glacier Albedo Analysis Framework  
**Version**: 1.0  
**Date**: July 31, 2025  
**Compatibility**: Python 3.7+