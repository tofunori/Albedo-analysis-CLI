# MODIS Albedo Analysis Framework

A comprehensive Python framework for analyzing glacier albedo using MODIS satellite data with AWS weather station validation.

## Features

- **Multi-glacier analysis** (Athabasca, Haig, Coropuna)
- **7 enhanced visualization types** (comprehensive analysis, temporal, distribution, outlier, seasonal, correlation & bias analysis)
- **Intelligent pixel selection** (2 closest best-performing pixels to AWS stations)
- **Spatial mapping** with analysis pixel highlighting
- **Statistical validation** (correlation, RMSE, bias, MAE with outlier detection)
- **Interactive interface** for easy glacier selection and analysis

## Quick Start

```bash
# Interactive mode
python interactive_main.py

# Direct analysis
python pivot_based_main.py
```

## Key Components

- `pivot_based_main.py` - Enhanced analysis pipeline with 7 plot types
- `interactive_main.py` - User-friendly menu interface
- `src/` - Core analysis modules and data loaders
- `config/` - Glacier configurations and settings

## Requirements

- Python 3.8+
- pandas, numpy, matplotlib, scipy
- geopandas (for spatial mapping)

## Output

Each analysis generates:
- **7 visualization plots** in `plots/` directory
- **Spatial maps** with pixel highlighting in `maps/` directory  
- **Statistical results** in CSV format in `results/` directory

## Supported Glaciers

- **Athabasca Glacier** (Canadian Rockies)
- **Haig Glacier** (Canadian Rockies)
- **Coropuna Glacier** (Peruvian Andes)

Each glacier supports both standard analysis (all pixels) and enhanced analysis (selected best pixels).
