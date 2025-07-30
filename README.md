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
# Interactive mode (default)
python interactive_main.py

# Command-line mode for automation
python interactive_main.py --glacier haig --analysis-mode comprehensive --selected-pixels
python interactive_main.py --all-glaciers --analysis-mode basic
python interactive_main.py --comparative-analysis

# See all options
python interactive_main.py --help
```

## Key Components

- `interactive_main.py` - **Single entry point** with both interactive and command-line interfaces
- `albedo_engine/` - Core unified analysis engine
- `data_processing/` - Data loading and processing modules  
- `analysis/` - Analysis components (core, comparative, spatial)
- `visualization/` - Comprehensive visualization suite
- `utils/` - Configuration, validation, and system utilities
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
