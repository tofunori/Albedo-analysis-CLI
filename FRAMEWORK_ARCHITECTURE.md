# MODIS Albedo Analysis Framework Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Detailed File Structure](#detailed-file-structure)
4. [Module Interactions](#module-interactions)
5. [Entry Point System](#entry-point-system)
6. [Data Flow](#data-flow)
7. [Configuration System](#configuration-system)
8. [Analysis Pipelines](#analysis-pipelines)
9. [Visualization Suite](#visualization-suite)
10. [Usage Examples](#usage-examples)
11. [Developer Guide](#developer-guide)

---

## System Overview

The MODIS Albedo Analysis Framework is a professionally reorganized Python-based system for analyzing glacier albedo using satellite data from MODIS sensors. It compares three MODIS albedo products (MOD09GA, MOD10A1, MCD43A3) against ground-truth AWS (Automatic Weather Station) measurements with a clean, modular architecture.

### Key Features
- **Single Unified Entry Point**: Interactive + command-line interface in one script
- **Professional Module Organization**: Clean separation of concerns across logical modules
- **Multiple Analysis Modes**: Auto, basic, enhanced, and comprehensive analysis
- **Intelligent Pixel Selection**: Distance + glacier fraction weighted optimization
- **Comprehensive Visualization**: 7-plot suite + spatial maps + interactive components
- **Advanced Statistical Analysis**: Correlation, RMSE, bias, outlier detection with 2.5σ threshold
- **Multi-Glacier Comparisons**: Cross-glacier statistical testing and regional analysis
- **Publication-Ready Outputs**: High-resolution plots and comprehensive data exports
- **Robust Error Handling**: Graceful degradation with missing dependencies

---

## Architecture Diagram

```
                       🎯 SINGLE UNIFIED ENTRY POINT 🎯
         ┌─────────────────────────────────────────────────────────────────┐
         │                  interactive_main.py                           │
         │                                                                 │
         │  ┌─────────────────┐              ┌─────────────────────────┐   │
         │  │ INTERACTIVE MODE│              │   COMMAND-LINE MODE     │   │
         │  │                 │              │                         │   │
         │  │ • Menu System   │              │ • Automated Processing │   │
         │  │ • Glacier Select│              │ • Batch Operations     │   │
         │  │ • Data Valid.   │              │ • CI/CD Integration    │   │
         │  │ • Progress Track│              │ • Script-Friendly      │   │
         │  └─────────────────┘              └─────────────────────────┘   │
         └─────────────────────────────┬───────────────────────────────────┘
                                       │
                                       ▼
    ═══════════════════════════════════════════════════════════════════════
                           🔧 PROFESSIONAL MODULE ARCHITECTURE 🔧

    ┌─────────────────────────────────────────────────────────────────────┐
    │                     CORE ENGINE LAYER                               │
    │                                                                     │
    │ 📁 albedo_engine/                                                   │
    │ ├── engine.py              🔧 UNIFIED ANALYSIS ENGINE               │
    │ │   • Multi-mode analysis (auto/basic/enhanced/comprehensive)      │
    │ │   • Pixel selection algorithms                                   │
    │ │   • Statistical analysis coordination                            │
    │ │   • Output generation management                                 │
    │ └── modes/                 ⚙️ Analysis mode definitions            │
    └─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                   DATA PROCESSING LAYER                             │
    │                                                                     │
    │ 📁 data_processing/                                                 │
    │ ├── loaders/               📥 DATA INPUT MODULES                    │
    │ │   ├── pivot_loaders.py   • Pivot-based data loading              │
    │ │   ├── modis_loaders.py   • MODIS satellite data                  │
    │ │   ├── aws_loaders.py     • Weather station data                  │
    │ │   └── csv_loaders.py     • Generic CSV handling                  │
    │ └── processors/            ⚙️ DATA PROCESSING MODULES              │
    │     ├── pivot_processor.py • Terra/Aqua merging                    │
    │     └── data_processor.py  • Quality filtering & validation        │
    └─────────────────────────────────────────────────────────────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    │                  │                  │
                    ▼                  ▼                  ▼
    ┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
    │   ANALYSIS LAYER    │  │ SPATIAL ANALYSIS    │  │ VISUALIZATION LAYER │
    │                     │  │      LAYER          │  │                     │
    │ 📁 analysis/        │  │                     │  │ 📁 visualization/   │
    │ ├── core/           │  │ 📁 spatial_analysis/│  │ ├── plots/          │
    │ │   ├── statistical │  │ ├── coordinates/    │  │ │   ├── statistical │
    │ │   ├── outlier     │  │ │   └── spatial_utils│  │ │   └── time_series │
    │ │   └── albedo_calc │  │ ├── masks/          │  │ ├── maps/           │
    │ ├── comparative/    │  │ │   └── glacier_masks│  │ │   └── map_generator│
    │ │   ├── interface   │  │ └── visualization/  │  │ └── interactive/    │
    │ │   ├── multi_glac  │  │     └── spatial_maps│  │     └── dashboard   │
    │ │   └── stat_tests  │  │                     │  │                     │
    │ └── spatial/        │  │ 🗺️ SPATIAL FEATURES:│  │ 📊 VISUALIZATION:   │
    │     ├── pixel_sel   │  │ • Coordinate systems │  │ • 7-plot suite     │
    │     ├── glacier_map │  │ • Glacier boundaries │  │ • Statistical plots │
    │     └── multi_plots │  │ • Map generation     │  │ • Spatial maps      │
    │                     │  │ • Overlay creation   │  │ • Interactive dashb │
    │ 🧮 ANALYSIS TYPES:  │  └─────────────────────┘  └─────────────────────┘
    │ • Core statistics   │
    │ • Multi-glacier     │
    │ • Spatial analysis  │
    │ • Pixel selection   │
    └─────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    UTILITIES LAYER                                  │
    │                                                                     │
    │ 📁 utils/                                                           │
    │ ├── config/                ⚙️ CONFIGURATION MANAGEMENT             │
    │ │   └── helpers.py         • Config loading, logging setup          │
    │ ├── data/                  ✅ DATA VALIDATION                       │
    │ │   └── validation.py      • Data quality checks, structure valid   │
    │ ├── system/                🔧 SYSTEM DIAGNOSTICS                    │
    │ │   └── diagnostics.py     • Health monitoring, dependency check    │
    │ └── logging/               📝 LOGGING ENHANCEMENTS                  │
    │     └── __init__.py        • Future custom logging features         │
    └─────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                              📂 OUTPUTS & RESULTS 📂
                    ┌─────────────────────────────────────┐
                    │ • High-resolution plots            │
                    │ • Statistical analysis results     │
                    │ • Spatial maps with overlays       │
                    │ • Comprehensive CSV exports        │
                    │ • Publication-ready visualizations │
                    └─────────────────────────────────────┘
```

---

## Detailed File Structure

### 📁 Root Level - Entry Point
```
interactive_main.py                    🎯 SINGLE UNIFIED ENTRY POINT
├── InteractiveGlacierAnalysis         # Main application class
├── Interactive Mode (default)         • Menu-driven glacier selection
│                                     • Data availability validation  
│                                     • Analysis mode selection
│                                     • Real-time progress tracking
└── Command-Line Mode                  • Automated batch processing
                                      • CI/CD integration support
                                      • Script-friendly operation
                                      • --help documentation

Usage Examples:
  python interactive_main.py                                    # Interactive
  python interactive_main.py --glacier haig --selected-pixels  # CLI
  python interactive_main.py --all-glaciers --analysis-mode comprehensive
```

### 📁 Core Engine Layer
```
albedo_engine/
├── engine.py                         🔧 UNIFIED ANALYSIS ENGINE
│   ├── AlbedoAnalysisEngine          # Main orchestration class
│   ├── Multi-Mode Analysis           • auto (smart detection)
│   │                                • basic (essential stats)
│   │                                • enhanced (advanced processing)
│   │                                • comprehensive (full suite)
│   ├── Pipeline Coordination         • Data loading management
│   │                                • Statistical analysis flow
│   │                                • Visualization generation
│   │                                • Output file organization
│   └── Result Aggregation            • Method comparison tables
│                                    • Outlier analysis summaries
│                                    • Performance metrics
└── modes/                           ⚙️ ANALYSIS MODE DEFINITIONS
    └── __init__.py                   # Mode configuration constants
```

### 📁 Data Processing Layer
```
data_processing/
├── loaders/                         📥 DATA INPUT MODULES
│   ├── pivot_loaders.py             • AthabascaMultiProductLoader
│   │                               • AthabascaAWSLoader  
│   │                               • create_pivot_based_loader()
│   ├── modis_loaders.py             • MOD09GA data handling
│   │                               • MOD10A1 data handling
│   │                               • MCD43A3 data handling
│   ├── aws_loaders.py               • Weather station data
│   │                               • Daily measurements
│   │                               • Quality filtering
│   └── csv_loaders.py               • Generic CSV parsing
│                                   • Data type inference
│                                   • Column mapping
└── processors/                      ⚙️ DATA PROCESSING MODULES
    ├── pivot_processor.py           • PivotBasedProcessor class
    │                               • Terra/Aqua data merging
    │                               • Pivot table creation
    │                               • Residual-based outlier detection
    └── data_processor.py            • DataProcessor base class
                                    • Quality filtering
                                    • Data validation
```

### 📁 Analysis Layer
```
analysis/
├── core/                           🧮 CORE ANALYSIS MODULES
│   ├── statistical_analyzer.py     • StatisticalAnalyzer class
│   │                              • RMSE, bias, correlation computation
│   │                              • Method comparison rankings
│   │                              • Performance metrics calculation
│   ├── outlier_detector.py         • OutlierDetector class
│   │                              • Residual-based detection (2.5σ)
│   │                              • Before/after comparisons
│   │                              • Improvement metrics
│   └── albedo_calculator.py        • AlbedoCalculator class
│                                  • MODIS product processing
│                                  • AWS data integration
├── comparative/                     📊 MULTI-GLACIER ANALYSIS
│   ├── interface.py                • ComparativeAnalysisInterface
│   │                              • Interactive comparative analysis
│   │                              • Cross-glacier workflows
│   ├── multi_glacier.py            • MultiGlacierComparativeAnalysis
│   │                              • Regional comparisons
│   │                              • Statistical aggregation
│   └── statistical_tests.py        • Significance testing
│                                  • ANOVA analysis
│                                  • Method consistency evaluation
└── spatial/                        🗺️ SPATIAL ANALYSIS
    ├── pixel_selection.py          • PixelSelector class
    │                              • Distance-based selection
    │                              • Glacier fraction weighting
    │                              • AWS proximity optimization
    ├── glacier_mapping_simple.py   • MultiGlacierMapperSimple (unified mapping suite)
    │                              • Basic mapping (no cartopy)
    │                              • Pixel location tracking
    └── multi_glacier_plots.py      • MultiGlacierVisualizer
                                   • Cross-glacier visualizations
                                   • Regional comparison plots
```

### 📁 Spatial Analysis Layer
```
spatial_analysis/
├── coordinates/                     🌍 COORDINATE SYSTEMS
│   └── spatial_utils.py            • SpatialProcessor class
│                                  • MODIS grid generation
│                                  • CRS transformations
│                                  • Distance calculations
├── masks/                          🎭 GLACIER BOUNDARIES
│   └── glacier_masks.py            • GlacierMaskProcessor class
│                                  • Shapefile/raster loading
│                                  • Geometry validation
│                                  • Mask operations
└── visualization/                   🗺️ SPATIAL VISUALIZATION
    └── spatial_maps.py             • SpatialMapGenerator class
                                   • Pixel location maps
                                   • AWS station overlays
                                   • Comprehensive spatial views
```

### 📁 Visualization Layer
```
visualization/
├── plots/                          📈 STATISTICAL PLOTS
│   ├── statistical_plots.py        • PlotGenerator class
│   │                              • Scatter plots (MODIS vs AWS)
│   │                              • Box plots (distribution analysis)
│   │                              • Multi-method comparisons
│   │                              • Summary figures (7-plot suite)
│   └── time_series_plots.py        • TimeSeriesPlotter class
│                                  • Temporal analysis plots
│                                  • Seasonal decomposition
│                                  • Monthly climatology
├── maps/                           🗺️ SPATIAL MAPS
│   └── map_generator.py            • MapGenerator class
│                                  • Glacier overview maps
│                                  • Albedo spatial distribution
│                                  • Method comparison maps
│                                  • Cartopy integration (optional)
└── interactive/                    🖥️ INTERACTIVE COMPONENTS
    └── dashboard_placeholder.py    • Future dashboard capabilities
                                   • Real-time visualization
                                   • Interactive parameter adjustment
```

### 📁 Utilities Layer
```
utils/
├── config/                         ⚙️ CONFIGURATION MANAGEMENT
│   └── helpers.py                  • load_config()
│                                  • setup_logging()
│                                  • ensure_directory_exists()
│                                  • validate_file_exists()
│                                  • get_timestamp()
├── data/                           ✅ DATA VALIDATION
│   └── validation.py               • validate_dataframe_structure()
│                                  • validate_albedo_values()
│                                  • validate_correlation_data()
│                                  • validate_glacier_config()
├── system/                         🔧 SYSTEM DIAGNOSTICS
│   └── diagnostics.py              • diagnose_system_environment()
│                                  • diagnose_data_availability()
│                                  • generate_diagnostic_report()
└── logging/                        📝 LOGGING ENHANCEMENTS
    └── __init__.py                 • Future custom formatters
                                   • Log analysis tools
```

### 📁 Configuration & Data
```
config/
├── config.yaml                     ⚙️ MAIN CONFIGURATION
│                                  • Analysis parameters
│                                  • Data paths
│                                  • Output settings
│                                  • Quality thresholds
└── glacier_sites.yaml             🏔️ GLACIER DEFINITIONS
                                   • Glacier metadata
                                   • AWS station coordinates
                                   • Data file mappings
                                   • Analysis mode flags

data/
├── modis/                          🛰️ SATELLITE DATA
│   ├── athabasca/                  • Multi-product CSV files
│   ├── haig/                       • Pixel-level analysis data
│   └── coropuna/                   • Time series observations
├── aws/                            🌡️ WEATHER STATION DATA
│   ├── iceAWS_Atha_*.csv          • Daily albedo measurements
│   ├── HaigAWS_*.csv              • Gap-filled time series
│   └── COROPUNA_*.csv             • Regional measurements
└── glacier_masks/                  🗺️ SPATIAL BOUNDARIES
    ├── athabasca/                  • Shapefiles and points
    ├── haig/                       • Glacier boundary files
    └── coropuna/                   • Mask rasters and vectors
```

---

## Module Interactions

```
User Input → interactive_main.py
     ↓
albedo_engine/engine.py (Analysis Coordination)
     ↓
┌────────────────────────────────────────────────────────────┐
│                    PROCESSING FLOW                         │
│                                                            │
│ data_processing/loaders → data_processing/processors       │
│           ↓                        ↓                       │
│    analysis/core → analysis/spatial → analysis/comparative │
│           ↓                        ↓                       │
│ spatial_analysis/coordinates → spatial_analysis/masks      │
│           ↓                                                │
│ visualization/plots → visualization/maps                   │
│           ↓                                                │
│        📂 outputs/ (Results, Plots, Maps)                 │
└────────────────────────────────────────────────────────────┘
```

**Key Interaction Patterns:**
- **🎯 Single Entry Point**: All user interactions flow through `interactive_main.py`
- **🔧 Central Orchestration**: `albedo_engine/engine.py` coordinates all analysis steps
- **📥 Modular Data Loading**: Specialized loaders handle different data types
- **🧮 Layered Analysis**: Core → Spatial → Comparative analysis progression
- **📊 Flexible Visualization**: Multiple output formats and styles
- **⚙️ Configuration-Driven**: YAML files control all operational parameters

---

## Entry Point System

The framework now uses a **single unified entry point** that supports both interactive and command-line modes for maximum flexibility and ease of use.

### 🎯 Single Unified Entry Point (`interactive_main.py`)
**The only entry point you need - supports all analysis modes**

```python
class InteractiveGlacierAnalysis:
    def __init__(self, config_path: str = 'config/config.yaml'):
        # Initialize unified analysis engine
        self.analysis_engine = AlbedoAnalysisEngine(config_path)
```

**Dual-Mode Operation:**

#### 🖥️ Interactive Mode (Default)
```bash
python interactive_main.py
```
- **Features:** Menu-driven interface with glacier selection
- **Data Validation:** Automatic data availability checking
- **Pixel Selection:** Choose between all pixels or optimally selected pixels
- **Analysis Modes:** Auto-determined based on glacier type (basic/comprehensive)
- **Comparative Analysis:** Access multi-glacier analysis suite
- **Real-time Progress:** Live status updates and error handling

#### ⚡ Command-Line Mode (Automated)
```bash
# Single glacier analysis
python interactive_main.py --glacier haig --analysis-mode comprehensive --selected-pixels

# Batch processing
python interactive_main.py --all-glaciers --analysis-mode basic

# Comparative analysis
python interactive_main.py --comparative-analysis

# Get help
python interactive_main.py --help
```

**Command-Line Features:**
- **Automation-Friendly:** Perfect for scripts and CI/CD pipelines
- **Flexible Analysis Modes:** auto, basic, enhanced, comprehensive
- **Pixel Selection Options:** `--selected-pixels` for optimized analysis
- **Batch Processing:** Process all available glaciers with `--all-glaciers`
- **Output Control:** `--quiet` flag for automated scripts
- **Summary Export:** `--output-summary` for CSV reports

### 🔧 Core Analysis Engine (`albedo_engine/engine.py`)
**Unified processing engine with intelligent mode detection**

```python
class AlbedoAnalysisEngine:
    def __init__(self, config_path: str):
        self.processor = PivotBasedProcessor(self.config)
        # Unified system supporting all analysis modes
    
    def process_glacier(self, glacier_id: str, use_selected_pixels: bool, analysis_mode: str):
        # analysis_mode: 'auto', 'basic', 'enhanced', 'comprehensive'
```

**Analysis Mode Intelligence:**
- **Auto Mode**: Automatically determines best approach based on glacier data type
- **Basic Mode**: Essential statistics and standard visualizations
- **Enhanced Mode**: Advanced processing with pivot-based algorithms
- **Comprehensive Mode**: Full 7-plot suite + spatial mapping + outlier analysis

**Core Capabilities:**
- **7-Plot Visualization Suite:** Complete publication-ready visualization pipeline
- **Spatial Mapping:** Glacier boundaries, MODIS pixels, AWS stations
- **Pixel Selection Algorithm:** Distance + glacier fraction weighted optimization
- **Outlier Detection:** Residual-based analysis with 2.5σ threshold
- **Statistical Analysis:** Comprehensive metrics, correlations, and rankings

### 📊 Comparative Analysis Integration
**Seamless access to multi-glacier analysis**

Access through Interactive Mode → Option [C] or command-line:
```bash
python interactive_main.py --comparative-analysis
```

**Features:**
- **Quick Comparison:** Essential statistics using all available pixels
- **Best Pixel Analysis:** Comprehensive analysis with optimally selected pixels
- **Statistical Testing:** ANOVA, regional comparisons, method consistency
- **Advanced Visualizations:** 7 plot types + pixel selection maps

---

## Usage Examples

### 📋 Interactive Mode (Recommended)
```bash
# Launch interactive interface
python interactive_main.py

# Follow the menu workflow:
# 1. Select glacier (1, 2, 3...)
# 2. Choose analysis mode:
#    - [1] Standard Analysis (all pixels)
#    - [2] Best Pixel Analysis (recommended)
# 3. View results in outputs/ directory
```

### ⚙️ Command-Line Mode
```bash
# Process specific glacier with enhanced analysis
python interactive_main.py --glacier haig --analysis-mode comprehensive --selected-pixels

# Batch process all available glaciers
python interactive_main.py --all-glaciers --analysis-mode auto --selected-pixels

# Quick basic analysis for automation
python interactive_main.py --glacier athabasca --analysis-mode basic --quiet

# Run comparative analysis
python interactive_main.py --comparative-analysis

# Export analysis summary
python interactive_main.py --all-glaciers --output-summary results_summary.csv
```

### 🎯 Analysis Mode Selection Guide
- **Auto Mode** (default): Framework intelligently selects best approach
- **Basic Mode**: Quick analysis with essential outputs
- **Enhanced Mode**: Advanced processing with pivot-based algorithms  
- **Comprehensive Mode**: Full publication-ready analysis suite

### 🔍 Pixel Selection Options
- **All Pixels** (default): Uses all available MODIS pixels
- **Selected Pixels** (`--selected-pixels`): Uses optimally selected pixels closest to AWS stations
  - Athabasca: 2 pixels (all available)
  - Haig: 2 best pixels from 13 candidates
  - Coropuna: 2 best pixels from 197 candidates

---

## Data Flow

### 🎯 Single Entry Point Processing Flow
```
User Input → interactive_main.py (Single Entry Point)
     ↓
┌─────────────────────────────────────────────────────────────┐
│                    MODE DETERMINATION                       │
│ • Interactive Menu Selection OR Command-Line Arguments      │
│ • Glacier Selection & Data Availability Validation         │
│ • Analysis Mode Selection (auto/basic/enhanced/comprehensive)│
│ • Pixel Selection Mode (all pixels vs selected pixels)     │
└─────────────────────────────────────────────────────────────┘
     ↓
albedo_engine/engine.py (Unified Analysis Engine)
     ↓
┌─────────────────────────────────────────────────────────────┐
│                 INTELLIGENT PROCESSING FLOW                 │
│                                                             │
│ Raw Data → Auto Mode Detection → Data Loading (Specialized) │
│     ↓              ↓              ↓                         │
│ Pixel Selection → Terra/Aqua → Pivot Processing            │
│ (Optional)        Merge        (Enhanced/Comprehensive)     │
│     ↓              ↓              ↓                         │
│ AWS Integration → Statistical → Outlier Detection          │
│                   Analysis     (2.5σ Residual-based)       │
│     ↓              ↓              ↓                         │
│ Visualization → Spatial Maps → Results Export              │
│ (Mode-Dependent) (Optional)    (CSV/Plots/Maps)            │
└─────────────────────────────────────────────────────────────┘
```

### 🔄 Analysis Mode Processing Paths

#### Auto Mode (Intelligent Selection)
```
Glacier Data Type Detection → 
  ├─ "athabasca_multiproduct" → Comprehensive Mode
  └─ Standard data type → Basic Mode
```

#### Basic Mode Flow
```
Data Loading → Quality Filtering → AWS Merge → Basic Statistics → 
Standard Plots → CSV Export
```

#### Enhanced/Comprehensive Mode Flow
```
Data Loading → Pixel Selection (Optional) → Terra/Aqua Merge → 
Pivot Table Creation → AWS Integration → Advanced Statistics → 
Outlier Analysis → 7-Plot Visualization Suite → Spatial Mapping → 
Comprehensive Export
```

### 📊 Comparative Analysis Flow
```
Interactive/CLI Request → ComparativeAnalysisInterface
     ↓
Multiple Glacier Data Discovery → Available Results Validation
     ↓
┌─────────────────────────────────────────────────────────────┐
│                COMPARATIVE PROCESSING PIPELINE              │
│                                                             │
│ Quick Mode:     All Pixels → Basic Aggregation →           │
│                 Essential Stats → 3 Key Plots              │
│                                                             │
│ Best Pixel Mode: Selected Pixels → Advanced Aggregation →  │
│                  Comprehensive Stats → 7 Visualization     │
│                  Types → Statistical Testing → Mapping     │
└─────────────────────────────────────────────────────────────┘
     ↓
Cross-Glacier Statistics → Regional Comparisons → 
Method Consistency Analysis → Statistical Testing (ANOVA) → 
Multi-Glacier Visualization → Comprehensive Reporting
```

### 📈 Visualization Generation Flow
```
Analysis Results → PlotGenerator (statistical_plots.py) → 
  ├─ 7-Plot Suite (Comprehensive Mode)
  │  ├─ User-style comprehensive analysis
  │  ├─ Multi-panel summary figure  
  │  ├─ Time series analysis
  │  ├─ Distribution analysis
  │  ├─ Outlier analysis (before/after)
  │  ├─ Seasonal analysis
  │  └─ Correlation & bias analysis
  └─ Basic Plots (Basic Mode)
     ├─ Scatter plots (MODIS vs AWS)
     ├─ Time series plots
     └─ Summary statistics tables

Spatial Data → MapGenerator (map_generator.py) → 
  ├─ Glacier overview maps
  ├─ Pixel location maps
  ├─ AWS station overlays
  └─ Method comparison maps
```

### 💾 Data Export Pipeline
```
Analysis Results → 
  ├─ CSV Export (pandas DataFrames)
  │  ├─ Statistical summaries
  │  ├─ Method comparisons
  │  ├─ Correlation matrices
  │  └─ Raw merged data
  ├─ Plot Export (matplotlib figures)
  │  ├─ High-resolution PNG (300 DPI)
  │  ├─ Publication-ready formats
  │  └─ Interactive plots (optional)
  └─ Report Generation
     ├─ Comprehensive text reports
     ├─ Analysis summaries
     └─ Configuration logs
```

---

## Configuration System

### Main Configuration (`config/config.yaml`)
```yaml
# Analysis parameters
analysis:
  albedo:
    modis_products: [MOD09GA, MOD10A1, MCD43A3]
    quality_filters:
      cloud_threshold: 0.2
      outlier_threshold: 2.5
  statistics:
    metrics: [rmse, bias, correlation, mae]
    confidence_level: 0.95

# Data paths
data:
  modis_path: "data/modis"
  aws_path: "data/aws"
  glacier_masks_path: "data/glacier_masks"

# Output configuration
output:
  base_path: "outputs"
  plots_path: "outputs/plots"
  results_path: "outputs/results"
```

### Glacier Sites Configuration (`config/glacier_sites.yaml`)
```yaml
glaciers:
  athabasca:
    name: "Athabasca Glacier"
    region: "Canadian Rockies"
    coordinates: {lat: 52.2, lon: -117.25}
    data_files:
      modis:
        MOD09GA: "Athabasca_Terra_Aqua_MultiProduct_2014-01-01_to_2021-01-01.csv"
      aws: "iceAWS_Atha_albedo_daily_20152020_filled_clean.csv"
      mask: "data/glacier_masks/athabasca/masque_athabasa_zone_ablation.shp"
    aws_stations:
      iceAWS_Atha:
        name: "Athabasca Glacier AWS"
        lat: 50.7124
        lon: -115.3018
        elevation: 2200
    data_type: "athabasca_multiproduct"  # Enables enhanced plotting
```

**Configuration Features:**
- **Flexible Paths:** Supports glacier-specific data directories
- **Data Type Flags:** Controls which analysis pipeline to use
- **AWS Station Metadata:** Enables pixel selection algorithms
- **Quality Thresholds:** Customizable filtering parameters

---

## Analysis Pipelines

### Pixel Selection Algorithm
The framework implements an intelligent pixel selection system for enhanced accuracy:

```python
def _apply_pixel_selection(self, modis_data, glacier_id, glacier_config):
    """
    Select 2 closest best-performing pixels to AWS station
    Criteria:
    - Distance to AWS station (60% weight)
    - Glacier fraction coverage (40% weight)
    - Minimum data quality thresholds
    """
```

**Selection Criteria:**
1. **Quality Filters:**
   - Glacier fraction > 0.1
   - Observation count > 10
2. **Performance Ranking:**
   - Sort by glacier fraction (descending)
   - Sort by distance to AWS (ascending)
3. **Final Selection:** Top 2 pixels meeting all criteria

### Statistical Analysis Suite

**Basic Metrics (All Pipelines):**
- Correlation coefficient (r)
- Root Mean Square Error (RMSE)
- Mean Absolute Error (MAE)
- Bias (mean difference)

**Enhanced Metrics (Pivot-Based Pipeline):**
- R-squared
- Relative bias (%)
- Mean absolute relative error
- Within-threshold percentages (5%, 10%, 15%)
- Regression statistics (slope, intercept, standard error)

**Outlier Detection:**
- Residual-based method (2.5σ threshold)
- Before/after comparison
- Improvement metrics (correlation, RMSE)

---

## Visualization Suite

### Standard Pipeline Outputs
- Basic scatter plots (MODIS vs AWS)
- Simple statistics tables
- Time series plots

### Enhanced Pipeline Outputs (7-Plot Suite)

#### 1. User-Style Comprehensive Analysis
- Seasonal scatter plots with color coding
- Regression lines and 1:1 reference
- Statistical annotation boxes
- Method-specific panels

#### 2. Multi-Panel Summary Figure
- Individual method scatter plots
- Performance comparison table
- Distribution comparisons (box plots)
- Bias analysis

#### 3. Time Series Analysis
- Separate panels for each method
- Daily observation scatter plots
- Temporal pattern visualization
- Data availability display

#### 4. Distribution Analysis
- Box plots for method comparison
- Violin plots showing density
- Histograms with overlays
- Cumulative distribution functions

#### 5. Outlier Analysis (Before/After)
- Side-by-side comparison plots
- Outlier highlighting (red X markers)
- Statistical improvement metrics
- Residual-based detection visualization

#### 6. Seasonal Analysis
- Monthly distribution box plots
- Seasonal trend analysis
- Method performance by season
- Sample size information

#### 7. Correlation & Bias Analysis
- Correlation matrix heatmap
- Method performance comparison bars
- Bias vs AWS albedo scatter
- RMSE vs sample size analysis

#### 8. Spatial Mapping
- Glacier boundary visualization
- MODIS pixel locations
- AWS station marking
- Pixel selection highlighting
- Distance information display

### Comparative Analysis Outputs
- Multi-glacier performance matrices
- Regional comparison plots
- Statistical significance testing
- Cross-glacier correlation analysis
- Method consistency evaluation

---

## Usage Examples

### 🚀 Quick Start - Interactive Mode (Recommended)
```bash
# Launch the unified interactive interface
python interactive_main.py

# Interactive Menu Workflow:
# 1. Select glacier from available list (1, 2, 3...)
#    • Shows data availability status for each glacier
#    • Displays enhanced plotting capabilities where available
# 2. Choose analysis type:
#    - [1] Standard Analysis (all available pixels) 
#    - [2] Best Pixel Analysis (optimally selected pixels - RECOMMENDED)
# 3. View comprehensive results in timestamped outputs/ directory
# 4. Optional: Access comparative analysis with [C]
```

### ⚡ Command-Line Mode - Single Glacier
```bash
# Comprehensive analysis with selected pixels (recommended)
python interactive_main.py --glacier haig --analysis-mode comprehensive --selected-pixels

# Quick basic analysis for automation
python interactive_main.py --glacier athabasca --analysis-mode basic --quiet

# Auto-mode (framework decides best approach)
python interactive_main.py --glacier coropuna --analysis-mode auto --selected-pixels

# Export results summary
python interactive_main.py --glacier haig --selected-pixels --output-summary haig_results.csv
```

### 🔄 Batch Processing - All Glaciers
```bash
# Process all available glaciers with selected pixels
python interactive_main.py --all-glaciers --analysis-mode auto --selected-pixels

# Quick batch processing for all glaciers
python interactive_main.py --all-glaciers --analysis-mode basic

# Comprehensive batch analysis with summary export
python interactive_main.py --all-glaciers --analysis-mode comprehensive --selected-pixels --output-summary batch_results.csv
```

### 📊 Comparative Analysis
```bash
# Interactive comparative analysis
python interactive_main.py
# Then select [C] Comparative Analysis from menu

# Direct command-line comparative analysis
python interactive_main.py --comparative-analysis

# Automated comparative analysis (if available)
python interactive_main.py --comparative-analysis --quiet
```

### 🛠️ Advanced Usage Examples
```bash
# Help and documentation
python interactive_main.py --help

# Silent processing for scripts
python interactive_main.py --glacier athabasca --quiet --output-summary results.csv

# Custom configuration file
python interactive_main.py --config /path/to/custom/config.yaml --glacier haig

# Debug mode with verbose output
python interactive_main.py --glacier haig --selected-pixels
# (Default shows progress and detailed output)
```

### 📋 Analysis Mode Selection Guide
```bash
# Auto Mode (Recommended) - Framework intelligently selects approach
python interactive_main.py --glacier haig --analysis-mode auto

# Basic Mode - Quick essential analysis
python interactive_main.py --glacier haig --analysis-mode basic

# Enhanced Mode - Advanced processing algorithms
python interactive_main.py --glacier haig --analysis-mode enhanced --selected-pixels

# Comprehensive Mode - Full publication-ready analysis
python interactive_main.py --glacier haig --analysis-mode comprehensive --selected-pixels
```

### ⚙️ Configuration Customization
```yaml
# config/config.yaml - Custom quality thresholds
analysis:
  albedo:
    quality_filters:
      cloud_threshold: 0.1      # Stricter cloud filtering
      outlier_threshold: 3.0    # Looser outlier detection (default: 2.5)
    statistics:
      confidence_level: 0.99    # Higher confidence level

# Custom output paths
output:
  base_path: "/path/to/custom/outputs"
  plots_dpi: 400               # Higher resolution plots
  
# Glacier-specific data paths
data:
  athabasca_modis_path: "/custom/athabasca/modis"
  haig_aws_path: "/custom/haig/aws"
  coropuna_modis_path: "/custom/coropuna/data"
```

### 🔍 Pixel Selection Examples
```bash
# Use all available pixels (default)
python interactive_main.py --glacier coropuna --analysis-mode comprehensive

# Use optimally selected pixels (recommended for accuracy)
python interactive_main.py --glacier coropuna --analysis-mode comprehensive --selected-pixels

# Compare both approaches
python interactive_main.py --glacier haig --analysis-mode comprehensive
# Then run:
python interactive_main.py --glacier haig --analysis-mode comprehensive --selected-pixels
```

### 📈 Output Structure Examples
```bash
# After running analysis, expect this structure:
outputs/
├── glacier_analysis_20240130_143022/          # Timestamped directory
│   ├── plots/                                 # All visualization outputs
│   │   ├── haig_comprehensive_analysis.png
│   │   ├── haig_multi_panel_summary.png
│   │   ├── haig_time_series_analysis.png
│   │   ├── haig_distribution_analysis.png
│   │   ├── haig_outlier_analysis.png
│   │   ├── haig_seasonal_analysis.png
│   │   └── haig_correlation_bias_analysis.png
│   ├── maps/                                  # Spatial visualizations
│   │   ├── haig_glacier_overview_map.png
│   │   └── haig_pixel_locations_map.png
│   └── results/                               # Data exports
│       ├── haig_statistical_summary.csv
│       ├── haig_method_comparison.csv
│       ├── haig_correlation_matrix.csv
│       └── haig_merged_data.csv
```

---

## Developer Guide

### 🏔️ Adding New Glaciers

1. **Add Configuration Entry:**
```yaml
# In config/glacier_sites.yaml
glaciers:
  new_glacier:
    name: "New Glacier"
    region: "Mountain Range"
    coordinates: {lat: XX.XXXX, lon: -XXX.XXXX}
    data_files:
      modis:
        MOD09GA: "NewGlacier_MultiProduct.csv"
      aws: "NewGlacier_AWS_data.csv"
      mask: "data/glacier_masks/new_glacier/boundary.shp"
    aws_stations:
      station_1:
        name: "New Glacier AWS"
        lat: XX.XXXX
        lon: -XXX.XXXX
        elevation: XXXX
    data_type: "athabasca_multiproduct"  # For enhanced analysis (7-plot suite)
```

2. **Prepare Data Files:**
- Place MODIS data in `data/modis/new_glacier/`
- Place AWS data in `data/aws/`
- Place spatial files in `data/glacier_masks/new_glacier/`

3. **Test Integration:**
```bash
# Test in interactive mode
python interactive_main.py
# New glacier should appear in menu with data availability status

# Test in command-line mode
python interactive_main.py --glacier new_glacier --analysis-mode auto
```

### 🔧 Extending Analysis Methods

1. **Create Custom Analysis Module:**
```python
# In analysis/core/custom_analyzer.py
from typing import Dict, Any
import pandas as pd

class CustomAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def analyze_custom_metric(self, merged_data: pd.DataFrame) -> Dict[str, float]:
        """Implement custom analysis logic."""
        # Your custom analysis implementation
        return {"custom_metric": value}
```

2. **Add Custom Visualization:**
```python
# In visualization/plots/custom_plots.py
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

class CustomPlotGenerator:
    def create_custom_plot(self, data: pd.DataFrame, 
                          output_path: Optional[str] = None) -> plt.Figure:
        """Create custom visualization."""
        fig, ax = plt.subplots(figsize=(10, 6))
        # Your plotting logic here
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        return fig
```

3. **Integrate with Analysis Engine:**
```python
# In albedo_engine/engine.py - add to process_glacier method
from analysis.core.custom_analyzer import CustomAnalyzer

# Within process_glacier method:
if analysis_mode in ['enhanced', 'comprehensive']:
    custom_analyzer = CustomAnalyzer(self.config)
    custom_results = custom_analyzer.analyze_custom_metric(merged_data)
    statistics['custom'] = custom_results
```

### 🎯 Modifying the Single Entry Point

1. **Add New Command-Line Options:**
```python
# In interactive_main.py - main() function
parser.add_argument('--custom-analysis', action='store_true',
                   help='Run custom analysis module')
parser.add_argument('--export-format', type=str, choices=['csv', 'json', 'xlsx'],
                   default='csv', help='Output format for results')
```

2. **Add New Interactive Menu Options:**
```python
# In interactive_main.py - display_options_menu method
print("| [X] Custom Analysis Module                                            |")
print("|     >> Specialized analysis for research workflows                  |")
```

### 🛠️ Creating Analysis Plugins

1. **Plugin Structure:**
```python
# In plugins/research_plugin.py
class ResearchAnalysisPlugin:
    """Plugin for specialized research analysis."""
    
    def __init__(self, config_path: str):
        self.config = load_config(config_path)
    
    def run_specialized_analysis(self, glacier_id: str) -> Dict[str, Any]:
        """Run specialized analysis workflow."""
        # Plugin implementation
        return results
    
    def integrate_with_main(self, main_app):
        """Integration method for main application."""
        # Add plugin to main application
        pass
```

2. **Plugin Registration:**
```python
# In interactive_main.py - __init__ method
def load_plugins(self):
    """Load available analysis plugins."""
    plugins_dir = Path('plugins')
    if plugins_dir.exists():
        for plugin_file in plugins_dir.glob('*_plugin.py'):
            # Dynamic plugin loading logic
            pass
```

### 🔍 Troubleshooting Common Issues

#### Data Loading Problems
```python
# Interactive diagnostic check
python interactive_main.py
# Select [S] Show System Status for comprehensive diagnostics

# Programmatic check
python -c "
from interactive_main import InteractiveGlacierAnalysis
app = InteractiveGlacierAnalysis()
available, missing = app.check_data_availability('glacier_name')
print(f'Data Available: {available}')
if missing:
    print('Missing Files:')
    for file in missing:
        print(f'  - {file}')
"
```

#### Configuration Issues
```python
# Validate all configurations
from utils.config.helpers import load_config
from utils.data.validation import validate_glacier_config

config = load_config('config/config.yaml')
glacier_config = load_config('config/glacier_sites.yaml')

# Validate each glacier configuration
for glacier_id, glacier_info in glacier_config['glaciers'].items():
    is_valid, errors = validate_glacier_config(glacier_info)
    print(f"{glacier_id}: {'Valid' if is_valid else 'Invalid'}")
    if errors:
        for error in errors:
            print(f"  - {error}")
```

#### Analysis Engine Issues
```python
# Debug analysis engine
from albedo_engine.engine import AlbedoAnalysisEngine
import logging

logging.basicConfig(level=logging.DEBUG)
engine = AlbedoAnalysisEngine('config/config.yaml')

# Test engine initialization
print(f"Engine config loaded: {engine.config is not None}")
print(f"Processor initialized: {engine.processor is not None}")
```

#### Output Directory Problems
```bash
# Check output structure and permissions
ls -la outputs/
# Should show timestamped directories with results/, plots/, maps/

# Check disk space
df -h outputs/

# Check permissions
ls -la outputs/ | head -5
```

### ⚡ Performance Optimization

1. **Memory Management:**
```python
# Use pixel selection for large datasets
python interactive_main.py --glacier coropuna --selected-pixels  # 2 pixels vs 197

# Process glaciers individually for memory efficiency
python interactive_main.py --glacier athabasca --analysis-mode basic
python interactive_main.py --glacier haig --analysis-mode basic

# Monitor memory usage
import psutil
process = psutil.Process()
print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
```

2. **Processing Speed:**
```yaml
# config/config.yaml - Enable performance optimizations
analysis:
  performance:
    parallel_processing: true
    cache_intermediate_results: true
    optimize_memory_usage: true
  albedo:
    quality_filters:
      fast_mode: true  # Skip expensive quality checks
```

3. **Storage Optimization:**
```yaml
# config/config.yaml - Optimize storage
output:
  compression: true
  plots_format: "png"  # Smaller than PDF
  plots_dpi: 150      # Lower DPI for faster generation
  export_raw_data: false  # Skip large CSV exports
```

### 🧪 Testing New Features

1. **Unit Testing:**
```python
# tests/test_analysis_engine.py
import unittest
from albedo_engine.engine import AlbedoAnalysisEngine

class TestAnalysisEngine(unittest.TestCase):
    def setUp(self):
        self.engine = AlbedoAnalysisEngine('config/config.yaml')
    
    def test_glacier_processing(self):
        result = self.engine.process_glacier('athabasca', False, 'basic')
        self.assertIsNotNone(result)
        self.assertIn('statistics', result)
```

2. **Integration Testing:**
```bash
# Test full pipeline
python interactive_main.py --glacier athabasca --analysis-mode basic --quiet
echo "Exit code: $?"

# Test batch processing
python interactive_main.py --all-glaciers --analysis-mode basic --quiet
echo "Exit code: $?"
```

3. **Performance Testing:**
```bash
# Time analysis execution
time python interactive_main.py --glacier haig --analysis-mode comprehensive --selected-pixels

# Memory profiling
python -m memory_profiler interactive_main.py --glacier coropuna --analysis-mode basic
```

---

## File Structure Summary

```
Albedo_analysis_New/
├── interactive_main.py                    # 🎯 SINGLE UNIFIED ENTRY POINT
│                                         #    (Interactive + Command-Line modes)
├── albedo_engine/                        # 🔧 CORE ANALYSIS ENGINE
│   └── engine.py                        #    Unified processing engine with all modes
├── data_processing/                      # 📥 DATA PROCESSING LAYER
│   ├── loaders/                         #    Specialized data input modules
│   │   ├── pivot_loaders.py            #    Pivot-based data loading
│   │   ├── modis_loaders.py            #    MODIS satellite data handling
│   │   ├── aws_loaders.py              #    Weather station data
│   │   └── csv_loaders.py              #    Generic CSV processing
│   └── processors/                      #    Data processing modules
│       ├── pivot_processor.py          #    Terra/Aqua merging & outlier detection
│       └── data_processor.py           #    Quality filtering & validation
├── analysis/                            # 🧮 ANALYSIS LAYER
│   ├── core/                           #    Core analysis modules
│   │   ├── statistical_analyzer.py    #    RMSE, bias, correlation computation
│   │   ├── outlier_detector.py        #    Residual-based outlier detection (2.5σ)
│   │   └── albedo_calculator.py        #    MODIS product processing
│   ├── comparative/                     #    Multi-glacier analysis
│   │   ├── interface.py               #    Comparative analysis interface
│   │   ├── multi_glacier.py           #    Regional comparisons
│   │   └── statistical_tests.py       #    Significance testing & ANOVA
│   └── spatial/                        #    Spatial analysis
│       ├── pixel_selection.py         #    Distance + glacier fraction weighting
│       ├── glacier_mapping_simple.py  #    Unified mapping suite (no cartopy required)
│       └── multi_glacier_plots.py     #    Cross-glacier visualizations
├── spatial_analysis/                    # 🗺️ SPATIAL ANALYSIS LAYER
│   ├── coordinates/                     #    Coordinate systems
│   │   └── spatial_utils.py           #    MODIS grid generation & CRS transforms
│   ├── masks/                          #    Glacier boundaries
│   │   └── glacier_masks.py           #    Shapefile/raster loading & operations
│   └── visualization/                   #    Spatial visualization
│       └── spatial_maps.py            #    Pixel location maps & AWS overlays
├── visualization/                       # 📊 VISUALIZATION LAYER
│   ├── plots/                          #    Statistical plots
│   │   ├── statistical_plots.py       #    7-plot suite & method comparisons
│   │   └── time_series_plots.py       #    Temporal analysis plots
│   ├── maps/                           #    Spatial maps
│   │   └── map_generator.py           #    Cartopy integration (optional)
│   └── interactive/                     #    Interactive components
│       └── dashboard_placeholder.py    #    Future dashboard capabilities
├── utils/                              # ⚙️ UTILITIES LAYER
│   ├── config/                         #    Configuration management
│   │   └── helpers.py                 #    Config loading & logging setup
│   ├── data/                           #    Data validation
│   │   └── validation.py              #    DataFrame structure & value validation
│   ├── system/                         #    System diagnostics
│   │   └── diagnostics.py             #    Health monitoring & dependency checks
│   └── logging/                        #    Logging enhancements
│       └── __init__.py                #    Future custom logging features
├── config/                             # ⚙️ CONFIGURATION FILES
│   ├── config.yaml                    #    Main configuration & analysis parameters
│   └── glacier_sites.yaml            #    Glacier metadata & AWS station info
├── data/                              # 📂 DATA STORAGE
│   ├── modis/                         #    MODIS satellite data by glacier
│   │   ├── athabasca/                 #    Multi-product CSV files
│   │   ├── haig/                      #    Pixel-level analysis data
│   │   └── coropuna/                  #    Time series observations
│   ├── aws/                           #    Weather station data
│   │   ├── iceAWS_Atha_*.csv         #    Daily albedo measurements
│   │   ├── HaigAWS_*.csv             #    Gap-filled time series
│   │   └── COROPUNA_*.csv            #    Regional measurements
│   └── glacier_masks/                 #    Spatial boundaries
│       ├── athabasca/                 #    Shapefiles and points
│       ├── haig/                      #    Glacier boundary files
│       └── coropuna/                  #    Mask rasters and vectors
├── outputs/                           # 📂 ANALYSIS RESULTS
│   ├── glacier_analysis_YYYYMMDD_HHMMSS/  # Timestamped analysis directories
│   │   ├── plots/                     #    Generated visualizations (7 plot types)
│   │   ├── maps/                      #    Spatial visualizations
│   │   └── results/                   #    Statistical outputs & CSV exports
│   └── comparative_analysis_YYYYMMDD_HHMMSS/  # Comparative analysis results
└── glacier_interactive_dashboard/      # 🖥️ OPTIONAL INTERACTIVE FEATURES
    └── [Future dashboard components]
```

## 🎯 Key Architecture Features

### **Single Entry Point System:**
- **`interactive_main.py`** - Only file you need to run
- **Dual Mode:** Interactive menu + Command-line automation
- **Intelligent Mode Selection:** Auto-determines best analysis approach
- **Unified Interface:** All features accessible from one place

### **Professional Module Organization:**
- **Clean Separation:** Each layer has specific responsibilities
- **Modular Design:** Easy to extend and maintain
- **Optional Dependencies:** Graceful degradation when packages missing
- **Proper Imports:** Well-structured `__init__.py` files throughout

### **Advanced Analysis Capabilities:**
- **7-Plot Visualization Suite:** Publication-ready comprehensive analysis
- **Pixel Selection Algorithm:** Distance + glacier fraction optimization
- **Outlier Detection:** Residual-based analysis with 2.5σ threshold
- **Multi-Glacier Comparisons:** Statistical testing and regional analysis
- **Spatial Mapping:** Glacier boundaries, MODIS pixels, AWS stations

This framework provides a **comprehensive, user-friendly system** for analyzing glacier albedo with multiple analysis modes, advanced visualization capabilities, robust statistical analysis tools, and a professional software architecture suitable for research and operational use.