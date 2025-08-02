#!/usr/bin/env python3
"""
Configuration Examples and Patterns

This file provides various configuration examples and patterns for different 
types of glacier albedo analyses. Use these as references when setting up 
new analysis scripts.

Author: Analysis System
Date: 2025-08-02
"""

# ============================================================================
# STANDARD BASE CONFIGURATION
# ============================================================================

# This is the standard configuration that should be included in every analysis
STANDARD_BASE_CONFIG = {
    'data_paths': {
        'athabasca': {
            'modis': "D:/Documents/Projects/athabasca_analysis/data/csv/Athabasca_Terra_Aqua_MultiProduct_2014-01-01_to_2021-01-01.csv",
            'aws': "D:/Documents/Projects/athabasca_analysis/data/csv/iceAWS_Atha_albedo_daily_20152020_filled_clean.csv"
        },
        'haig': {
            'modis': "D:/Documents/Projects/Haig_analysis/data/csv/Haig_MODIS_Pixel_Analysis_MultiProduct_2002_to_2016_fraction.csv",
            'aws': "D:/Documents/Projects/Haig_analysis/data/csv/HaigAWS_daily_2002_2015_gapfilled.csv"
        },
        'coropuna': {
            'modis': "D:/Documents/Projects/Coropuna_glacier/data/csv/coropuna_glacier_2014-01-01_to_2025-01-01.csv",
            'aws': "D:/Documents/Projects/Coropuna_glacier/data/csv/COROPUNA_simple.csv"
        }
    },
    'aws_stations': {
        'athabasca': {'lat': 52.1949, 'lon': -117.2431, 'name': 'Athabasca AWS'},
        'haig': {'lat': 50.7186, 'lon': -115.3433, 'name': 'Haig AWS'},
        'coropuna': {'lat': -15.5181, 'lon': -72.6617, 'name': 'Coropuna AWS'}
    },
    'methods': ['MCD43A3', 'MOD09GA', 'MOD10A1'],
    'method_mapping': {
        'mcd43a3': 'MCD43A3', 'MCD43A3': 'MCD43A3',
        'mod09ga': 'MOD09GA', 'MOD09GA': 'MOD09GA',
        'myd09ga': 'MOD09GA', 'MYD09GA': 'MOD09GA',  # Group Aqua with Terra
        'mod10a1': 'MOD10A1', 'MOD10A1': 'MOD10A1',
        'myd10a1': 'MOD10A1', 'MYD10A1': 'MOD10A1'   # Group Aqua with Terra
    },
    'outlier_threshold': 2.5,
    'quality_filters': {
        'min_glacier_fraction': 0.1,
        'min_observations': 10
    }
}

# ============================================================================
# COLOR SCHEME EXAMPLES
# ============================================================================

# Standard glacier color scheme (recommended for consistency)
GLACIER_COLORS = {
    'athabasca': '#1f77b4',    # Blue
    'haig': '#ff7f0e',         # Orange  
    'coropuna': '#2ca02c'      # Green
}

# Method-specific colors (for MODIS product comparisons)
METHOD_COLORS = {
    'MOD09GA': '#9467bd',      # Purple (Terra)
    'MYD09GA': '#17becf',      # Cyan (Aqua)
    'MCD43A3': '#d62728',      # Red
    'MOD10A1': '#8c564b',      # Brown (Terra)
    'MYD10A1': '#e377c2',      # Pink (Aqua)
    'AWS': '#000000'           # Black
}

# Alternative color schemes for different analysis types
QUALITATIVE_COLORS = {
    'scheme_1': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
    'scheme_2': ['#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'colorbrind_safe': ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161']
}

# Seasonal analysis colors (for monthly/seasonal comparisons)
SEASONAL_COLORS = {
    'spring': '#90EE90',   # Light green
    'summer': '#FFD700',   # Gold
    'autumn': '#FF6347',   # Tomato
    'winter': '#87CEEB'    # Sky blue
}

# ============================================================================
# VISUALIZATION CONFIGURATION EXAMPLES
# ============================================================================

# Standard visualization settings
STANDARD_VIZ_CONFIG = {
    'figsize': (12, 10),
    'dpi': 300,
    'style': 'seaborn-v0_8'
}

# Different figure size options for various analysis types
FIGURE_SIZES = {
    'single_plot': (8, 6),
    'comparison_2x2': (12, 10),
    'matrix_3x3': (15, 12),
    'timeline': (16, 8),
    'bar_chart_3x4': (16, 12),
    'scatterplot_matrix': (12, 12),
    'large_comparison': (20, 15)
}

# Style options
PLOT_STYLES = {
    'publication': 'seaborn-v0_8',
    'presentation': 'seaborn-v0_8-bright',
    'print': 'seaborn-v0_8-paper',
    'poster': 'seaborn-v0_8-poster'
}

# ============================================================================
# ANALYSIS-SPECIFIC CONFIGURATION EXAMPLES
# ============================================================================

# Scatterplot analysis configuration
SCATTERPLOT_CONFIG = {
    **STANDARD_BASE_CONFIG,
    'colors': {**GLACIER_COLORS},
    'visualization': {
        'figsize': (12, 12),
        'dpi': 300,
        'style': 'seaborn-v0_8'
    },
    'output': {
        'analysis_name': 'aws_vs_modis_scatterplot',
        'base_dir': 'outputs',
        'plot_filename': 'aws_vs_modis_scatterplot_matrix.png',
        'summary_template': {
            'analysis_type': 'AWS vs MODIS Albedo Correlation Analysis',
            'description': 'Scatterplot matrix comparing AWS and MODIS albedo measurements across three glaciers'
        }
    }
}

# Bar chart comparison configuration
BAR_CHART_CONFIG = {
    **STANDARD_BASE_CONFIG,
    'colors': {**GLACIER_COLORS, **METHOD_COLORS},
    'visualization': {
        'figsize': (16, 12),
        'dpi': 300,
        'style': 'seaborn-v0_8'
    },
    'output': {
        'analysis_name': 'method_performance',
        'base_dir': 'outputs',
        'plot_filename': 'method_performance_bar_chart.png',
        'summary_template': {
            'analysis_type': 'MODIS Method Performance Comparison',
            'description': 'Bar chart comparison of correlation, RMSE, bias, and MAE metrics across glaciers and MODIS methods'
        }
    }
}

# Seasonal analysis configuration
SEASONAL_CONFIG = {
    **STANDARD_BASE_CONFIG,
    'colors': {**GLACIER_COLORS, **METHOD_COLORS, 'AWS': '#d62728'},
    'methods': ['AWS', 'MCD43A3', 'MOD09GA', 'MOD10A1'],  # Include AWS for seasonal comparison
    'seasonal_months': [6, 7, 8, 9],  # June through September
    'month_names': {6: 'June', 7: 'July', 8: 'August', 9: 'September'},
    'visualization': {
        'figsize': (20, 15),
        'dpi': 300,
        'style': 'seaborn-v0_8'
    },
    'output': {
        'analysis_name': 'seasonal_analysis',
        'base_dir': 'outputs',
        'plot_filename': 'seasonal_boxplots.png',
        'summary_template': {
            'analysis_type': 'Multi-Glacier Seasonal Analysis',
            'description': 'Seasonal boxplot analysis of AWS vs MODIS method comparisons across summer months'
        }
    }
}

# Residual analysis configuration
RESIDUAL_CONFIG = {
    **STANDARD_BASE_CONFIG,
    'colors': {**METHOD_COLORS},
    'visualization': {
        'figsize': (18, 12),
        'dpi': 300,
        'style': 'seaborn-v0_8'
    },
    'output': {
        'analysis_name': 'residual_analysis',
        'base_dir': 'outputs',
        'plot_filename': 'residual_analysis_grid.png',
        'summary_template': {
            'analysis_type': 'Multi-Glacier Residual Analysis',
            'description': 'Residual analysis showing AWS vs MODIS method error patterns across glaciers'
        }
    }
}

# Time series analysis configuration
TIMESERIES_CONFIG = {
    **STANDARD_BASE_CONFIG,
    'colors': {**GLACIER_COLORS, **METHOD_COLORS},
    'time_windows': {
        'daily': 1,
        'weekly': 7,
        'monthly': 30,
        'seasonal': 90
    },
    'visualization': {
        'figsize': (16, 10),
        'dpi': 300,
        'style': 'seaborn-v0_8'
    },
    'output': {
        'analysis_name': 'timeseries_analysis',
        'base_dir': 'outputs',
        'plot_filename': 'albedo_timeseries.png',
        'summary_template': {
            'analysis_type': 'Albedo Time Series Analysis',
            'description': 'Time series comparison of AWS and MODIS albedo measurements'
        }
    }
}

# ============================================================================
# QUALITY FILTER VARIATIONS
# ============================================================================

# Conservative quality filters (higher standards)
CONSERVATIVE_FILTERS = {
    'min_glacier_fraction': 0.2,   # 20% glacier coverage minimum
    'min_observations': 20,        # At least 20 observations
    'max_distance_km': 5.0,        # Within 5km of AWS station
    'min_correlation': 0.3         # Minimum correlation threshold
}

# Liberal quality filters (lower standards, more data retention)
LIBERAL_FILTERS = {
    'min_glacier_fraction': 0.05,  # 5% glacier coverage minimum
    'min_observations': 5,         # At least 5 observations
    'max_distance_km': 15.0,       # Within 15km of AWS station
    'min_correlation': 0.1         # Lower correlation threshold
}

# Standard quality filters (recommended default)
STANDARD_FILTERS = {
    'min_glacier_fraction': 0.1,   # 10% glacier coverage minimum
    'min_observations': 10,        # At least 10 observations
    'max_distance_km': 10.0,       # Within 10km of AWS station
    'min_correlation': 0.2         # Minimum correlation threshold
}

# ============================================================================
# OUTPUT CONFIGURATION PATTERNS
# ============================================================================

# Standard output template
def get_output_config(analysis_name: str, plot_filename: str, 
                     analysis_type: str, description: str) -> dict:
    """Generate standard output configuration.
    
    Args:
        analysis_name: Lowercase name for the analysis
        plot_filename: Name of the main plot file
        analysis_type: Descriptive title for the analysis
        description: Detailed description of the analysis
        
    Returns:
        Dictionary with output configuration
    """
    return {
        'analysis_name': analysis_name,
        'base_dir': 'outputs',
        'plot_filename': plot_filename,
        'summary_template': {
            'analysis_type': analysis_type,
            'description': description
        }
    }

# Common output configurations
OUTPUT_EXAMPLES = {
    'correlation': get_output_config(
        'correlation_analysis',
        'correlation_matrix.png',
        'Correlation Analysis',
        'Statistical correlation analysis between AWS and MODIS albedo measurements'
    ),
    'performance': get_output_config(
        'method_performance',
        'performance_comparison.png',
        'Method Performance Comparison',
        'Comparative analysis of MODIS method performance against AWS reference data'
    ),
    'temporal': get_output_config(
        'temporal_analysis',
        'temporal_patterns.png',
        'Temporal Pattern Analysis',
        'Analysis of temporal patterns and trends in albedo measurements'
    ),
    'spatial': get_output_config(
        'spatial_analysis',
        'spatial_distribution.png',
        'Spatial Distribution Analysis',
        'Analysis of spatial patterns in albedo measurements across glacier surfaces'
    )
}

# ============================================================================
# SPECIALIZED ANALYSIS CONFIGURATIONS
# ============================================================================

# Statistical significance testing configuration
STATISTICAL_CONFIG = {
    **STANDARD_BASE_CONFIG,
    'statistical_tests': {
        'significance_level': 0.05,
        'correction_method': 'bonferroni',  # Multiple comparison correction
        'bootstrap_samples': 1000,
        'confidence_interval': 0.95
    },
    'output': get_output_config(
        'statistical_analysis',
        'statistical_results.png',
        'Statistical Significance Analysis',
        'Statistical significance testing of AWS-MODIS albedo correlations'
    )
}

# Error analysis configuration
ERROR_ANALYSIS_CONFIG = {
    **STANDARD_BASE_CONFIG,
    'error_metrics': ['rmse', 'mae', 'bias', 'mape', 'r2'],
    'error_thresholds': {
        'excellent': {'rmse': 0.05, 'mae': 0.03, 'bias': 0.02},
        'good': {'rmse': 0.10, 'mae': 0.06, 'bias': 0.05},
        'acceptable': {'rmse': 0.15, 'mae': 0.10, 'bias': 0.08}
    },
    'output': get_output_config(
        'error_analysis',
        'error_metrics.png',
        'Error Analysis',
        'Comprehensive error analysis of MODIS albedo retrievals'
    )
}

# Environmental conditions configuration
ENVIRONMENTAL_CONFIG = {
    **STANDARD_BASE_CONFIG,
    'environmental_factors': {
        'cloud_cover_threshold': 0.2,
        'solar_zenith_max': 70,
        'temperature_range': [-30, 20],
        'wind_speed_max': 15
    },
    'stratification_variables': ['cloud_cover', 'solar_zenith', 'temperature'],
    'output': get_output_config(
        'environmental_analysis',
        'environmental_impacts.png',
        'Environmental Impact Analysis',
        'Analysis of environmental factors affecting albedo measurement accuracy'
    )
}

# ============================================================================
# USAGE EXAMPLES
# ============================================================================

# Example: Creating a custom configuration by merging base with specific settings
def create_custom_config(analysis_type: str, custom_settings: dict) -> dict:
    """Create a custom configuration by merging base settings with specific requirements.
    
    Args:
        analysis_type: Type of analysis (e.g., 'scatterplot', 'bar_chart')
        custom_settings: Dictionary with analysis-specific settings
        
    Returns:
        Complete configuration dictionary
    """
    # Start with base configuration
    config = STANDARD_BASE_CONFIG.copy()
    
    # Add standard visualization settings
    config['visualization'] = STANDARD_VIZ_CONFIG.copy()
    
    # Add standard colors
    config['colors'] = {**GLACIER_COLORS, **METHOD_COLORS}
    
    # Merge custom settings
    for key, value in custom_settings.items():
        if key in config and isinstance(config[key], dict) and isinstance(value, dict):
            config[key].update(value)
        else:
            config[key] = value
    
    return config

# Example usage:
if __name__ == "__main__":
    # Create a custom correlation analysis configuration
    custom_correlation_config = create_custom_config('correlation', {
        'visualization': {'figsize': (14, 12)},
        'quality_filters': CONSERVATIVE_FILTERS,
        'output': OUTPUT_EXAMPLES['correlation']
    })
    
    print("Custom correlation configuration created with:")
    print(f"- Figure size: {custom_correlation_config['visualization']['figsize']}")
    print(f"- Conservative quality filters")
    print(f"- Analysis name: {custom_correlation_config['output']['analysis_name']}")