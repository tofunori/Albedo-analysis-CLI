# Haig Trend Analysis Analysis

## Analysis Description

Non-parametric trend analysis of Haig glacier albedo (2002-2024)

## Analysis Metadata

- **Generated**: 2025-08-03T11:05:10.841800
- **Analysis Type**: haig_trend_analysis
- **Output Directory**: `haig_trend_analysis_20250803_110510`

## Directory Structure

```
haig_trend_analysis_20250803_110510/
├── plots/          # Generated visualizations
├── results/        # Statistical outputs and summaries
└── README.md       # This documentation
```

## Key Findings

1. MODIS albedo shows significant decreasing trend of -0.0087 units/yr (p=0.015) with prewhitening

## Generated Files

### Plots
- `haig_correlations_analysis.png`
- `haig_trends_analysis.png`

### Results
- `summary.txt`

## Additional Information

**Analysis Period**: 2002-2024 (based on available data)

**Methodology**: Mann–Kendall trend test (optional Yue–Pilon prewhitening) and Sen slope estimator

**Statistical Significance**: alpha = 0.05 (95% confidence level)

**Data Sources**: AWS measurements and MODIS satellite observations

**Quality Control**: Minimum 5 years of data required

**Seasonal Definition**: Melt season months: [6, 7, 8, 9]; aggregation: mean

**Sensitivity**: Sensitivity JAS: slope=-0.0079 units/yr, p=0.042; Sensitivity median: slope=-0.0094 units/yr, p=0.080

