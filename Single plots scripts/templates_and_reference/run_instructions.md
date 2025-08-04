# Reproducible Run Instructions (without changing existing paths)

Prereqs
- Python 3.10+
- Install deps: pip install -r requirements.txt

How to run the Haig trend analysis
- Ensure the CSVs referenced in CONFIG exist at the current absolute paths in sen_slope_mann_kendall_trend_analysis.py
- Run: python sen_slope_mann_kendall_trend_analysis.py
- Outputs: timestamped directory under outputs/<run>/ with plots/ and results/; summary.txt appended with trend tables

What the script does
- Loads MODIS, AWS, and temperature/AOD data
- Selects best MODIS pixel(s), aggregates melt-season annual series
- Applies Mann–Kendall (optionally Yue–Pilon prewhitening) and Sen slope with 95% CI
- Produces trends and correlations plots and an augmented summary

Quality/Reporting
- Reports n, period, τ, p, prewhitening flag, Sen slope ±95% CI
- Sensitivity: JAS vs configured months; mean vs median aggregation summary noted

Tests
- Run: pytest -q
- Synthetic tests validate MK/Sen behavior and prewhitening effect
