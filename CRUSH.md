# CRUSH Configuration for MODIS Albedo Analysis Framework

## Build/Lint/Test Commands

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_analysis.py

# Run individual test method
pytest tests/test_analysis.py::TestAlbedoAnalysis::test_comprehensive_analysis

# Run with coverage
pytest --cov=analysis --cov=data_processing --cov=visualization

# Type checking
mypy .

# Linting
flake8 .
black --check .

# Format code
black .

# Run the framework (interactive mode)
python interactive_main.py

# Run specific glacier analysis
python interactive_main.py --glacier athabasca --analysis-mode comprehensive

# Run comparative analysis
python interactive_main.py --comparative-analysis
```

## Code Style Guidelines

### Import Ordering
1. Standard library imports (os, sys, pathlib)
2. Third-party imports (numpy, pandas, matplotlib)
3. Local project imports (relative imports)

### Naming Conventions
- Classes: PascalCase (AlbedoCalculator, StatisticalAnalyzer)
- Methods/Functions: snake_case (calculate_metrics, load_modis_data)
- Constants: UPPER_SNAKE_CASE (DEFAULT_CONFIDENCE_LEVEL)
- Variables: snake_case (modis_data, aws_station)

### Type Hints
- Use extensive type hints for function parameters and return types
- Common patterns: Dict[str, Any], pd.DataFrame, pd.Series, Optional[Type]

### Error Handling
- Use try/except blocks with specific error logging
- Graceful fallbacks returning empty data structures when missing data
- Warning logs for non-critical issues

### Documentation
- Google-style docstrings for classes and methods
- Args/Returns sections in docstrings
- Inline comments for complex calculations

### Code Organization
- Small, focused classes with single responsibility
- Private helper methods start with underscore (_)
- Configuration passed as dictionary parameters
- Clear separation between data processing, analysis, and visualization