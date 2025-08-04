# CRUSH.md

Project: Glacier albedo analysis (Python scripts + templates)

Build/run
- Python 3.10+ with numpy, pandas, matplotlib, scipy
- Run an analysis script: python <script>.py (e.g., python sen_slope_mann_kendall_trend_analysis.py)
- Jupyter notebooks live in jupyter/*.ipynb for exploratory runs

Lint/format/type
- Lint: ruff .
- Format: ruff format .  (or: black . if black is preferred)
- Type check: mypy .  (focus on templates_and_reference/analysis_template.py patterns)

Tests
- No test framework configured. To add pytest: pip install pytest; put tests/ and run: pytest -q
- Run a single test (once pytest exists): pytest -q tests/<file>::<test_name>

Code style (from templates_and_reference/CODE_STRUCTURE_REFERENCE.md)
- Imports order: stdlib (logging, datetime, pathlib) → scientific (matplotlib.pyplot as plt, numpy as np, pandas as pd, warnings) → stats (scipy.stats) → typing → local (from output_manager import OutputManager)
- Section headers: 76 '=' divider lines; modules grouped under IMPORTS, LOGGING SETUP, CONFIGURATION, MODULES, SUMMARY, MAIN
- Logging: warnings.filterwarnings('ignore'); logging.basicConfig(level=INFO, format='%(asctime)s - %(levelname)s - %(message)s'); logger = logging.getLogger(__name__)
- Types: use typing hints for all public functions; return explicit types; prefer pandas.DataFrame, numpy.ndarray, Dict[str, Any], Tuple[...]
- Naming: snake_case for vars/functions; PascalCase for classes; private helpers prefixed with _; consistent ids: glacier_id, method, pixel_id
- Config: central CONFIG dict with data_paths, aws_stations, colors, methods, method_mapping, outlier_threshold, quality_filters, visualization, output
- Error handling: FileNotFoundError for missing paths; ValueError for bad inputs; warn and continue on insufficient data; return empty DataFrames instead of None
- Data standards: columns include date (pd.Timestamp), albedo (MODIS), Albedo (AWS), method, glacier_id, pixel_id, latitude, longitude, glacier_fraction
- Visualization: use CONFIG['visualization'] for figsize/dpi/style; consistent colors from CONFIG; legends/labels; dpi=300; tight_layout; save via OutputManager
- Output: always use OutputManager for timestamped outputs, plot paths, save_summary, save_readme

Cursor/Copilot rules
- No .cursor/ or .github/copilot-instructions.md found; follow the above templates as authoritative style.
