# Haig Glacier Albedo – Multiple Regression Analysis (Detailed Documentation)

This document explains, in detail, the analysis pipeline, data handling, models, diagnostics, robustness checks, and how to interpret the results that appear in `summary.txt` and the saved figures. It is written to be a self-contained methods and results description for a supplement or README.

## Contents
- Overview
- Data sources and preprocessing
- Annual aggregation (seasonal definition)
- Modeling strategy
  - Primary additive model
  - Standardized effect sizes
  - Centered interaction model
- Model estimation and statistics
  - Ordinary Least Squares (OLS)
  - Variance Inflation Factor (VIF)
  - Partial regression
  - Influence diagnostics (Cook’s distance)
- Diagnostic plots
- Robustness checks
  - Trimmed periods
  - Median aggregation
- Key results and interpretation
- Reproducibility: files produced and where to find them
- Limitations and recommended next steps

---

## Overview

Goal: Quantify how summer albedo over Haig Glacier varies with summer near-surface temperature and regional black carbon aerosol optical depth (BC_AOD). Specifically:
- Estimate the partial effect of Temperature on Albedo after controlling for BC_AOD, and vice versa.
- Assess whether Temperature’s effect depends on BC_AOD (an interaction).
- Check linear model assumptions and ensure results are not driven by a few influential years.
- Provide standardized effect sizes to compare the relative importance of predictors.

All results are saved into the latest run directory like:
`outputs/haig_trend_analysis_YYYYMMDD_HHMMSS/`

Key outputs:
- `summary.txt`: model summaries and robustness checks
- Diagnostic plots: residuals, Q–Q, pairwise, partial regression, and interaction simple slopes

---

## Data sources and preprocessing

Input paths are configured in the script (`CONFIG['data_paths']`):
- MODIS albedo products (multiple methods possible; we use `MOD09GA` here).
- AWS (Automatic Weather Station) albedo time series.
- Temperature and regional `BC_AOD` (daily).

Preprocessing steps:
1) Read each dataset; standardize column names; parse `date`.
2) For MODIS:
   - If the data are wide, convert to long format with columns: `pixel_id`, `date`, `method`, `Albedo`, plus coordinates and glacier fraction where available.
   - Filter to physically plausible albedo [0, 1].
3) AWS:
   - Build `date` from year/day; select the albedo column; filter to (0, 1].
4) Temperature file:
   - Extract `Temperature` (°C) and, if present, `BC_AOD` (unitless AOD proxy).
5) Pixel selection:
   - Among available MODIS pixels, pick the “best” pixel by combining observation count, average glacier fraction, and distance to the AWS site (closest and most glaciated with sufficient observations).
6) Time-series preparation:
   - For each series, keep `date` and the value column; derive `year`, `month`, `day_of_year` for convenience.

---

## Annual aggregation (seasonal definition)

The analysis focuses on JJAS (June–September) to represent melt-season conditions:
- Filter by months [6, 7, 8, 9].
- Aggregate to annual values using the mean (primary) or the median (robustness check).
- Build annual series for:
  - Albedo (from selected MODIS method)
  - Temperature
  - BC_AOD

The final analysis table merges the three annual series on `date` and drops years with missing values.

---

## Modeling strategy

We fit three closely related linear models, all using Ordinary Least Squares (OLS) on annual JJAS data:

### 1) Primary additive model
Albedo ~ Temperature + BC_AOD

Purpose:
- Estimate the partial effect of Temperature on Albedo after controlling for BC_AOD.
- Estimate the partial effect of BC_AOD on Albedo after controlling for Temperature.

Reported:
- Coefficients, standard errors, t-statistics, p-values, R², adjusted R².
- VIF for Temperature and BC_AOD.

Interpretation:
- The Temperature coefficient is the expected change in Albedo (unitless) for +1 °C, holding BC_AOD fixed.
- The BC_AOD coefficient is the expected change in Albedo for a +0.001 change in BC_AOD (or native units), holding Temperature fixed.

### 2) Standardized effect sizes
Albedo ~ Temperature_z + BC_AOD_z

Method:
- Z-score only the predictors: subtract mean and divide by standard deviation (SD). Albedo remains in its native units.
- Coefficients now represent the change in Albedo associated with a 1 SD increase in each predictor.

Why:
- Allows direct comparison of predictor effect sizes regardless of their units.

### 3) Centered interaction model (secondary check)
Albedo ~ T_c + B_c + T_c×B_c

Method:
- Mean-center predictors to reduce multicollinearity in interaction models:
  - T_c = Temperature − mean(Temperature)
  - B_c = BC_AOD − mean(BC_AOD)
  - Interaction term: T_c×B_c
- Report coefficients and VIFs.

Interpretation:
- The T_c coefficient is the Temperature effect at average BC_AOD.
- The interaction term tests whether the Temperature–Albedo slope changes as BC_AOD departs from its mean.

Note:
- The additive model remains the primary result. The centered interaction is exploratory; we include it once with centering to make it well-conditioned.

---

## Model estimation and statistics

### Ordinary Least Squares (OLS)
- The script solves the normal equations to obtain β, then computes standard errors from the residual variance and the inverse of X'X.
- R² quantifies the fraction of variance in Albedo explained by the predictors.
- Adjusted R² penalizes model complexity.

Assumptions:
- Linearity, independence, homoscedasticity (constant variance), and normally distributed residuals (for inference).

### Variance Inflation Factor (VIF)
- VIF for predictor j = 1 / (1 − R²_j), where R²_j is from regressing predictor j on other predictors.
- VIF ~ 1 means negligible collinearity; >5 or >10 indicates notable or severe collinearity.
- We report VIFs for the additive model and for the centered interaction model.

### Partial regression
- Visual explanation of partial effects:
  - Regress Y on the “other” predictor(s) and take residuals.
  - Regress X on the “other” predictor(s) and take residuals.
  - Plot residual(Y|others) vs residual(X|others). The slope matches the partial regression coefficient (up to scaling).

### Influence diagnostics (Cook’s distance)
- Cook’s D measures how much a point influences the overall fit.
- Rule-of-thumb flag: D > 4/n.
- We list indices of high-D years and refit the additive model without them to show sensitivity.

---

## Diagnostic plots

Saved in the run folder:
- `residuals_vs_fitted.png`: Checks for nonlinearity or heteroscedasticity. Random scatter around zero is desirable.
- `residuals_qq_plot.png`: Compares residual quantiles to normal quantiles. Near-linearity suggests acceptable normality.
- `regression_pairs.png`: Pairwise scatter plots with simple linear fits for Albedo–Temperature, Albedo–BC_AOD, and Temperature–BC_AOD.
- `partial_regression_temperature.png`: Partial effect of Temperature controlling for BC_AOD.
- `partial_regression_bc_aod.png`: Partial effect of BC_AOD controlling for Temperature.
- `interaction_simple_slopes.png`: Fitted lines showing Temperature–Albedo slopes at low/median/high BC_AOD (using the centered-interaction model).

How to read them:
- If residuals show curvature or funnel shapes, consider mild nonlinearity or variance-stabilizing approaches. In our case, diagnostics look acceptable for OLS.
- Partial plots reveal Temperature’s robust negative partial relationship with Albedo; BC_AOD’s partial relationship is weak.

---

## Robustness checks

### Trimmed periods
- Drop the first 3 years and refit (to check early years aren’t driving results).
- Drop the last 3 years and refit (to check recent years aren’t driving results).

Outcome:
- Temperature stays significantly negative; BC_AOD remains non-significant.

### Median aggregation
- Rebuild JJAS annual series using the median instead of the mean and refit the additive model.

Outcome:
- Overall fit improves (higher adjusted R²).
- Temperature’s negative effect becomes slightly stronger; BC_AOD remains non-significant.

---

## Key results and interpretation

From your `summary.txt`:

- Primary additive model (mean JJAS):
  - R² ≈ 0.58, adj. R² ≈ 0.54.
  - Temperature effect: about −0.052 albedo units per +1 °C (SE ≈ 0.016, p ≈ 0.003). Strong negative relationship.
  - BC_AOD effect: negative but not statistically significant (p ≈ 0.24).
  - VIFs ≈ 1.6: low collinearity in the additive model.

- Standardized coefficients:
  - Temperature_z ≈ −0.058 (p ≈ 0.003), clearly larger in magnitude than BC_AOD_z ≈ −0.021 (p ≈ 0.24).
  - Conclusion: Temperature is the dominant predictor of interannual albedo variability.

- Influence (Cook’s distance):
  - Two high-influence years flagged. Excluding them leaves conclusions unchanged; Temperature remains significant, BC_AOD remains not significant. VIFs drop further.

- Centered interaction model:
  - VIFs lowered to ~1.5–2.3 due to centering.
  - Interaction term not significant (p ≈ 0.20). Temperature still significantly negative (p ≈ 0.008) at mean BC_AOD.
  - Interpretation: No robust evidence that temperature sensitivity of albedo depends on BC_AOD at this sample size and data resolution.

- Robustness checks (trimmed, median):
  - All lead to the same conclusion: Temperature is a stable, significant negative driver of JJAS albedo; BC_AOD’s effect is consistently negative but not statistically robust here.
  - Median aggregation improves fit and slightly strengthens the Temperature effect.

Practical summary:
- For this glacier and period, warmer JJAS seasons are reliably associated with lower surface albedo.
- A soot proxy (BC_AOD) is directionally consistent with darkening but not statistically decisive after accounting for temperature with these data.

---

## Reproducibility: files produced and where to find them

In your latest run directory (`outputs/haig_trend_analysis_*`):
- `summary.txt`: Text summary containing:
  - Additive model results + VIF
  - Standardized coefficients
  - Cook’s distance summary and optional refit without high-D years
  - Centered interaction model + VIF
  - Robustness: trimmed periods and median aggregation
- Figures:
  - `regression_pairs.png`
  - `partial_regression_temperature.png`
  - `partial_regression_bc_aod.png`
  - `residuals_vs_fitted.png`
  - `residuals_qq_plot.png`
  - `interaction_simple_slopes.png`

---

## Limitations and recommended next steps

Limitations:
- Sample size is modest (n ≈ 23 annual observations). This can limit power, particularly for interaction tests.
- BC_AOD is a regional proxy; it doesn’t capture on-glacier deposition, post-deposition processes, or coincident dust/organic impurities.
- Omitted variables: snow depth/timing, precipitation, cloud cover, radiative forcing, circulation indices, etc., may further explain variance.

Next steps (lightweight):
- Report the additive model as primary and include standardized betas.
- Retain the centered interaction model as a non-significant exploratory result.
- Keep the Cook’s distance note; if desired, identify which calendar years correspond to the flagged indices and briefly comment.
- Optionally export a CSV table of all model summaries for manuscript tables.

Heavier options (only if needed):
- Include a quadratic Temperature term if residuals ever suggest curvature.
- Use ridge regression for interaction stabilization if you explore more predictors.
- Consider adding time trends or lagged terms if there’s autocorrelation (not evident here, but worth checking with Durbin–Watson if you expand the model).
