#!/usr/bin/env python3
"""
Statistical Testing Module for Multi-Glacier Comparative Analysis

This module provides comprehensive statistical testing capabilities for
comparing MODIS albedo method performance across different glaciers.

Statistical Tests Included:
- ANOVA for method comparisons across glaciers
- Post-hoc tests for pairwise comparisons
- Correlation analysis with environmental factors
- Non-parametric tests for non-normal distributions
- Effect size calculations
- Bootstrap confidence intervals
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import pearsonr, spearmanr, kruskal, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import logging
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logger = logging.getLogger(__name__)


class MultiGlacierStatisticalAnalysis:
    """
    Comprehensive statistical analysis for multi-glacier comparisons.
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical analysis with significance level.
        
        Args:
            alpha: Significance level for statistical tests (default: 0.05)
        """
        self.alpha = alpha
        self.results = {}
    
    def method_performance_anova(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform ANOVA to test if method performance differs significantly across glaciers.
        
        Args:
            df: Aggregated dataframe with method performance data
            
        Returns:
            Dictionary containing ANOVA results for each performance metric
        """
        logger.info("Performing ANOVA for method performance across glaciers...")
        
        anova_results = {}
        metrics = ['r', 'rmse', 'mae', 'bias']
        
        for metric in metrics:
            if metric not in df.columns:
                continue
                
            logger.info(f"Running ANOVA for {metric}...")
            
            try:
                # Prepare data for ANOVA
                metric_data = df[['glacier_id', 'method', metric]].dropna()
                
                if len(metric_data) < 3:
                    logger.warning(f"Insufficient data for ANOVA on {metric}")
                    continue
                
                # One-way ANOVA: Does method performance vary across glaciers?
                glacier_groups = [group[metric].values for name, group in metric_data.groupby('glacier_id')]
                
                if len(glacier_groups) < 2:
                    logger.warning(f"Need at least 2 glaciers for ANOVA on {metric}")
                    continue
                
                f_stat, p_value = stats.f_oneway(*glacier_groups)
                
                # Effect size (eta-squared)
                # Calculate sum of squares
                grand_mean = metric_data[metric].mean()
                ss_total = ((metric_data[metric] - grand_mean) ** 2).sum()
                
                glacier_means = metric_data.groupby('glacier_id')[metric].mean()
                glacier_counts = metric_data.groupby('glacier_id')[metric].count()
                ss_between = ((glacier_means - grand_mean) ** 2 * glacier_counts).sum()
                
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                # Post-hoc analysis if significant
                posthoc_results = None
                if p_value < self.alpha:
                    try:
                        # Tukey's HSD for pairwise comparisons
                        tukey_results = pairwise_tukeyhsd(endog=metric_data[metric], 
                                                        groups=metric_data['glacier_id'], 
                                                        alpha=self.alpha)
                        posthoc_results = {
                            'tukey_summary': str(tukey_results),
                            'significant_pairs': []
                        }
                        
                        # Extract significant pairs
                        for i, row in enumerate(tukey_results.summary().data[1:]):
                            if row[5] == 'True':  # Reject null hypothesis
                                posthoc_results['significant_pairs'].append({
                                    'group1': row[0],
                                    'group2': row[1], 
                                    'mean_diff': float(row[2]),
                                    'p_adj': float(row[4])
                                })
                                
                    except Exception as e:
                        logger.warning(f"Post-hoc analysis failed for {metric}: {e}")
                
                anova_results[metric] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'eta_squared': eta_squared,
                    'effect_size_interpretation': self._interpret_eta_squared(eta_squared),
                    'n_groups': len(glacier_groups),
                    'total_n': len(metric_data),
                    'posthoc_results': posthoc_results
                }
                
                logger.info(f"ANOVA for {metric}: F={f_stat:.3f}, p={p_value:.3f}, η²={eta_squared:.3f}")
                
            except Exception as e:
                logger.error(f"ANOVA failed for {metric}: {e}")
                continue
        
        self.results['method_anova'] = anova_results
        return anova_results
    
    def method_comparison_within_glaciers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare method performance within each glacier using paired tests.
        
        Args:
            df: Aggregated dataframe with method performance data
            
        Returns:
            Dictionary containing within-glacier method comparison results
        """
        logger.info("Performing within-glacier method comparisons...")
        
        within_glacier_results = {}
        metrics = ['r', 'rmse', 'mae', 'bias']
        
        for glacier_id in df['glacier_id'].unique():
            glacier_data = df[df['glacier_id'] == glacier_id]
            
            if len(glacier_data) < 2:
                logger.warning(f"Insufficient methods for comparison in {glacier_id}")
                continue
            
            glacier_results = {}
            
            for metric in metrics:
                if metric not in glacier_data.columns:
                    continue
                
                metric_values = glacier_data[metric].dropna()
                methods = glacier_data.loc[metric_values.index, 'method'].values
                
                if len(metric_values) < 2:
                    continue
                
                # Friedman test (non-parametric alternative to repeated measures ANOVA)
                try:
                    # For this we need to reshape data, but since we don't have repeated measures,
                    # we'll use Kruskal-Wallis instead
                    method_groups = [glacier_data[glacier_data['method'] == method][metric].values 
                                   for method in glacier_data['method'].unique()]
                    method_groups = [group for group in method_groups if len(group) > 0]
                    
                    if len(method_groups) >= 2:
                        h_stat, p_value = kruskal(*method_groups)
                        
                        glacier_results[metric] = {
                            'test': 'Kruskal-Wallis',
                            'statistic': h_stat,
                            'p_value': p_value,
                            'significant': p_value < self.alpha,
                            'methods_compared': list(glacier_data['method'].unique()),
                            'best_method': glacier_data.loc[glacier_data[metric].idxmax() if metric == 'r' 
                                                          else glacier_data[metric].idxmin(), 'method'],
                            'worst_method': glacier_data.loc[glacier_data[metric].idxmin() if metric == 'r' 
                                                           else glacier_data[metric].idxmax(), 'method']
                        }
                        
                except Exception as e:
                    logger.warning(f"Within-glacier test failed for {glacier_id} {metric}: {e}")
            
            within_glacier_results[glacier_id] = glacier_results
        
        self.results['within_glacier_comparisons'] = within_glacier_results
        return within_glacier_results
    
    def environmental_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze correlations between environmental factors and method performance.
        
        Args:
            df: Aggregated dataframe with environmental and performance data
            
        Returns:
            Dictionary containing correlation analysis results
        """
        logger.info("Performing environmental correlation analysis...")
        
        environmental_vars = ['elevation', 'latitude', 'longitude']
        performance_vars = ['r', 'rmse', 'mae', 'bias']
        
        correlation_results = {}
        
        for env_var in environmental_vars:
            if env_var not in df.columns:
                continue
                
            var_results = {}
            
            for perf_var in performance_vars:
                if perf_var not in df.columns:
                    continue
                
                # Remove NaN values
                valid_data = df[[env_var, perf_var]].dropna()
                
                if len(valid_data) < 3:
                    continue
                
                try:
                    # Pearson correlation
                    pearson_r, pearson_p = pearsonr(valid_data[env_var], valid_data[perf_var])
                    
                    # Spearman correlation (non-parametric)
                    spearman_r, spearman_p = spearmanr(valid_data[env_var], valid_data[perf_var])
                    
                    var_results[perf_var] = {
                        'pearson_r': pearson_r,
                        'pearson_p': pearson_p,
                        'pearson_significant': pearson_p < self.alpha,
                        'spearman_r': spearman_r,
                        'spearman_p': spearman_p,
                        'spearman_significant': spearman_p < self.alpha,
                        'n_observations': len(valid_data),
                        'relationship_strength': self._interpret_correlation(abs(max(pearson_r, spearman_r, key=abs)))
                    }
                    
                except Exception as e:
                    logger.warning(f"Correlation analysis failed for {env_var} vs {perf_var}: {e}")
            
            correlation_results[env_var] = var_results
        
        self.results['environmental_correlations'] = correlation_results
        return correlation_results
    
    def regional_comparison_tests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare method performance between different regions.
        
        Args:
            df: Aggregated dataframe with regional data
            
        Returns:
            Dictionary containing regional comparison test results
        """
        logger.info("Performing regional comparison tests...")
        
        if 'region' not in df.columns:
            logger.warning("No region information available for regional comparisons")
            return {}
        
        regions = df['region'].unique()
        if len(regions) < 2:
            logger.warning("Need at least 2 regions for comparison")
            return {}
        
        regional_results = {}
        performance_vars = ['r', 'rmse', 'mae', 'bias']
        
        for perf_var in performance_vars:
            if perf_var not in df.columns:
                continue
                
            try:
                # Group data by region
                region_groups = [df[df['region'] == region][perf_var].dropna().values 
                               for region in regions]
                region_groups = [group for group in region_groups if len(group) > 0]
                
                if len(region_groups) < 2:
                    continue
                
                # Mann-Whitney U test for two regions, Kruskal-Wallis for more
                if len(region_groups) == 2:
                    statistic, p_value = mannwhitneyu(region_groups[0], region_groups[1], 
                                                    alternative='two-sided')
                    test_name = 'Mann-Whitney U'
                else:
                    statistic, p_value = kruskal(*region_groups)
                    test_name = 'Kruskal-Wallis'
                
                # Calculate effect size (Cliff's delta for Mann-Whitney)
                effect_size = None
                if len(region_groups) == 2:
                    effect_size = self._cliffs_delta(region_groups[0], region_groups[1])
                
                # Calculate regional means and standard deviations
                regional_stats = {}
                for i, region in enumerate(regions):
                    region_data = df[df['region'] == region][perf_var].dropna()
                    if len(region_data) > 0:
                        regional_stats[region] = {
                            'mean': region_data.mean(),
                            'std': region_data.std(),
                            'median': region_data.median(),
                            'n': len(region_data)
                        }
                
                regional_results[perf_var] = {
                    'test': test_name,
                    'statistic': statistic,
                    'p_value': p_value,
                    'significant': p_value < self.alpha,
                    'effect_size': effect_size,
                    'effect_size_interpretation': self._interpret_cliffs_delta(effect_size) if effect_size else None,
                    'regional_stats': regional_stats,
                    'regions_compared': list(regions)
                }
                
                logger.info(f"Regional comparison for {perf_var}: {test_name} p={p_value:.3f}")
                
            except Exception as e:
                logger.error(f"Regional comparison failed for {perf_var}: {e}")
        
        self.results['regional_comparisons'] = regional_results
        return regional_results
    
    def method_consistency_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze consistency of method performance across glaciers.
        
        Args:
            df: Aggregated dataframe with method performance data
            
        Returns:
            Dictionary containing method consistency analysis results
        """
        logger.info("Performing method consistency analysis...")
        
        consistency_results = {}
        performance_vars = ['r', 'rmse', 'mae', 'bias']
        
        for method in df['method'].unique():
            method_data = df[df['method'] == method]
            
            if len(method_data) < 2:
                continue
            
            method_results = {}
            
            for perf_var in performance_vars:
                if perf_var not in method_data.columns:
                    continue
                
                values = method_data[perf_var].dropna()
                
                if len(values) < 2:
                    continue
                
                # Calculate consistency metrics
                mean_val = values.mean()
                std_val = values.std()
                cv = std_val / abs(mean_val) if mean_val != 0 else np.inf  # Coefficient of variation
                
                method_results[perf_var] = {
                    'mean': mean_val,
                    'std': std_val,
                    'coefficient_of_variation': cv,
                    'min': values.min(),
                    'max': values.max(),
                    'range': values.max() - values.min(),
                    'n_glaciers': len(values),
                    'consistency_rating': self._rate_consistency(cv)
                }
            
            consistency_results[method] = method_results
        
        # Rank methods by consistency
        consistency_rankings = {}
        for perf_var in performance_vars:
            method_cvs = []
            for method, results in consistency_results.items():
                if perf_var in results:
                    method_cvs.append((method, results[perf_var]['coefficient_of_variation']))
            
            # Sort by coefficient of variation (lower = more consistent)
            method_cvs.sort(key=lambda x: x[1])
            consistency_rankings[perf_var] = method_cvs
        
        self.results['method_consistency'] = {
            'individual_methods': consistency_results,
            'consistency_rankings': consistency_rankings
        }
        
        return self.results['method_consistency']
    
    def bootstrap_confidence_intervals(self, df: pd.DataFrame, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Calculate bootstrap confidence intervals for performance metrics.
        
        Args:
            df: Aggregated dataframe
            n_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary containing bootstrap confidence intervals
        """
        logger.info(f"Calculating bootstrap confidence intervals with {n_bootstrap} samples...")
        
        bootstrap_results = {}
        performance_vars = ['r', 'rmse', 'mae', 'bias']
        
        for glacier_id in df['glacier_id'].unique():
            glacier_data = df[df['glacier_id'] == glacier_id]
            glacier_results = {}
            
            for method in glacier_data['method'].unique():
                method_data = glacier_data[glacier_data['method'] == method]
                method_results = {}
                
                for perf_var in performance_vars:
                    if perf_var not in method_data.columns or method_data[perf_var].isna().all():
                        continue
                    
                    observed_value = method_data[perf_var].iloc[0]  # Single value per method-glacier
                    
                    # Since we have single values, we'll bootstrap across methods within glacier
                    # or across glaciers within method for CI estimation
                    
                    # Bootstrap across all data points for this performance variable
                    all_values = df[df['method'] == method][perf_var].dropna().values
                    
                    if len(all_values) < 3:
                        continue
                    
                    bootstrap_means = []
                    np.random.seed(42)  # For reproducibility
                    
                    for _ in range(n_bootstrap):
                        bootstrap_sample = np.random.choice(all_values, size=len(all_values), replace=True)
                        bootstrap_means.append(np.mean(bootstrap_sample))
                    
                    # Calculate confidence intervals
                    ci_lower = np.percentile(bootstrap_means, (1 - 0.95) / 2 * 100)
                    ci_upper = np.percentile(bootstrap_means, (1 + 0.95) / 2 * 100)
                    
                    method_results[perf_var] = {
                        'observed_value': observed_value,
                        'bootstrap_mean': np.mean(bootstrap_means),
                        'bootstrap_std': np.std(bootstrap_means),
                        'ci_lower': ci_lower,
                        'ci_upper': ci_upper,
                        'ci_width': ci_upper - ci_lower
                    }
                
                glacier_results[method] = method_results
            
            bootstrap_results[glacier_id] = glacier_results
        
        self.results['bootstrap_ci'] = bootstrap_results
        return bootstrap_results
    
    def _interpret_eta_squared(self, eta_squared: float) -> str:
        """Interpret eta-squared effect size."""
        if eta_squared < 0.01:
            return "negligible"
        elif eta_squared < 0.06:
            return "small"
        elif eta_squared < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_correlation(self, r: float) -> str:
        """Interpret correlation coefficient magnitude."""
        r = abs(r)
        if r < 0.1:
            return "negligible"
        elif r < 0.3:
            return "small"
        elif r < 0.5:
            return "medium"
        elif r < 0.7:
            return "large"
        else:
            return "very large"
    
    def _cliffs_delta(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Cliff's delta effect size."""
        if len(x) == 0 or len(y) == 0:
            return 0.0
        
        diff_count = 0
        total_comparisons = len(x) * len(y)
        
        for x_val in x:
            for y_val in y:
                if x_val > y_val:
                    diff_count += 1
                elif x_val < y_val:
                    diff_count -= 1
        
        return diff_count / total_comparisons
    
    def _interpret_cliffs_delta(self, delta: float) -> str:
        """Interpret Cliff's delta effect size."""
        delta = abs(delta)
        if delta < 0.147:
            return "negligible"
        elif delta < 0.33:
            return "small"
        elif delta < 0.474:
            return "medium"
        else:
            return "large"
    
    def _rate_consistency(self, cv: float) -> str:
        """Rate consistency based on coefficient of variation."""
        if cv < 0.1:
            return "very consistent"
        elif cv < 0.2:
            return "consistent"
        elif cv < 0.3:
            return "moderately consistent"
        else:
            return "inconsistent"
    
    def export_statistical_results(self, output_dir: Path) -> None:
        """Export all statistical test results to files."""
        logger.info("Exporting statistical test results...")
        
        results_dir = output_dir / "results"
        
        # Export ANOVA results
        if 'method_anova' in self.results:
            anova_df = pd.DataFrame.from_dict(
                {metric: {k: v for k, v in results.items() if k != 'posthoc_results'} 
                 for metric, results in self.results['method_anova'].items()}, 
                orient='index'
            )
            anova_df.to_csv(results_dir / "statistical_anova_results.csv")
        
        # Export regional comparison results
        if 'regional_comparisons' in self.results:
            regional_df = pd.DataFrame.from_dict(
                {metric: {k: v for k, v in results.items() if k != 'regional_stats'} 
                 for metric, results in self.results['regional_comparisons'].items()}, 
                orient='index'
            )
            regional_df.to_csv(results_dir / "statistical_regional_comparisons.csv")
        
        # Export method consistency results
        if 'method_consistency' in self.results:
            consistency_data = []
            for method, metrics in self.results['method_consistency']['individual_methods'].items():
                for metric, stats in metrics.items():
                    row = {'method': method, 'performance_metric': metric}
                    row.update(stats)
                    consistency_data.append(row)
            
            consistency_df = pd.DataFrame(consistency_data)
            consistency_df.to_csv(results_dir / "statistical_method_consistency.csv", index=False)
        
        logger.info(f"Statistical results exported to {results_dir}")
    
    def run_comprehensive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Run all statistical analyses and return comprehensive results.
        
        Args:
            df: Aggregated dataframe with all glacier data
            
        Returns:
            Dictionary containing all statistical analysis results
        """
        logger.info("Running comprehensive statistical analysis...")
        
        # Run all analyses
        self.method_performance_anova(df)
        self.method_comparison_within_glaciers(df)
        self.environmental_correlation_analysis(df)
        self.regional_comparison_tests(df)
        self.method_consistency_analysis(df)
        self.bootstrap_confidence_intervals(df)
        
        logger.info("Comprehensive statistical analysis completed")
        return self.results


def main():
    """Test the statistical analysis module."""
    print("Multi-Glacier Statistical Analysis Module initialized")
    print("Available analyses:")
    print("- ANOVA for method performance across glaciers")
    print("- Within-glacier method comparisons")
    print("- Environmental correlation analysis") 
    print("- Regional comparison tests")
    print("- Method consistency analysis")
    print("- Bootstrap confidence intervals")


if __name__ == "__main__":
    main()