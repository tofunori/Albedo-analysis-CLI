"""
Residual Analysis Generator for MODIS vs AWS Albedo Validation

Generates comprehensive residual plots to identify systematic biases, 
heteroscedasticity, and error patterns in albedo retrieval methods.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class ResidualAnalyzer:
    """Advanced residual analysis for albedo validation."""
    
    def __init__(self, style: str = 'scientific'):
        self.setup_style(style)
        self.colors = {'MCD43A3': '#2E86AB', 'MOD09GA': '#A23B72', 'MOD10A1': '#F18F01'}
    
    def setup_style(self, style: str):
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 11,
            'xtick.labelsize': 9, 'ytick.labelsize': 9, 'legend.fontsize': 9,
            'figure.titlesize': 14, 'axes.grid': True, 'grid.alpha': 0.3, 'figure.dpi': 150
        })
    
    def calculate_residuals(self, aws_data: pd.Series, modis_data: pd.Series) -> Dict:
        # Data alignment and cleaning
        common_idx = aws_data.index.intersection(modis_data.index)
        aws_clean = aws_data.loc[common_idx].dropna()
        modis_clean = modis_data.loc[common_idx].dropna()
        final_common = aws_clean.index.intersection(modis_clean.index)
        
        if len(final_common) < 3:
            return {'residuals': np.array([]), 'fitted': np.array([]), 'n': 0}
        
        aws_vals = aws_clean.loc[final_common].values
        modis_vals = modis_clean.loc[final_common].values
        
        # Core calculations
        residuals = modis_vals - aws_vals
        slope, intercept = np.polyfit(aws_vals, modis_vals, 1)
        fitted_values = slope * aws_vals + intercept
        residual_std = np.sqrt(np.mean(residuals**2))
        standardized_residuals = residuals / residual_std if residual_std > 0 else residuals
        
        diagnostics = {
            'residuals': residuals, 'fitted': fitted_values, 'aws_values': aws_vals,
            'modis_values': modis_vals, 'standardized_residuals': standardized_residuals,
            'n': len(residuals), 'slope': slope, 'intercept': intercept,
            'residual_std': residual_std, 'mean_residual': np.mean(residuals),
            'median_residual': np.median(residuals)
        }
        
        # Heteroscedasticity test
        if len(residuals) > 10:
            abs_residuals = np.abs(residuals)
            heteroscedasticity_corr = np.corrcoef(fitted_values, abs_residuals)[0,1]
            diagnostics['heteroscedasticity_indicator'] = abs(heteroscedasticity_corr)
        
        return diagnostics
    
    def create_comprehensive_residual_plots(self, 
                                          aws_data: pd.Series,
                                          modis_methods: Dict[str, pd.Series],
                                          glacier_name: str = "Glacier",
                                          output_path: Optional[str] = None) -> plt.Figure:
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{glacier_name} - Residual Analysis for MODIS Albedo Methods', 
                    fontsize=14, fontweight='bold', y=0.95)
        
        # Calculate residuals for all methods
        method_diagnostics = {method: self.calculate_residuals(aws_data, modis_data) 
                            for method, modis_data in modis_methods.items()}
        
        # Panel 1: Residuals vs Fitted Values
        self._plot_residuals_vs_fitted(axes[0, 0], method_diagnostics)
        
        # Panel 2: Normal Q-Q Plot
        self._plot_qq_normal(axes[0, 1], method_diagnostics)
        
        # Panel 3: Residuals vs AWS Values
        self._plot_residuals_vs_aws(axes[1, 0], method_diagnostics)
        
        # Panel 4: Residual Distribution
        self._plot_residual_distribution(axes[1, 1], method_diagnostics)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
            print(f"Residual plots saved to: {output_path}")
        
        return fig
    
    def _plot_residuals_vs_fitted(self, ax, method_diagnostics):
        for method, diag in method_diagnostics.items():
            if diag['n'] > 0:
                color = self.colors.get(method, 'gray')
                ax.scatter(diag['fitted'], diag['residuals'], alpha=0.6, s=25, 
                          color=color, label=method, edgecolors='white', linewidth=0.5)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1)
        ax.set_xlabel('Fitted Values (MODIS Albedo)')
        ax.set_ylabel('Residuals (MODIS - AWS)')
        ax.set_title('Residuals vs Fitted Values\n(Detect Non-linearity & Heteroscedasticity)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_qq_normal(self, ax, method_diagnostics):
        for method, diag in method_diagnostics.items():
            if diag['n'] > 0:
                color = self.colors.get(method, 'gray')
                standardized_res = diag['standardized_residuals']
                theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(standardized_res)))
                sample_quantiles = np.sort(standardized_res)
                ax.scatter(theoretical_quantiles, sample_quantiles, alpha=0.6, s=25, 
                          color=color, label=method, edgecolors='white', linewidth=0.5)
        
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]), np.max([ax.get_xlim(), ax.get_ylim()])]
        ax.plot(lims, lims, 'r--', alpha=0.8, linewidth=1)
        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.set_title('Q-Q Plot (Normal Distribution)\n(Check Residual Normality)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_residuals_vs_aws(self, ax, method_diagnostics):
        for method, diag in method_diagnostics.items():
            if diag['n'] > 0:
                color = self.colors.get(method, 'gray')
                ax.scatter(diag['aws_values'], diag['residuals'], alpha=0.6, s=25, 
                          color=color, label=method, edgecolors='white', linewidth=0.5)
        
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=1)
        ax.set_xlabel('AWS Albedo (Reference)')
        ax.set_ylabel('Residuals (MODIS - AWS)')
        ax.set_title('Residuals vs AWS Albedo\n(Check Range-Dependent Bias)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_residual_distribution(self, ax, method_diagnostics):
        for method, diag in method_diagnostics.items():
            if diag['n'] > 0:
                color = self.colors.get(method, 'gray')
                ax.hist(diag['residuals'], bins=20, alpha=0.6, color=color, label=method, 
                       density=True, edgecolor='white', linewidth=0.5)
        
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.8, linewidth=1)
        ax.set_xlabel('Residuals (MODIS - AWS)')
        ax.set_ylabel('Density')
        ax.set_title('Residual Distribution\n(Check Symmetry & Outliers)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_residual_diagnostic_summary(self, 
                                         aws_data: pd.Series,
                                         modis_methods: Dict[str, pd.Series],
                                         glacier_name: str = "Glacier") -> pd.DataFrame:
        
        diagnostic_results = []
        
        for method, modis_data in modis_methods.items():
            diag = self.calculate_residuals(aws_data, modis_data)
            
            if diag['n'] > 0:
                # Statistical tests
                normality_p = self._test_normality(diag['residuals'], diag['n'])
                durbin_watson = self._test_autocorrelation(diag['residuals'], diag['n'])
                
                diagnostic_results.append({
                    'Method': method, 'N_Samples': diag['n'],
                    'Mean_Residual': diag['mean_residual'], 'Median_Residual': diag['median_residual'],
                    'Residual_Std': diag['residual_std'], 'Normality_p_value': normality_p,
                    'Heteroscedasticity_Indicator': diag.get('heteroscedasticity_indicator', np.nan),
                    'Durbin_Watson': durbin_watson, 'Regression_Slope': diag['slope'],
                    'Regression_Intercept': diag['intercept']
                })
        
        df_diagnostics = pd.DataFrame(diagnostic_results)
        df_diagnostics['Diagnostic_Issues'] = df_diagnostics.apply(self._interpret_diagnostics, axis=1)
        
        return df_diagnostics
    
    def _test_normality(self, residuals, n):
        if 3 <= n <= 5000:
            try:
                _, normality_p = stats.shapiro(residuals)
                return normality_p
            except Exception:
                pass
        return np.nan
    
    def _test_autocorrelation(self, residuals, n):
        if n > 2:
            diff_residuals = np.diff(residuals)
            return np.sum(diff_residuals**2) / np.sum(residuals**2)
        return np.nan
    
    def _interpret_diagnostics(self, row):
        issues = []
        
        if abs(row['Mean_Residual']) > 0.05:
            issues.append("Significant bias")
        if not pd.isna(row['Normality_p_value']) and row['Normality_p_value'] < 0.05:
            issues.append("Non-normal residuals")
        if not pd.isna(row['Heteroscedasticity_Indicator']) and row['Heteroscedasticity_Indicator'] > 0.3:
            issues.append("Heteroscedasticity")
        if not pd.isna(row['Durbin_Watson']) and (row['Durbin_Watson'] < 1.5 or row['Durbin_Watson'] > 2.5):
            issues.append("Autocorrelation")
        
        return "; ".join(issues) if issues else "Good"

def main():
    """Example usage with synthetic data."""
    print("MODIS Albedo Residual Analysis Generator")
    print("="*50)
    
    # Generate synthetic data
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=200, freq='D')
    
    aws_albedo = pd.Series(
        0.3 + 0.2 * np.random.beta(2, 2, 200) + 0.1 * np.sin(2*np.pi*np.arange(200)/365),
        index=dates
    )
    
    modis_methods = {
        'MCD43A3': aws_albedo + np.random.normal(0, 0.05, 200) - 0.02,
        'MOD09GA': aws_albedo + np.random.normal(0, 0.08, 200) + 0.03 * (aws_albedo - 0.5),
        'MOD10A1': aws_albedo + np.random.normal(0, 0.06, 200) * (1 + aws_albedo)
    }
    
    # Analysis pipeline
    analyzer = ResidualAnalyzer()
    
    analyzer.create_comprehensive_residual_plots(
        aws_albedo, modis_methods, 
        glacier_name="Example Glacier",
        output_path="residual_analysis_example.png"
    )
    
    diagnostics_df = analyzer.create_residual_diagnostic_summary(
        aws_albedo, modis_methods, "Example Glacier"
    )
    
    print("\nDiagnostic Summary:")
    print(diagnostics_df.round(4))
    
    plt.show()

if __name__ == "__main__":
    main()
