#!/usr/bin/env python3
"""
Multi-Glacier Comparative Analysis Interface

This module provides the main interface for conducting comprehensive
comparative analysis across all glaciers in the framework. It integrates
data aggregation, visualization, and statistical testing into a unified
workflow accessible from the interactive menu.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.analysis.comparative_analysis import MultiGlacierComparativeAnalysis
from src.analysis.multi_glacier_plots import MultiGlacierVisualizer
from src.analysis.statistical_testing import MultiGlacierStatisticalAnalysis
from src.analysis.glacier_mapping_simple import MultiGlacierMapperSimple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ComparativeAnalysisInterface:
    """
    Main interface for multi-glacier comparative analysis.
    
    This class provides a user-friendly interface for conducting comprehensive
    comparative analysis across multiple glaciers with interactive menu system.
    
    Features:
    - Interactive CLI menu with multiple analysis modes
    - Automatic data discovery and validation
    - Pixel selection optimization (all pixels vs best pixels)
    - Statistical testing suite (ANOVA, regional comparisons, correlations)
    - 7 types of publication-ready visualizations
    - Geographic mapping with pixel selection visualization
    - Comprehensive reporting and data export
    
    Analysis Modes:
    1. Quick Overview: Essential plots and key statistics
    2. Full Statistical Analysis: All tests and visualizations  
    3. Custom Analysis: User-selected components
    4. Best Pixel Analysis: Optimized pixel selection (RECOMMENDED)
    
    Components:
    - data_analyzer: Core data aggregation and processing
    - visualizer: Plot generation and visualization management
    - statistician: Statistical testing and analysis suite
    - mapper: Geographic mapping and pixel visualization
    
    Example Usage:
        >>> interface = ComparativeAnalysisInterface()
        >>> interface.run_interactive_session()  # Interactive menu
        >>> # OR programmatic usage:
        >>> interface.run_best_pixel_analysis()  # Direct analysis
    """
    
    def __init__(self, outputs_dir: str = "outputs", config_path: str = "config/glacier_sites.yaml"):
        """
        Initialize the comparative analysis interface.
        
        Args:
            outputs_dir (str): Directory for storing analysis outputs and results.
                             Will create timestamped subdirectories for each analysis.
            config_path (str): Path to YAML configuration file containing glacier
                             definitions, coordinates, and analysis parameters.
        
        Note:
            Initializes all component classes and prepares for analysis.
            No data is loaded until analysis methods are called.
        """
        self.outputs_dir = outputs_dir
        self.config_path = config_path
        
        # Initialize components
        self.data_analyzer = MultiGlacierComparativeAnalysis(outputs_dir, config_path)
        self.visualizer = MultiGlacierVisualizer()
        self.statistician = MultiGlacierStatisticalAnalysis()
        self.mapper = MultiGlacierMapperSimple(config_path)
        
        # Results storage
        self.aggregated_data = None
        self.output_directory = None
        
        logger.info("Comparative Analysis Interface initialized")
    
    def display_welcome_menu(self) -> None:
        """Display welcome message and analysis options."""
        print("\n" + "=" * 70)
        print("    Multi-Glacier Comparative Analysis Suite")
        print("=" * 70)
        print()
        print("This module provides comprehensive comparative analysis across all")
        print("glaciers in your framework, including:")
        print()
        print("[STATS] Statistical Comparisons:")
        print("   • Method performance ANOVA across glaciers")
        print("   • Regional comparison tests (Canadian Rockies vs Peruvian Andes)")
        print("   • Environmental correlation analysis")
        print("   • Method consistency assessment")
        print()
        print("[PLOTS] Advanced Visualizations (7 plot types):")
        print("   • Method performance matrix heatmap")
        print("   • AWS vs MODIS albedo scatterplot matrix (3×3)")
        print("   • Glacier-specific method performance analysis")
        print("   • Sample size vs performance analysis")
        print("   • Bias comparison radar charts")
        print("   • Combined statistical confidence dashboard")
        print("   • Multi-glacier mapping suite (individual + overview + correlation maps)")
        print()
        print("📋 Comprehensive Reporting:")
        print("   • Detailed statistical test results")
        print("   • Method ranking recommendations")
        print("   • Publication-ready visualizations")
        print()
    
    def display_analysis_options(self) -> None:
        """Display analysis options menu."""
        print("Comparative Analysis Options:")
        print("-" * 30)
        print("1. Quick Overview (Choose pixel mode)")
        print("2. Full Statistical Analysis (Choose pixel mode)")
        print("3. Custom Analysis (Choose pixel mode)")
        print("4. Best Pixel Analysis (Selected pixels only - RECOMMENDED)")
        print("5. View Available Data")
        print("6. Return to Main Menu")
        print()
        print("Pixel Selection Info:")
        print("- All pixels: 2/13/197 (Athabasca/Haig/Coropuna)")
        print("- Selected pixels: 2/1/1 (Optimized for analysis)")
        print()
    
    def check_data_availability(self) -> Dict[str, Any]:
        """
        Check what glacier data is available for comparative analysis.
        
        Returns:
            Dictionary containing data availability information
        """
        logger.info("Checking data availability for comparative analysis...")
        
        results = self.data_analyzer.discover_latest_results()
        
        availability = {
            'total_glaciers_found': len(results),
            'glaciers_available': list(results.keys()),
            'glaciers_missing': [],
            'analysis_timestamps': {glacier: info['timestamp'] for glacier, info in results.items()},
            'ready_for_analysis': len(results) >= 2  # Need at least 2 glaciers for comparison
        }
        
        # Check for expected glaciers
        expected_glaciers = ['athabasca', 'haig', 'coropuna']
        availability['glaciers_missing'] = [g for g in expected_glaciers if g not in results]
        
        return availability
    
    def display_data_availability(self) -> bool:
        """
        Display data availability status and return whether analysis can proceed.
        
        Returns:
            True if sufficient data is available for analysis
        """
        availability = self.check_data_availability()
        
        print("\n" + "=" * 50)
        print("Data Availability Status")
        print("=" * 50)
        
        print(f"Glaciers with analysis results: {availability['total_glaciers_found']}")
        
        if availability['glaciers_available']:
            print("\n[OK] Available for comparison:")
            for glacier in availability['glaciers_available']:
                timestamp = availability['analysis_timestamps'][glacier]
                print(f"   • {glacier.title()} Glacier (analyzed: {timestamp})")
        
        if availability['glaciers_missing']:
            print("\n[MISSING] Missing analysis results:")
            for glacier in availability['glaciers_missing']:
                print(f"   • {glacier.title()} Glacier")
            print("\n[TIP] Run individual glacier analyses first to include them in comparison")
        
        print(f"\nReady for comparative analysis: {'[OK] Yes' if availability['ready_for_analysis'] else '[FAIL] No (need >=2 glaciers)'}")
        
        return availability['ready_for_analysis']
    
    def run_quick_overview(self) -> None:
        """Run a quick overview analysis with essential plots and statistics."""
        logger.info("Running quick overview analysis...")
        
        print("\n[RUNNING] Quick Overview Analysis...")
        
        # Load and aggregate data
        print("   [DATA] Aggregating glacier data...")
        self.aggregated_data = self.data_analyzer.aggregate_glacier_data()
        
        if self.aggregated_data.empty:
            print("[ERROR] No data available for analysis")
            return
        
        # Create output directory
        self.output_directory = self.data_analyzer.create_output_directory()
        print(f"   [OUTPUT] Created output directory: {self.output_directory}")
        
        # Generate summary statistics
        print("   [STATS] Calculating summary statistics...")
        summary = self.data_analyzer.get_summary_statistics()
        
        # Display key findings
        self._display_key_findings(summary)
        
        # Generate essential plots (subset of all plots)
        print("   🎨 Generating key visualizations...")
        essential_plots = [
            'plot_method_performance_matrix',
            'plot_regional_comparison_boxplots', 
            'plot_cross_glacier_scatterplot_matrix',
            'plot_statistical_confidence_dashboard'
        ]
        
        for plot_method in essential_plots:
            try:
                method = getattr(self.visualizer, plot_method)
                method(self.aggregated_data, self.output_directory)
            except Exception as e:
                logger.warning(f"Failed to generate {plot_method}: {e}")
        
        # Export data
        self.data_analyzer.export_aggregated_data(self.output_directory)
        
        print(f"\n[SUCCESS] Quick overview completed!")
        print(f"[OUTPUT] Results saved to: {self.output_directory}")
        
    def run_full_statistical_analysis(self) -> None:
        """Run comprehensive statistical analysis with all tests and visualizations."""
        logger.info("Running full statistical analysis...")
        
        print("\n[RUNNING] Full Statistical Analysis...")
        
        # Load and aggregate data
        print("   [DATA] Aggregating glacier data...")
        self.aggregated_data = self.data_analyzer.aggregate_glacier_data()
        
        if self.aggregated_data.empty:
            print("[ERROR] No data available for analysis")
            return
        
        # Create output directory
        self.output_directory = self.data_analyzer.create_output_directory()
        print(f"   [OUTPUT] Created output directory: {self.output_directory}")
        
        # Run comprehensive statistical tests
        print("   [TESTS] Running statistical tests...")
        statistical_results = self.statistician.run_comprehensive_analysis(self.aggregated_data)
        
        # Generate all visualizations
        print("   🎨 Generating all visualizations (7 plot types)...")
        try:
            self.visualizer.generate_all_plots(self.aggregated_data, self.output_directory)
            print("   [OK] All plots generated successfully")
        except Exception as e:
            logger.error(f"Error generating plots: {e}")
            print(f"   [FAIL] Some plots failed to generate: {e}")
        
        # Export all results
        print("   [EXPORT] Exporting results...")
        self.data_analyzer.export_aggregated_data(self.output_directory)
        self.statistician.export_statistical_results(self.output_directory)
        
        # Generate summary report
        self._generate_summary_report(statistical_results)
        
        print(f"\n[SUCCESS] Full analysis completed!")
        print(f"[OUTPUT] Results saved to: {self.output_directory}")
        print(f"[STATS] Generated 7 visualization types (including maps)")
        print(f"[STATS] Completed 6 statistical test categories")
        
    def run_custom_analysis(self) -> None:
        """Run custom analysis with user-selected components."""
        logger.info("Running custom analysis...")
        
        print("\n[CONFIG] Custom Analysis Configuration")
        print("-" * 40)
        
        # Load data first
        self.aggregated_data = self.data_analyzer.aggregate_glacier_data()
        if self.aggregated_data.empty:
            print("[ERROR] No data available for analysis")
            return
        
        self.output_directory = self.data_analyzer.create_output_directory()
        
        # Statistical tests menu
        print("\nSelect Statistical Tests:")
        stat_options = {
            '1': ('Method Performance ANOVA', 'method_performance_anova'),
            '2': ('Regional Comparisons', 'regional_comparison_tests'),
            '3': ('Environmental Correlations', 'environmental_correlation_analysis'),
            '4': ('Method Consistency Analysis', 'method_consistency_analysis'),
            '5': ('Bootstrap Confidence Intervals', 'bootstrap_confidence_intervals'),
            'A': ('All Statistical Tests', 'all_stats')
        }
        
        for key, (name, _) in stat_options.items():
            print(f"   {key}. {name}")
        
        stat_choice = input("\nEnter statistical test choice (1-5, A, or skip): ").strip().upper()
        
        # Visualization menu
        print("\nSelect Visualizations:")
        viz_options = {
            '1': ('Method Performance Matrix', 'plot_method_performance_matrix'),
            '2': ('AWS vs MODIS Scatterplots (3x3)', 'plot_cross_glacier_scatterplot_matrix'),
            '3': ('Glacier-Specific Method Analysis', 'plot_regional_comparison_boxplots'),
            '4': ('Sample Size Analysis', 'plot_sample_size_vs_performance'),
            '5': ('Bias Radar Chart', 'plot_bias_comparison_radar'),
            '6': ('Combined Statistical Dashboard', 'plot_statistical_confidence_dashboard'),
            '7': ('Multi-Glacier Mapping Suite', 'plot_glacier_maps'),
            'A': ('All Visualizations', 'all_plots')
        }
        
        for key, (name, _) in viz_options.items():
            print(f"   {key}. {name}")
        
        viz_choice = input("\nEnter visualization choice (1-7, A, or skip): ").strip().upper()
        
        # Execute selected analyses
        print(f"\n[RUNNING] Custom analysis...")
        
        # Run statistical tests
        if stat_choice and stat_choice != 'SKIP':
            print("   [TESTS] Running selected statistical tests...")
            if stat_choice == 'A':
                self.statistician.run_comprehensive_analysis(self.aggregated_data)
            else:
                method_map = {
                    '1': self.statistician.method_performance_anova,
                    '2': self.statistician.regional_comparison_tests,
                    '3': self.statistician.environmental_correlation_analysis,
                    '4': self.statistician.method_consistency_analysis,
                    '5': self.statistician.bootstrap_confidence_intervals
                }
                if stat_choice in method_map:
                    method_map[stat_choice](self.aggregated_data)
        
        # Generate visualizations
        if viz_choice and viz_choice != 'SKIP':
            print("   🎨 Generating selected visualizations...")
            if viz_choice == 'A':
                self.visualizer.generate_all_plots(self.aggregated_data, self.output_directory)
            else:
                viz_method_name = viz_options.get(viz_choice, [None, None])[1]
                if viz_method_name and hasattr(self.visualizer, viz_method_name):
                    method = getattr(self.visualizer, viz_method_name)
                    method(self.aggregated_data, self.output_directory)
        
        # Export results
        self.data_analyzer.export_aggregated_data(self.output_directory)
        if stat_choice and stat_choice != 'SKIP':
            self.statistician.export_statistical_results(self.output_directory)
        
        print(f"\n[SUCCESS] Custom analysis completed!")
        print(f"[OUTPUT] Results saved to: {self.output_directory}")
        
    def _display_key_findings(self, summary: Dict[str, Any]) -> None:
        """Display key findings from the analysis."""
        print("\n📋 Key Findings:")
        print("-" * 20)
        
        print(f"   • Analyzed {summary['total_glaciers']} glaciers with {summary['total_methods']} MODIS methods")
        print(f"   • Total observations: {summary['total_observations']:,}")
        
        # Geographic span
        geo = summary['geographic_range']
        print(f"   • Geographic span: {geo['min_latitude']:.1f}° to {geo['max_latitude']:.1f}° latitude")
        print(f"   • Elevation range: {geo['min_elevation']:.0f}m to {geo['max_elevation']:.0f}m")
        
        # Performance ranges
        perf = summary['performance_ranges']
        print(f"   • Correlation range: {perf['correlation_range'][0]:.3f} to {perf['correlation_range'][1]:.3f}")
        print(f"   • RMSE range: {perf['rmse_range'][0]:.3f} to {perf['rmse_range'][1]:.3f}")
        
        # Best performing methods
        best_corr = summary['best_performing_method']['by_correlation']
        best_rmse = summary['best_performing_method']['by_rmse']
        print(f"   • Best correlation: {best_corr['method']} on {best_corr['glacier_id']} (r={best_corr['r']:.3f})")
        print(f"   • Best RMSE: {best_rmse['method']} on {best_rmse['glacier_id']} (RMSE={best_rmse['rmse']:.3f})")
    
    def _generate_summary_report(self, statistical_results: Dict[str, Any]) -> None:
        """Generate a comprehensive summary report."""
        if not self.output_directory:
            return
        
        report_file = self.output_directory / "results" / "comparative_analysis_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("Multi-Glacier Comparative Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Summary statistics
            summary = self.data_analyzer.get_summary_statistics()
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Glaciers: {summary['total_glaciers']}\n")
            f.write(f"Total Methods: {summary['total_methods']}\n")
            f.write(f"Total Observations: {summary['total_observations']:,}\n\n")
            
            # Geographic coverage
            geo = summary['geographic_range']
            f.write("GEOGRAPHIC COVERAGE\n")
            f.write("-" * 20 + "\n")
            f.write(f"Latitude Range: {geo['min_latitude']:.2f}° to {geo['max_latitude']:.2f}°\n")
            f.write(f"Elevation Range: {geo['min_elevation']:.0f}m to {geo['max_elevation']:.0f}m\n\n")
            
            # Statistical test summaries
            if 'method_anova' in statistical_results:
                f.write("STATISTICAL TEST RESULTS\n")
                f.write("-" * 20 + "\n")
                f.write("Method Performance ANOVA:\n")
                for metric, results in statistical_results['method_anova'].items():
                    significance = "significant" if results['significant'] else "not significant"
                    f.write(f"  {metric}: F={results['f_statistic']:.3f}, p={results['p_value']:.3f} ({significance})\n")
                f.write("\n")
            
            # Method consistency rankings
            if 'method_consistency' in statistical_results and 'consistency_rankings' in statistical_results['method_consistency']:
                f.write("METHOD CONSISTENCY RANKINGS\n")
                f.write("-" * 20 + "\n")
                for metric, rankings in statistical_results['method_consistency']['consistency_rankings'].items():
                    f.write(f"{metric.upper()} Consistency (most to least consistent):\n")
                    for i, (method, cv) in enumerate(rankings, 1):
                        f.write(f"  {i}. {method} (CV={cv:.3f})\n")
                    f.write("\n")
            
            f.write("GENERATED OUTPUTS\n")
            f.write("-" * 20 + "\n")
            f.write("Visualizations:\n")
            f.write("  • Method performance matrix heatmap\n")
            f.write("  • Cross-glacier scatterplot matrix\n")
            f.write("  • Regional comparison boxplots\n")
            f.write("  • Sample size vs performance analysis\n")
            f.write("  • Bias comparison radar chart\n")
            f.write("  • Environmental factor analysis\n")
            f.write("  • Temporal coverage comparison\n")
            f.write("  • Statistical confidence dashboard\n\n")
            
            f.write("Data Files:\n")
            f.write("  • comparative_summary.csv\n")
            f.write("  • statistical_test_results.csv\n")
            f.write("  • method_consistency_analysis.csv\n")
        
        logger.info(f"Summary report generated: {report_file}")
    
    def get_pixel_selection_mode(self, analysis_type: str) -> bool:
        """Get pixel selection mode for comparative analysis."""
        print(f"\nPixel Selection Mode for {analysis_type}:")
        print("-" * 50)
        print("1. All Pixels (2 for Athabasca, 13 for Haig, 197 for Coropuna)")
        print("2. Selected Best Pixels (2 for Athabasca, 1 for Haig, 1 for Coropuna - RECOMMENDED)")
        print()
        print("Selected pixels are optimized based on:")
        print("- Distance to AWS station (60% weight)")
        print("- Glacier fraction coverage (40% weight)")
        print()
        
        while True:
            choice = input("Enter your choice (1, 2): ").strip()
            if choice == '1':
                return False  # Use all pixels
            elif choice == '2':
                return True   # Use selected pixels
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    def run_best_pixel_analysis(self) -> None:
        """Run comprehensive comparative analysis using best selected pixels closest to AWS stations."""
        print("\n" + "=" * 60)
        print("Best Pixel Analysis - Full Comparative Analysis with Selected Pixels")
        print("=" * 60)
        print()
        print("This analysis uses the optimal MODIS pixels for each glacier:")
        print("- Athabasca: All pixels (only 2 available)")
        print("- Haig & Coropuna: Best performing pixel closest to AWS station")
        print("- Selection criteria: Distance to AWS (60%) + Glacier fraction (40%)")
        print()
        
        try:
            # Show pixel selection details for each glacier
            print("Pixel Selection Details:")
            print("-" * 30)
            
            selected_data = {}
            for glacier_id in ['athabasca', 'haig', 'coropuna']:
                print(f"\n{glacier_id.upper()} GLACIER:")
                
                # Load data in analysis mode (selected pixels)
                modis_data = self.mapper.load_original_modis_data(glacier_id, analysis_mode=True)
                if modis_data is not None and not modis_data.empty:
                    selected_data[glacier_id] = modis_data
                    print(f"[OK] {len(modis_data)} pixel(s) selected for analysis")
                    
                    # Show pixel details
                    for idx, row in modis_data.iterrows():
                        pixel_id = row.get('pixel_id', 'unknown')
                        lat, lon = row['latitude'], row['longitude']
                        glacier_frac = row.get('glacier_fraction', 'N/A')
                        if glacier_frac != 'N/A':
                            print(f"  - Pixel {pixel_id}: {lat:.4f}°N, {lon:.4f}°E (glacier_frac: {glacier_frac:.3f})")
                        else:
                            print(f"  - Pixel {pixel_id}: {lat:.4f}°N, {lon:.4f}°E")
                else:
                    print("[FAIL] No data available")
            
            if not selected_data:
                print("\n[FAIL] No pixel data available for analysis.")
                return
            
            print(f"\n[OK] Pixel selection completed!")
            print(f"Ready to proceed with FULL COMPARATIVE ANALYSIS using selected pixels...")
            
            # NOW RUN THE ACTUAL COMPARATIVE ANALYSIS
            print("\n" + "=" * 60)
            print("RUNNING FULL COMPARATIVE ANALYSIS WITH SELECTED PIXELS")
            print("=" * 60)
            
            # Load and aggregate data using selected pixels
            print("   [DATA] Aggregating glacier data with selected pixels...")
            
            # Use the new pixel-aware aggregation method
            self.aggregated_data = self.data_analyzer.aggregate_glacier_data_with_pixel_selection()
            
            if self.aggregated_data.empty:
                print("[ERROR] No data available for analysis after pixel selection")
                return
            
            # Create output directory with subdirectories
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.output_directory = Path(self.outputs_dir) / f"best_pixel_comparative_analysis_{timestamp}"
            self.output_directory.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (self.output_directory / "plots").mkdir(parents=True, exist_ok=True)
            (self.output_directory / "results").mkdir(parents=True, exist_ok=True)
            
            print(f"   [OUTPUT] Created output directory: {self.output_directory}")
            
            # Generate summary statistics
            print("   [STATS] Calculating summary statistics...")
            summary = self.data_analyzer.get_summary_statistics()
            
            # Display key findings
            self._display_key_findings(summary)
            
            # Run comprehensive statistical tests
            print("   [TESTS] Running statistical tests...")
            statistical_results = self.statistician.run_comprehensive_analysis(self.aggregated_data)
            
            # Generate all visualizations
            print("   [PLOTS] Generating all visualizations (7 plot types)...")
            try:
                self.visualizer.generate_all_plots(self.aggregated_data, self.output_directory)
                print("   [OK] All plots generated successfully")
            except Exception as e:
                logger.error(f"Error generating plots: {e}")
                print(f"   [FAIL] Some plots failed to generate: {e}")
            
            # Generate pixel selection maps
            print("   [MAPS] Generating pixel selection maps...")
            self.mapper.generate_all_maps(self.output_directory)
            self._create_pixel_selection_summary(selected_data)
            
            # Export all results
            print("   [EXPORT] Exporting results...")
            self.data_analyzer.export_aggregated_data(self.output_directory)
            self.statistician.export_statistical_results(self.output_directory)
            
            # Generate summary report
            self._generate_summary_report(statistical_results)
            
            print(f"\n[SUCCESS] COMPLETE COMPARATIVE ANALYSIS WITH SELECTED PIXELS FINISHED!")
            print("=" * 60)
            print(f"Results saved to: {self.output_directory}")
            print(f"Generated 7 visualization types + pixel selection maps")
            print(f"Completed 6 statistical test categories")
            print()
            print("Key findings:")
            print(f"- Athabasca: Using all {len(selected_data.get('athabasca', []))} available pixels")
            if 'haig' in selected_data:
                print(f"- Haig: Selected {len(selected_data['haig'])} best pixel from 13 candidates")
            if 'coropuna' in selected_data:
                print(f"- Coropuna: Selected {len(selected_data['coropuna'])} best pixel from 197 candidates")
            print(f"- Full statistical analysis completed with selected pixels only")
            
        except Exception as e:
            logger.error(f"Error in best pixel analysis: {e}")
            print(f"\n[ERROR] Analysis failed: {e}")
    
    def _create_pixel_selection_summary(self, selected_data: Dict[str, pd.DataFrame]) -> None:
        """Create a summary visualization of pixel selection."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            for idx, glacier_id in enumerate(['athabasca', 'haig', 'coropuna']):
                ax = axes[idx]
                
                if glacier_id in selected_data:
                    # Load glacier mask first
                    mask_gdf = self.mapper.load_glacier_mask(glacier_id)
                    
                    # Load all pixels for comparison
                    all_pixels = self.mapper.load_original_modis_data(glacier_id, analysis_mode=False)
                    selected_pixels = selected_data[glacier_id]
                    
                    # Plot glacier mask first (background) - only add to legend once
                    if mask_gdf is not None and not mask_gdf.empty:
                        mask_gdf.plot(ax=ax, facecolor='lightblue', alpha=0.5,
                                     edgecolor='darkblue', linewidth=2, zorder=1)
                        print(f"  Plotted glacier mask for {glacier_id}: {len(mask_gdf)} features")
                    else:
                        print(f"  No glacier mask found for {glacier_id}")
                    
                    if all_pixels is not None and not all_pixels.empty:
                        # Plot all pixels in light gray
                        ax.scatter(all_pixels['longitude'], all_pixels['latitude'], 
                                 c='lightgray', s=25, alpha=0.7, zorder=2,
                                 label=f'All pixels (n={len(all_pixels)})' if idx == 0 else "")
                        
                        # Highlight selected pixels in red
                        ax.scatter(selected_pixels['longitude'], selected_pixels['latitude'],
                                 c='red', s=100, marker='*', edgecolors='darkred', linewidth=2,
                                 zorder=5, label=f'Selected pixels (n={len(selected_pixels)})' if idx == 0 else "")
                        
                        # Add AWS station if available
                        aws_coords = self.mapper.get_aws_coordinates(glacier_id)
                        if aws_coords:
                            for station_id, coords in aws_coords.items():
                                if coords['lat'] is not None and coords['lon'] is not None:
                                    ax.scatter(coords['lon'], coords['lat'], 
                                             c='blue', s=120, marker='^', 
                                             edgecolors='darkblue', linewidth=2, zorder=5,
                                             label='AWS Station' if idx == 0 else "")
                
                ax.set_title(f"{glacier_id.title()} Glacier", fontweight='bold')
                ax.set_xlabel('Longitude (°)')
                ax.set_ylabel('Latitude (°)')
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal', adjustable='box')
            
            # Create unified legend for all subplots
            from matplotlib.patches import Rectangle
            import matplotlib.lines as mlines
            
            legend_elements = [
                Rectangle((0, 0), 1, 1, facecolor='lightblue', edgecolor='darkblue', alpha=0.5, label='Glacier Boundary'),
                mlines.Line2D([], [], color='lightgray', marker='o', linestyle='None', 
                             markersize=6, alpha=0.7, label='All MODIS Pixels'),
                mlines.Line2D([], [], color='red', marker='*', linestyle='None', 
                             markersize=10, markeredgecolor='darkred', markeredgewidth=1, label='Selected Best Pixels'),
                mlines.Line2D([], [], color='blue', marker='^', linestyle='None', 
                             markersize=8, markeredgecolor='darkblue', markeredgewidth=1, label='AWS Weather Station')
            ]
            
            # Add legend to the figure (not individual subplots)
            fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
                      fontsize=10, frameon=True, fancybox=True, shadow=True,
                      bbox_to_anchor=(0.5, 0.02))
            
            plt.suptitle('Pixel Selection Summary - Best Pixels for Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Make room for legend at bottom
            
            # Save the plot
            output_file = self.output_directory / "plots" / "pixel_selection_summary.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Pixel selection summary saved: {output_file}")
            
        except Exception as e:
            logger.warning(f"Could not create pixel selection summary: {e}")
    
    def run_interactive_session(self) -> None:
        """Run the interactive comparative analysis session."""
        while True:
            self.display_welcome_menu()
            
            # Check data availability
            if not self.display_data_availability():
                print("\n[ERROR] Insufficient data for comparative analysis.")
                print("Please run individual glacier analyses first.")
                input("\nPress Enter to return to main menu...")
                return
            
            self.display_analysis_options()
            
            choice = input("Enter your choice (1-6): ").strip()
            
            if choice == '1':
                # Quick Overview with pixel selection
                use_selected_pixels = self.get_pixel_selection_mode("Quick Overview")
                if use_selected_pixels:
                    print("\n[INFO] Quick Overview will use selected best pixels")
                    # TODO: Implement pixel-aware quick overview
                    self.run_best_pixel_analysis()  # Temporary - use best pixel analysis
                else:
                    print("\n[INFO] Quick Overview will use all available pixels")
                    self.run_quick_overview()
                input("\nPress Enter to continue...")
                
            elif choice == '2':
                # Full Statistical Analysis with pixel selection
                use_selected_pixels = self.get_pixel_selection_mode("Full Statistical Analysis")
                if use_selected_pixels:
                    print("\n[INFO] Full Statistical Analysis will use selected best pixels")
                    # TODO: Implement pixel-aware full analysis
                    self.run_best_pixel_analysis()  # Temporary - use best pixel analysis
                else:
                    print("\n[INFO] Full Statistical Analysis will use all available pixels")
                    self.run_full_statistical_analysis()
                input("\nPress Enter to continue...")
                
            elif choice == '3':
                # Custom Analysis with pixel selection
                use_selected_pixels = self.get_pixel_selection_mode("Custom Analysis")
                if use_selected_pixels:
                    print("\n[INFO] Custom Analysis will use selected best pixels")
                    # TODO: Implement pixel-aware custom analysis
                    self.run_best_pixel_analysis()  # Temporary - use best pixel analysis
                else:
                    print("\n[INFO] Custom Analysis will use all available pixels")
                    self.run_custom_analysis()
                input("\nPress Enter to continue...")
                
            elif choice == '4':
                self.run_best_pixel_analysis()
                input("\nPress Enter to continue...")
                
            elif choice == '5':
                self.display_data_availability()
                input("\nPress Enter to continue...")
                
            elif choice == '6':
                break
                
            else:
                print("Invalid choice. Please try again.")
                input("Press Enter to continue...")


def main():
    """Main function for testing the comparative analysis interface."""
    interface = ComparativeAnalysisInterface()
    interface.run_interactive_session()


if __name__ == "__main__":
    main()