#!/usr/bin/env python3
"""
Output Manager Utility

Standardized output directory and file management for all analysis scripts.
Provides consistent structure for plots, results, summaries, and metadata.

Author: Analysis System
Date: 2025-08-02
"""

# ============================================================================
# IMPORTS
# ============================================================================

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from typing import Any, Dict, List, Optional

# ============================================================================
# LOGGING SETUP
# ============================================================================

logger = logging.getLogger(__name__)

# ============================================================================
# OUTPUT MANAGER CLASS
# ============================================================================

class OutputManager:
    """Manages standardized output structure for analysis scripts."""
    
    def __init__(self, analysis_name: str, base_dir: str = "outputs"):
        """Initialize output manager with analysis-specific structure.
        
        Args:
            analysis_name: Name of the analysis (e.g., 'aws_vs_modis_scatterplot')
            base_dir: Base directory for all outputs
        """
        self.analysis_name = analysis_name
        self.base_dir = Path(base_dir)
        
        # Create timestamped directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_dir / f"{analysis_name}_{timestamp}"
        
        # Create directory structure
        self._create_directory_structure()
        
        # Initialize metadata
        self.metadata = {
            'analysis_name': analysis_name,
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat(),
            'output_directory': str(self.output_dir)
        }
        
        logger.info(f"Initialized OutputManager for {analysis_name}. Output: {self.output_dir}")
    
    def _create_directory_structure(self):
        """Create the simplified flat directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory: {self.output_dir}")
    
    def get_plot_path(self, filename: str) -> Path:
        """Get full path for a plot file.
        
        Args:
            filename: Plot filename (e.g., 'scatterplot_matrix.png')
            
        Returns:
            Full path to plot file in main output directory
        """
        return self.output_dir / filename
    
    def get_results_path(self, filename: str) -> Path:
        """Get full path for a results file.
        
        Args:
            filename: Results filename (e.g., 'statistics.csv')
            
        Returns:
            Full path to results file in main output directory
        """
        return self.output_dir / filename
    
    def save_summary(self, summary_data: Dict[str, Any], filename: str = "summary.txt"):
        """Save analysis summary to text file.
        
        Args:
            summary_data: Dictionary containing summary information
            filename: Summary filename
        """
        summary_path = self.get_results_path(filename)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"Comprehensive Analysis Report: {self.analysis_name.replace('_', ' ').title()}\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic metadata
            f.write(f"Generated: {summary_data.get('timestamp', 'N/A')}\n")
            f.write(f"Analysis Type: {summary_data.get('analysis_type', self.analysis_name)}\n")
            f.write(f"Glacier: {summary_data.get('glacier', 'N/A')}\n")
            f.write(f"Output Directory: {self.output_dir.name}\n")
            f.write(f"Analysis Period: {summary_data.get('data_info', {}).get('analysis_period', 'N/A')}\n\n")
            
            # Configuration
            if 'configuration' in summary_data:
                f.write("Configuration:\n")
                f.write("-" * 20 + "\n")
                config = summary_data['configuration']
                for key, value in config.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items():
                            f.write(f"  {sub_key}: {sub_value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Data processing info
            if 'data_info' in summary_data:
                f.write("Data Processing:\n")
                f.write("-" * 20 + "\n")
                data_info = summary_data['data_info']
                for key, value in data_info.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Key results
            if 'key_results' in summary_data:
                f.write("Key Results:\n")
                f.write("-" * 20 + "\n")
                results = summary_data['key_results']
                for key, value in results.items():
                    if isinstance(value, (int, float)):
                        if isinstance(value, float):
                            f.write(f"{key}: {value:.4f}\n")
                        else:
                            f.write(f"{key}: {value}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Detailed statistical results
            if 'detailed_statistics' in summary_data:
                f.write("Detailed Trend Statistics:\n")
                f.write("-" * 40 + "\n")
                stats = summary_data['detailed_statistics']
                for key, s in stats.items():
                    nice = key.replace('_annual', '').replace('modis_', 'MODIS ').replace('_', ' ').title()
                    ci_lo, ci_hi = s.get('slope_confidence_interval', (float('nan'), float('nan')))
                    f.write(f"{nice}:\n")
                    f.write(f"  Observations: {s.get('n_observations', 0)}, Duration: {s.get('duration_years', 0):.1f} years\n")
                    f.write(f"  Kendall tau: {s.get('kendall_tau', 0):.3f}, p-value: {s.get('p_value', 1):.3f}")
                    f.write(f"{' (prewhitened)' if s.get('prewhitened') else ''}\n")
                    f.write(f"  Sen slope: {s.get('sen_slope_per_year', 0):+.4f} per year\n")
                    if not (np.isnan(ci_lo) or np.isnan(ci_hi)):
                        f.write(f"  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]\n")
                    f.write(f"  Trend: {s.get('mann_kendall_trend', 'no trend')} ")
                    f.write(f"({'significant' if s.get('statistical_significance') else 'not significant'})\n\n")
                f.write("\n")
            
            # Sensitivity analysis
            if 'sensitivity_notes' in summary_data and summary_data['sensitivity_notes']:
                f.write("Sensitivity Analysis:\n")
                f.write("-" * 30 + "\n")
                for note in summary_data['sensitivity_notes']:
                    f.write(f"• {note}\n")
                f.write("\n")
            
            # Additional sections from summary_data
            if 'altitude_analysis' in summary_data:
                f.write("Altitude Band Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(summary_data['altitude_analysis'])
                f.write("\n")
            
            # Files generated
            f.write("Generated Output Files:\n")
            f.write("-" * 30 + "\n")
            plot_files = sorted([f for f in self.output_dir.glob("*.png")])
            other_files = sorted([f for f in self.output_dir.glob("*") if f.is_file() and f.name != filename and not f.name.endswith('.png')])
            
            if plot_files:
                f.write("Visualizations:\n")
                for plot_file in plot_files:
                    f.write(f"  • {plot_file.name}\n")
                f.write("\n")
            
            if other_files:
                f.write("Other files:\n")
                for other_file in other_files:
                    f.write(f"  • {other_file.name}\n")
                f.write("\n")
            
            f.write(f"\nReport generated on: {summary_data.get('timestamp', 'N/A')}\n")
            f.write(f"Total analysis duration: {summary_data.get('data_info', {}).get('analysis_period', 'N/A')}\n")
        
        logger.info(f"Comprehensive summary saved to: {summary_path}")
    
    def save_readme(self, analysis_description: str, key_findings: List[str], 
                   additional_info: Optional[Dict[str, str]] = None):
        """Save README.md file with analysis documentation.
        
        Args:
            analysis_description: Description of the analysis
            key_findings: List of key findings/results
            additional_info: Additional information to include
        """
        readme_path = self.output_dir / "README.md"
        
        with open(readme_path, 'w') as f:
            f.write(f"# {self.analysis_name.replace('_', ' ').title()} Analysis\n\n")
            
            # Analysis description
            f.write("## Analysis Description\n\n")
            f.write(f"{analysis_description}\n\n")
            
            # Metadata
            f.write("## Analysis Metadata\n\n")
            f.write(f"- **Generated**: {self.metadata['created_at']}\n")
            f.write(f"- **Analysis Type**: {self.analysis_name}\n")
            f.write(f"- **Output Directory**: `{self.output_dir.name}`\n\n")
            
            # Directory structure
            f.write("## Directory Structure\n\n")
            f.write("```\n")
            f.write(f"{self.output_dir.name}/\n")
            f.write("├── *.png           # Generated visualizations\n")
            f.write("└── summary.txt     # Comprehensive analysis results\n")
            f.write("```\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            for i, finding in enumerate(key_findings, 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")
            
            # Files generated
            f.write("## Generated Files\n\n")
            for output_file in sorted(self.output_dir.glob("*")):
                if output_file.is_file() and output_file.name != "README.md":
                    f.write(f"- `{output_file.name}`\n")
            f.write("\n")
            
            # Additional information
            if additional_info:
                f.write("## Additional Information\n\n")
                for key, value in additional_info.items():
                    f.write(f"**{key}**: {value}\n\n")
        
        logger.info(f"README saved to: {readme_path}")
    
    def log_file_saved(self, file_path: Path, file_type: str = "file"):
        """Log when a file is saved to the output structure.
        
        Args:
            file_path: Path to the saved file
            file_type: Type of file (plot, result, etc.)
        """
        relative_path = file_path.relative_to(self.output_dir)
        logger.info(f"{file_type.title()} saved: {relative_path}")