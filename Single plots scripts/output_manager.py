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
        """Create the standardized directory structure."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.plots_dir = self.output_dir / "plots"
        self.results_dir = self.output_dir / "results"
        
        self.plots_dir.mkdir(exist_ok=True)
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Created directory structure: {self.output_dir}")
    
    def get_plot_path(self, filename: str) -> Path:
        """Get full path for a plot file.
        
        Args:
            filename: Plot filename (e.g., 'scatterplot_matrix.png')
            
        Returns:
            Full path to plot file
        """
        return self.plots_dir / filename
    
    def get_results_path(self, filename: str) -> Path:
        """Get full path for a results file.
        
        Args:
            filename: Results filename (e.g., 'statistics.csv')
            
        Returns:
            Full path to results file
        """
        return self.results_dir / filename
    
    def save_summary(self, summary_data: Dict[str, Any], filename: str = "summary.txt"):
        """Save analysis summary to text file.
        
        Args:
            summary_data: Dictionary containing summary information
            filename: Summary filename
        """
        summary_path = self.get_results_path(filename)
        
        with open(summary_path, 'w') as f:
            f.write(f"Analysis Summary: {self.analysis_name}\n")
            f.write("=" * 60 + "\n\n")
            
            # Basic metadata
            f.write(f"Generated: {summary_data.get('timestamp', 'N/A')}\n")
            f.write(f"Analysis Type: {summary_data.get('analysis_type', self.analysis_name)}\n")
            f.write(f"Output Directory: {self.output_dir}\n\n")
            
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
            
            # Statistical results
            if 'statistics' in summary_data:
                f.write("Statistical Summary:\n")
                f.write("-" * 20 + "\n")
                stats = summary_data['statistics']
                for key, value in stats.items():
                    if isinstance(value, dict):
                        f.write(f"{key}:\n")
                        for sub_key, sub_value in value.items():
                            if isinstance(sub_value, float):
                                f.write(f"  {sub_key}: {sub_value:.4f}\n")
                            else:
                                f.write(f"  {sub_key}: {sub_value}\n")
                    elif isinstance(value, float):
                        f.write(f"{key}: {value:.4f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n")
            
            # Files generated
            f.write("Output Files:\n")
            f.write("-" * 20 + "\n")
            f.write("Plots:\n")
            for plot_file in self.plots_dir.glob("*"):
                f.write(f"  - {plot_file.name}\n")
            f.write("Results:\n")
            for result_file in self.results_dir.glob("*"):
                f.write(f"  - {result_file.name}\n")
        
        logger.info(f"Summary saved to: {summary_path}")
    
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
            f.write("├── plots/          # Generated visualizations\n")
            f.write("├── results/        # Statistical outputs and summaries\n")
            f.write("└── README.md       # This documentation\n")
            f.write("```\n\n")
            
            # Key findings
            f.write("## Key Findings\n\n")
            for i, finding in enumerate(key_findings, 1):
                f.write(f"{i}. {finding}\n")
            f.write("\n")
            
            # Files generated
            f.write("## Generated Files\n\n")
            f.write("### Plots\n")
            for plot_file in sorted(self.plots_dir.glob("*")):
                f.write(f"- `{plot_file.name}`\n")
            f.write("\n### Results\n")
            for result_file in sorted(self.results_dir.glob("*")):
                f.write(f"- `{result_file.name}`\n")
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