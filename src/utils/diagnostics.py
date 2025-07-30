#!/usr/bin/env python3
"""
Diagnostic Utilities for MODIS Albedo Analysis Framework

This module provides diagnostic tools to help troubleshoot issues and 
monitor the health of the analysis framework. All functions are designed
to be non-invasive and provide useful information for debugging.
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime
import psutil  # For system monitoring

logger = logging.getLogger(__name__)


def diagnose_system_environment() -> Dict[str, Any]:
    """
    Diagnose system environment and dependencies.
    
    Returns:
        Dict with system information:
        - python_version: Python version information
        - platform: Operating system information
        - memory: Available system memory
        - disk_space: Available disk space
        - key_dependencies: Status of important Python packages
        
    Example:
        >>> info = diagnose_system_environment()
        >>> print(f"Python: {info['python_version']}")
        >>> print(f"Memory: {info['memory']['available_gb']:.1f} GB available")
    """
    try:
        import platform
        
        # Basic system info
        system_info = {
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'platform': platform.system(),
            'platform_version': platform.release(),
            'architecture': platform.machine(),
            'processor': platform.processor()
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        system_info['memory'] = {
            'total_gb': memory.total / (1024**3),
            'available_gb': memory.available / (1024**3),
            'used_gb': memory.used / (1024**3),
            'percent_used': memory.percent
        }
        
        # Disk space for current working directory
        disk = psutil.disk_usage('.')
        system_info['disk_space'] = {
            'total_gb': disk.total / (1024**3),
            'free_gb': disk.free / (1024**3),
            'used_gb': disk.used / (1024**3),
            'percent_used': (disk.used / disk.total) * 100
        }
        
        # Check key dependencies
        key_packages = ['pandas', 'numpy', 'matplotlib', 'seaborn', 'geopandas', 'scipy']
        dependencies = {}
        
        for package in key_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'Unknown')
                dependencies[package] = {'installed': True, 'version': version}
            except ImportError:
                dependencies[package] = {'installed': False, 'version': None}
        
        system_info['key_dependencies'] = dependencies
        
        # Python path information
        system_info['python_path'] = sys.path[:5]  # First 5 entries
        system_info['current_directory'] = os.getcwd()
        
        logger.debug("System environment diagnosis completed successfully")
        return system_info
        
    except Exception as e:
        logger.error(f"Error diagnosing system environment: {e}")
        return {'error': str(e)}


def diagnose_data_availability(base_dir: str = ".") -> Dict[str, Any]:
    """
    Diagnose data file availability across all glacier projects.
    
    Args:
        base_dir: Base directory to search from
        
    Returns:
        Dict with data availability information:
        - glacier_projects: Status of each glacier data directory
        - missing_files: List of expected but missing files
        - file_sizes: Size information for available files
        - recommendations: Suggestions for fixing issues
        
    Example:
        >>> status = diagnose_data_availability()
        >>> for glacier, info in status['glacier_projects'].items():
        ...     print(f"{glacier}: {info['status']}")
    """
    try:
        base_path = Path(base_dir)
        
        # Expected glacier project directories
        expected_projects = {
            'athabasca': "D:/Documents/Projects/athabasca_analysis",
            'haig': "D:/Documents/Projects/Haig_analysis", 
            'coropuna': "D:/Documents/Projects/Coropuna_glacier"
        }
        
        glacier_status = {}
        missing_files = []
        all_file_sizes = {}
        
        for glacier_id, project_path in expected_projects.items():
            project_dir = Path(project_path)
            
            status = {
                'project_exists': project_dir.exists(),
                'csv_dir_exists': False,
                'modis_files': [],
                'aws_files': [],
                'missing_files': [],
                'total_size_mb': 0
            }
            
            if project_dir.exists():
                csv_dir = project_dir / "data" / "csv"
                status['csv_dir_exists'] = csv_dir.exists()
                
                if csv_dir.exists():
                    # Check for MODIS files
                    modis_patterns = [
                        "*MODIS*.csv", "*modis*.csv", "*Terra*.csv", 
                        "*Aqua*.csv", "*MultiProduct*.csv"
                    ]
                    
                    for pattern in modis_patterns:
                        modis_files = list(csv_dir.glob(pattern))
                        for file in modis_files:
                            if file.is_file():
                                size_mb = file.stat().st_size / (1024 * 1024)
                                status['modis_files'].append({
                                    'name': file.name,
                                    'size_mb': round(size_mb, 2)
                                })
                                status['total_size_mb'] += size_mb
                    
                    # Check for AWS files  
                    aws_patterns = ["*AWS*.csv", "*aws*.csv", "*daily*.csv"]
                    
                    for pattern in aws_patterns:
                        aws_files = list(csv_dir.glob(pattern))
                        for file in aws_files:
                            if file.is_file():
                                size_mb = file.stat().st_size / (1024 * 1024)
                                status['aws_files'].append({
                                    'name': file.name,
                                    'size_mb': round(size_mb, 2)  
                                })
                                status['total_size_mb'] += size_mb
                    
                    # Check for expected files
                    expected_files = {
                        'athabasca': [
                            "Athabasca_Terra_Aqua_MultiProduct_2014-01-01_to_2021-01-01.csv",
                            "iceAWS_Atha_albedo_daily_20152020_filled_clean.csv"
                        ],
                        'haig': [
                            "Haig_MODIS_Pixel_Analysis_MultiProduct_2002_to_2016_fraction.csv",
                            "HaigAWS_daily_2002_2015_gapfilled.csv"
                        ],
                        'coropuna': [
                            "coropuna_glacier_2014-01-01_to_2025-01-01.csv",
                            "COROPUNA_simple.csv"
                        ]
                    }
                    
                    for expected_file in expected_files.get(glacier_id, []):
                        file_path = csv_dir / expected_file
                        if not file_path.exists():
                            status['missing_files'].append(expected_file)
                            missing_files.append(f"{glacier_id}: {expected_file}")
            
            glacier_status[glacier_id] = status
            all_file_sizes[glacier_id] = status['total_size_mb']
        
        # Generate recommendations
        recommendations = []
        
        for glacier_id, status in glacier_status.items():
            if not status['project_exists']:
                recommendations.append(f"Create project directory for {glacier_id}")
            elif not status['csv_dir_exists']:
                recommendations.append(f"Create data/csv directory for {glacier_id}")
            elif status['missing_files']:
                recommendations.append(f"Add missing files for {glacier_id}: {status['missing_files']}")
            elif not status['modis_files']:
                recommendations.append(f"Add MODIS data files for {glacier_id}")
            elif not status['aws_files']:
                recommendations.append(f"Add AWS data files for {glacier_id}")
        
        result = {
            'glacier_projects': glacier_status,
            'missing_files': missing_files,
            'file_sizes_mb': all_file_sizes,
            'total_data_size_mb': sum(all_file_sizes.values()),
            'recommendations': recommendations,
            'diagnosis_time': datetime.now().isoformat()
        }
        
        logger.info(f"Data availability diagnosis completed: {len(missing_files)} missing files")
        return result
        
    except Exception as e:
        logger.error(f"Error diagnosing data availability: {e}")
        return {'error': str(e)}


def diagnose_analysis_outputs(outputs_dir: str = "outputs") -> Dict[str, Any]:
    """
    Diagnose existing analysis outputs and their completeness.
    
    Args:
        outputs_dir: Directory containing analysis outputs
        
    Returns:
        Dict with analysis output information:
        - total_analyses: Count of analysis runs found
        - by_glacier: Breakdown by glacier type
        - recent_analyses: Most recent analyses by glacier
        - incomplete_analyses: Analyses missing expected files
        - disk_usage: Space used by outputs
        
    Example:
        >>> status = diagnose_analysis_outputs()
        >>> print(f"Found {status['total_analyses']} analysis runs")
    """
    try:
        outputs_path = Path(outputs_dir)
        
        if not outputs_path.exists():
            return {
                'total_analyses': 0,
                'error': f"Outputs directory does not exist: {outputs_dir}",
                'recommendation': f"Create outputs directory or run some analyses first"
            }
        
        # Find analysis directories
        analysis_dirs = [d for d in outputs_path.iterdir() if d.is_dir()]
        
        # Categorize by type
        by_type = {
            'individual_glacier': [],
            'comparative': [],
            'best_pixel': [],
            'other': []
        }
        
        by_glacier = {'athabasca': 0, 'haig': 0, 'coropuna': 0}
        recent_analyses = {}
        incomplete_analyses = []
        total_size_mb = 0
        
        for analysis_dir in analysis_dirs:
            dir_name = analysis_dir.name
            
            # Calculate directory size
            dir_size = sum(f.stat().st_size for f in analysis_dir.rglob('*') if f.is_file())
            total_size_mb += dir_size / (1024 * 1024)
            
            # Categorize analysis type
            if 'comparative' in dir_name:
                by_type['comparative'].append(dir_name)
            elif 'best_pixel' in dir_name:
                by_type['best_pixel'].append(dir_name)
            elif any(glacier in dir_name for glacier in ['athabasca', 'haig', 'coropuna']):
                by_type['individual_glacier'].append(dir_name)
                
                # Count by glacier
                for glacier in ['athabasca', 'haig', 'coropuna']:
                    if glacier in dir_name:
                        by_glacier[glacier] += 1
                        
                        # Track most recent
                        if glacier not in recent_analyses or dir_name > recent_analyses[glacier]:
                            recent_analyses[glacier] = dir_name
                        break
            else:
                by_type['other'].append(dir_name)
            
            # Check completeness
            results_dir = analysis_dir / "results"
            plots_dir = analysis_dir / "plots"
            
            expected_files = []
            if results_dir.exists():
                csv_files = list(results_dir.glob("*.csv"))
                expected_files.extend(csv_files)
            
            if plots_dir.exists():
                plot_files = list(plots_dir.glob("*.png"))
                expected_files.extend(plot_files)
            
            if len(expected_files) == 0:
                incomplete_analyses.append({
                    'directory': dir_name,
                    'issue': 'No results or plots found',
                    'size_mb': round(dir_size / (1024 * 1024), 2)
                })
        
        result = {
            'total_analyses': len(analysis_dirs),
            'by_type': {k: len(v) for k, v in by_type.items()},
            'by_glacier': by_glacier,
            'recent_analyses': recent_analyses,
            'incomplete_analyses': incomplete_analyses,
            'total_size_mb': round(total_size_mb, 2),
            'analysis_types_found': list(by_type.keys()),
            'oldest_analysis': min(analysis_dirs, key=lambda x: x.name).name if analysis_dirs else None,
            'newest_analysis': max(analysis_dirs, key=lambda x: x.name).name if analysis_dirs else None,
            'diagnosis_time': datetime.now().isoformat()
        }
        
        logger.info(f"Analysis outputs diagnosis completed: {len(analysis_dirs)} directories found")
        return result
        
    except Exception as e:
        logger.error(f"Error diagnosing analysis outputs: {e}")
        return {'error': str(e)}


def generate_diagnostic_report(output_file: Optional[str] = None) -> str:
    """
    Generate a comprehensive diagnostic report.
    
    Args:
        output_file: Optional file path to save the report
        
    Returns:
        str: Formatted diagnostic report
        
    Example:
        >>> report = generate_diagnostic_report("diagnostic_report.txt")
        >>> print(report[:200])  # Print first 200 characters
    """
    try:
        # Run all diagnostics
        system_info = diagnose_system_environment()
        data_status = diagnose_data_availability()
        output_status = diagnose_analysis_outputs()
        
        # Generate report
        report_lines = [
            "=" * 60,
            "MODIS Albedo Analysis Framework - Diagnostic Report",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SYSTEM ENVIRONMENT",
            "-" * 20
        ]
        
        if 'error' not in system_info:
            report_lines.extend([
                f"Python Version: {system_info['python_version']}",
                f"Platform: {system_info['platform']} {system_info['platform_version']}",
                f"Architecture: {system_info['architecture']}",
                f"Memory: {system_info['memory']['available_gb']:.1f} GB available / {system_info['memory']['total_gb']:.1f} GB total",
                f"Disk Space: {system_info['disk_space']['free_gb']:.1f} GB free / {system_info['disk_space']['total_gb']:.1f} GB total",
                "",
                "Key Dependencies:"
            ])
            
            for package, info in system_info['key_dependencies'].items():
                status = f"✓ {info['version']}" if info['installed'] else "✗ Not installed"
                report_lines.append(f"  {package}: {status}")
        else:
            report_lines.append(f"ERROR: {system_info['error']}")
        
        report_lines.extend([
            "",
            "DATA AVAILABILITY",
            "-" * 20
        ])
        
        if 'error' not in data_status:
            for glacier, status in data_status['glacier_projects'].items():
                report_lines.append(f"{glacier.title()} Glacier:")
                report_lines.append(f"  Project exists: {'✓' if status['project_exists'] else '✗'}")
                report_lines.append(f"  MODIS files: {len(status['modis_files'])}")
                report_lines.append(f"  AWS files: {len(status['aws_files'])}")
                report_lines.append(f"  Data size: {status['total_size_mb']:.1f} MB")
                
                if status['missing_files']:
                    report_lines.append(f"  Missing: {', '.join(status['missing_files'])}")
                report_lines.append("")
            
            if data_status['recommendations']:
                report_lines.extend([
                    "Recommendations:",
                    *[f"  • {rec}" for rec in data_status['recommendations']]
                ])
        else:
            report_lines.append(f"ERROR: {data_status['error']}")
        
        report_lines.extend([
            "",
            "ANALYSIS OUTPUTS",
            "-" * 20
        ])
        
        if 'error' not in output_status:
            report_lines.extend([
                f"Total Analyses: {output_status['total_analyses']}",
                f"Individual Glacier: {output_status['by_type']['individual_glacier']}",
                f"Comparative: {output_status['by_type']['comparative']}",
                f"Best Pixel: {output_status['by_type']['best_pixel']}",
                f"Total Output Size: {output_status['total_size_mb']:.1f} MB",
                ""
            ])
            
            report_lines.append("By Glacier:")
            for glacier, count in output_status['by_glacier'].items():
                recent = output_status['recent_analyses'].get(glacier, 'None')
                report_lines.append(f"  {glacier.title()}: {count} analyses (recent: {recent})")
            
            if output_status['incomplete_analyses']:
                report_lines.extend([
                    "",
                    "Incomplete Analyses:",
                    *[f"  • {inc['directory']}: {inc['issue']}" for inc in output_status['incomplete_analyses']]
                ])
        else:
            report_lines.append(f"ERROR: {output_status['error']}")
        
        report_lines.extend([
            "",
            "=" * 60,
            "End of Diagnostic Report",
            "=" * 60
        ])
        
        report = "\n".join(report_lines)
        
        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            logger.info(f"Diagnostic report saved to: {output_file}")
        
        return report
        
    except Exception as e:
        error_msg = f"Error generating diagnostic report: {e}"
        logger.error(error_msg)
        return error_msg


if __name__ == "__main__":
    # Generate and display diagnostic report when run directly
    print(generate_diagnostic_report())