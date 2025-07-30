#!/usr/bin/env python3
"""
Interactive MODIS Albedo Analysis Framework

This script provides a user-friendly menu interface for selecting and processing glaciers.
It automatically detects data availability and guides users through the analysis process.

Usage:
    python interactive_main.py
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.helpers import load_config, setup_logging
from main import AlbedoAnalysisPipeline
from pivot_based_main import PivotBasedAlbedoAnalysis
from src.analysis.comparative_interface import ComparativeAnalysisInterface


class InteractiveGlacierAnalysis:
    """Interactive interface for glacier analysis."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the interactive interface."""
        self.config_path = config_path
        self.config = load_config(config_path)
        self.glacier_sites_config = load_config('config/glacier_sites.yaml')
        
        # Set up minimal logging for interactive mode
        logging.basicConfig(level=logging.WARNING)
        
        # Initialize pipelines
        self.pipeline = AlbedoAnalysisPipeline(config_path)
        self.pivot_pipeline = PivotBasedAlbedoAnalysis(config_path)
    
    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def check_data_availability(self, glacier_id: str) -> Tuple[bool, List[str]]:
        """Check if required data files exist for a glacier."""
        glacier_config = self.glacier_sites_config['glaciers'].get(glacier_id, {})
        missing_files = []
        
        if not glacier_config:
            return False, ["Glacier configuration not found"]
        
        # Check MODIS files
        if 'data_files' in glacier_config and 'modis' in glacier_config['data_files']:
            # For Athabasca-type glaciers, check for MultiProduct file
            if glacier_config.get('data_type') == 'athabasca_multiproduct':
                multiproduct_found = False
                glacier_name = glacier_config['name'].lower().split()[0].lower()
                
                # Determine search paths based on glacier
                search_paths = [self.config['data']['modis_path']]
                if glacier_name == 'athabasca' and 'athabasca_modis_path' in self.config['data']:
                    search_paths.insert(0, self.config['data']['athabasca_modis_path'])
                elif glacier_name == 'haig' and 'haig_modis_path' in self.config['data']:
                    search_paths.insert(0, self.config['data']['haig_modis_path'])
                elif glacier_name == 'coropuna' and 'coropuna_modis_path' in self.config['data']:
                    search_paths.insert(0, self.config['data']['coropuna_modis_path'])
                
                # Search for MultiProduct file in all paths
                for search_path in search_paths:
                    if not os.path.exists(search_path):
                        continue
                    try:
                        for filename in os.listdir(search_path):
                            # Look for MultiProduct files (with or without AWS) or Coropuna glacier files
                            if (('MultiProduct' in filename and 
                                 (glacier_name in filename.lower() or 
                                  any(keyword in filename.lower() for keyword in ['multiproduct', 'multi_product']))) or
                                (glacier_name == 'coropuna' and 'coropuna_glacier' in filename.lower() and filename.endswith('.csv'))):
                                multiproduct_found = True
                                break
                    except (FileNotFoundError, OSError):
                        continue
                    
                    if multiproduct_found:
                        break
                
                if not multiproduct_found:
                    missing_files.append(f"MultiProduct file for {glacier_config['name']}")
            else:
                # Check individual MODIS files
                modis_path = self.config['data']['modis_path']
                for product, filename in glacier_config['data_files']['modis'].items():
                    file_path = os.path.join(modis_path, filename)
                    if not os.path.exists(file_path):
                        missing_files.append(f"MODIS {product}: {filename}")
        
        # Check AWS file
        if 'data_files' in glacier_config and 'aws' in glacier_config['data_files']:
            aws_file = glacier_config['data_files']['aws']
            glacier_name = glacier_config['name'].lower().split()[0].lower()
            
            # Determine AWS search paths based on glacier
            aws_search_paths = [self.config['data']['aws_path']]
            if glacier_name == 'athabasca' and 'athabasca_aws_path' in self.config['data']:
                aws_search_paths.insert(0, self.config['data']['athabasca_aws_path'])
            elif glacier_name == 'haig' and 'haig_aws_path' in self.config['data']:
                aws_search_paths.insert(0, self.config['data']['haig_aws_path'])
            elif glacier_name == 'coropuna' and 'coropuna_aws_path' in self.config['data']:
                aws_search_paths.insert(0, self.config['data']['coropuna_aws_path'])
            
            # Search for AWS file in all paths
            aws_found = False
            for aws_path in aws_search_paths:
                if not os.path.exists(aws_path):
                    continue
                aws_file_path = os.path.join(aws_path, aws_file)
                if os.path.exists(aws_file_path):
                    aws_found = True
                    break
            
            if not aws_found:
                missing_files.append(f"AWS data: {aws_file}")
        
        return len(missing_files) == 0, missing_files
    
    def display_analysis_type_menu(self, glacier_name: str, glacier_id: str) -> None:
        """Display analysis type selection menu for individual glacier."""
        # Get pixel counts for display
        from src.analysis.glacier_mapping_simple import MultiGlacierMapperSimple
        try:
            mapper = MultiGlacierMapperSimple()
            all_pixels = mapper.load_original_modis_data(glacier_id, analysis_mode=False)
            selected_pixels = mapper.load_original_modis_data(glacier_id, analysis_mode=True)
            
            all_count = len(all_pixels) if all_pixels is not None else 0
            selected_count = len(selected_pixels) if selected_pixels is not None else 0
        except:
            all_count = "?"
            selected_count = "?"
        
        print()
        print("+" + "-" * 78 + "+")
        print("|" + f" ANALYSIS MODE SELECTION - {glacier_name} ".center(78) + "|")
        print("+" + "-" * 78 + "+")
        print(f"| [1] Standard Analysis                                                  |")
        print(f"|     >> All available pixels ({all_count})                                          |")
        print("+" + "-" * 78 + "+")
        print(f"| [2] Best Pixel Analysis (RECOMMENDED)                                 |")
        print(f"|     >> Optimally selected pixels ({selected_count}) closest to AWS station             |")
        print("+" + "-" * 78 + "+")
        print("| [B] Back to Main Menu                                                 |")
        print("+" + "-" * 78 + "+")
        print()
        print("+" + "-" * 78 + "+")
        print("|" + " PIXEL SELECTION ALGORITHM ".center(78) + "|")
        print("+" + "-" * 78 + "+")
        print("| >> Distance to AWS station (60% weight)                              |")
        print("| >> Glacier fraction coverage (40% weight)                            |") 
        print("| >> Optimal pixels for highest accuracy weather correlation           |")
        print("+" + "-" * 78 + "+")
        print()
    
    def get_analysis_type_choice(self) -> str:
        """Get and validate analysis type choice."""
        valid_choices = ['1', '2', 'B']
        
        while True:
            choice = input("> Select analysis mode (1, 2, B): ").strip().upper()
            
            if choice in valid_choices:
                return choice
            else:
                print(f"[X] Invalid choice. Please select: {', '.join(valid_choices)}")
                print()
    
    def display_header(self):
        """Display the application header."""
        print()
        print("+" + "=" * 78 + "+")
        print("|" + " " * 78 + "|")
        print("|" + "    MODIS ALBEDO ANALYSIS FRAMEWORK".center(78) + "|")
        print("|" + "Interactive Analysis Suite".center(78) + "|")  
        print("|" + " " * 78 + "|")
        print("+" + "=" * 78 + "+")
        print()
    
    def display_glacier_menu(self) -> Dict[str, Any]:
        """Display the glacier selection menu and return glacier information."""
        glaciers = self.glacier_sites_config['glaciers']
        glacier_info = {}
        
        # Header for glacier section
        print("+" + "-" * 78 + "+")
        print("|" + " AVAILABLE GLACIERS ".center(78) + "|")
        print("+" + "-" * 78 + "+")
        
        for i, (glacier_id, config) in enumerate(glaciers.items(), 1):
            data_available, missing_files = self.check_data_availability(glacier_id)
            status_icon = "[OK]" if data_available else "[!]"
            status_text = "READY" if data_available else "MISSING DATA"
            
            # Check if this glacier gets enhanced plotting
            enhanced_plotting = config.get('data_type') == 'athabasca_multiproduct'
            
            glacier_info[str(i)] = {
                'id': glacier_id,
                'config': config,
                'available': data_available,
                'missing_files': missing_files
            }
            
            # Glacier main line
            glacier_name = config['name']
            region = config.get('region', 'Unknown Region')
            print(f"| [{i}] {glacier_name:<25} | {region:<20} | {status_icon} {status_text:<12} |")
            
            # Features line2
            if data_available and enhanced_plotting:
                features = "7 Enhanced Plots >> Outlier Analysis >> Statistical Suite"
                print(f"|     {features:<67} |")
            elif data_available:
                features = "Standard Analysis Suite"
                print(f"|     {features:<67} |")
            else:
                # Show missing files compactly
                if missing_files:
                    missing_text = f"Missing: {', '.join(missing_files[:2])}"
                    if len(missing_files) > 2:
                        missing_text += f" (+{len(missing_files)-2} more)"
                    print(f"|     {missing_text[:67]:<67} |")
            
            # Add separator line between glaciers (except last one)
            if i < len(glaciers):
                print("+" + "-" * 78 + "+")
        
        print("+" + "-" * 78 + "+")
        print()
        
        return glacier_info
    
    def display_options_menu(self):
        """Display the additional options menu."""
        print("+" + "-" * 78 + "+")
        print("|" + " ANALYSIS OPTIONS ".center(78) + "|")
        print("+" + "-" * 78 + "+")
        print("| [A] Process All Available Glaciers                                     |")
        print("|     >> Batch analysis with pixel selection options                   |")
        print("+" + "-" * 78 + "+")
        print("| [C] Comparative Analysis (Multi-Glacier)                              |")
        print("|     >> Cross-glacier statistical comparisons & visualizations       |")
        print("+" + "-" * 78 + "+")
        print("| [S] Show System Status                                                |")
        print("|     >> View data paths, configuration, and system health            |")
        print("+" + "-" * 78 + "+")
        print("| [Q] Quit                                                              |")
        print("|     >> Exit the analysis framework                                   |")
        print("+" + "-" * 78 + "+")
        print()
    
    def show_system_status(self):
        """Display system configuration and data path status."""
        print("\n" + "=" * 50)
        print("System Status")
        print("=" * 50)
        
        print(f"Configuration File: {self.config_path}")
        print(f"MODIS Data Path: {self.config['data']['modis_path']}")
        print(f"AWS Data Path: {self.config['data']['aws_path']}")
        print(f"Output Path: {self.config['output']['base_path']}")
        print()
        
        # Check path accessibility
        paths_to_check = [
            ("MODIS Data", self.config['data']['modis_path']),
            ("AWS Data", self.config['data']['aws_path']),
            ("Output", self.config['output']['base_path'])
        ]
        
        print("Path Accessibility:")
        for name, path in paths_to_check:
            status = "[OK]" if os.path.exists(path) else "[Not Found]"
            print(f"  {name}: {status}")
            print(f"    Path: {path}")
        
        print()
        input("Press Enter to continue...")
    
    def get_user_choice(self, glacier_info: Dict[str, Any]) -> str:
        """Get and validate user choice."""
        valid_choices = list(glacier_info.keys()) + ['A', 'C', 'S', 'Q']
        
        print("+" + "-" * 78 + "+")
        print("|" + f" Select your option: {', '.join(sorted(glacier_info.keys()))}, A, C, S, Q ".ljust(78) + "|")
        print("+" + "-" * 78 + "+")
        
        while True:
            choice = input("> Your choice: ").strip().upper()
            
            if choice in valid_choices:
                return choice
            else:
                print(f"[X] Invalid choice. Please select: {', '.join(valid_choices)}")
                print()
    
    def process_single_glacier_with_pixel_selection(self, glacier_id: str, glacier_name: str, use_selected_pixels: bool = False) -> bool:
        """Process a single glacier with pixel selection option."""
        pixel_mode = "Selected pixels" if use_selected_pixels else "All pixels"
        print(f"\nProcessing {glacier_name} ({pixel_mode})...")
        print("-" * 50)
        
        if use_selected_pixels:
            # Show pixel selection details
            from src.analysis.glacier_mapping_simple import MultiGlacierMapperSimple
            try:
                mapper = MultiGlacierMapperSimple()
                selected_pixels = mapper.load_original_modis_data(glacier_id, analysis_mode=True)
                all_pixels = mapper.load_original_modis_data(glacier_id, analysis_mode=False)
                
                if selected_pixels is not None and all_pixels is not None:
                    print(f"Pixel Selection Details:")
                    print(f"- Selected {len(selected_pixels)} pixels from {len(all_pixels)} available")
                    
                    # Show selection details for each pixel
                    for idx, row in selected_pixels.iterrows():
                        pixel_id = row.get('pixel_id', 'unknown')
                        lat, lon = row['latitude'], row['longitude']
                        glacier_frac = row.get('glacier_fraction', 'N/A')
                        if glacier_frac != 'N/A':
                            print(f"- Pixel {pixel_id}: {lat:.4f}N, {lon:.4f}E (glacier_frac: {glacier_frac:.3f})")
                        else:
                            print(f"- Pixel {pixel_id}: {lat:.4f}N, {lon:.4f}E")
                    print()
            except Exception as e:
                print(f"Warning: Could not display pixel selection details: {e}")
        
        # Call the processing method with pixel selection parameter
        return self.process_single_glacier(glacier_id, glacier_name, use_selected_pixels)
    
    def process_single_glacier(self, glacier_id: str, glacier_name: str, use_selected_pixels: bool = False) -> bool:
        """Process a single glacier and return success status."""
        print(f"\nProcessing {glacier_name}...")
        print("-" * 50)
        
        try:
            # Set up logging to show progress
            logging.getLogger().setLevel(logging.INFO)
            
            # Determine which analysis pipeline to use
            glacier_config = self.glacier_sites_config['glaciers'].get(glacier_id, {})
            use_enhanced_plotting = glacier_config.get('data_type') == 'athabasca_multiproduct'
            
            if use_enhanced_plotting:
                print("Using enhanced visualization suite (7 plot types)...")
                result = self.pivot_pipeline.process_glacier(glacier_id, use_selected_pixels)
                
                # Enhanced results reporting
                print(f"\n[SUCCESS] Enhanced analysis completed successfully!")
                print(f"Generated comprehensive visualization suite with 7 plot types + spatial maps:")
                print("  1. User-style comprehensive analysis (seasonal)")
                print("  2. Multi-panel summary figure")
                print("  3. Time series analysis")
                print("  4. Distribution analysis")  
                print("  5. Outlier analysis (before/after)")
                print("  6. Seasonal analysis")
                print("  7. Correlation & bias analysis")
                print("  + Spatial maps (glacier mask, MODIS pixels, AWS station)")
            else:
                print("Using standard analysis pipeline...")
                result = self.pipeline.process_glacier(glacier_id)
                print(f"\n[SUCCESS] Standard analysis completed successfully!")
            
            print(f"Output directory: {result['output_directory']}")
            
            # Show key statistics if available
            if 'statistics' in result and 'method_comparison' in result['statistics']:
                print("\nKey Results:")
                for method, stats in result['statistics']['method_comparison'].items():
                    if use_enhanced_plotting:
                        # Enhanced statistics display for pivot-based analysis
                        print(f"  {method}: n={stats['n_samples']}, r={stats['r']:.3f}, RMSE={stats['rmse']:.3f}, Bias={stats['bias']:.3f}")
                    else:
                        # Standard statistics display
                        print(f"  {method}: n={stats['n_samples']}, r={stats.get('correlation', stats.get('r', 0)):.3f}, RMSE={stats['rmse']:.3f}")
            
            # Show outlier analysis results if available
            if use_enhanced_plotting and 'outlier_analysis' in result.get('statistics', {}):
                outlier_info = result['statistics']['outlier_analysis'].get('outlier_info', {})
                if outlier_info:
                    print("\nOutlier Analysis Results:")
                    for method, info in outlier_info.items():
                        improvement = info.get('r_improvement_pct', 0)
                        n_outliers = info.get('n_outliers', 0)
                        print(f"  {method}: Removed {n_outliers} outliers, correlation improved by {improvement:.1f}%")
            
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Analysis failed: {e}")
            return False
        finally:
            # Reset logging level
            logging.getLogger().setLevel(logging.WARNING)
    
    def get_pixel_mode_for_all_glaciers(self) -> bool:
        """Get pixel selection mode for processing all glaciers."""
        print()
        print("+" + "-" * 78 + "+")
        print("|" + " BATCH PROCESSING MODE ".center(78) + "|")
        print("+" + "-" * 78 + "+")
        print("| [1] Standard Analysis                                                  |")
        print("|     >> Process all available pixels for each glacier                 |")
        print("+" + "-" * 78 + "+")
        print("| [2] Best Pixel Analysis (RECOMMENDED)                                 |")
        print("|     >> Process optimally selected pixels for enhanced accuracy      |")
        print("+" + "-" * 78 + "+")
        print()
        
        while True:
            choice = input("> Select processing mode (1, 2): ").strip()
            if choice == '1':
                return False  # Use all pixels
            elif choice == '2':
                return True   # Use selected pixels
            else:
                print("Invalid choice. Please enter 1 or 2.")
    
    def process_all_available_glaciers(self, glacier_info: Dict[str, Any]) -> int:
        """Process all glaciers with available data."""
        available_glaciers = [(info['id'], info['config']['name']) 
                            for info in glacier_info.values() 
                            if info['available']]
        
        if not available_glaciers:
            print("No glaciers with complete data available for processing.")
            return 0
        
        # Get pixel selection mode
        use_selected_pixels = self.get_pixel_mode_for_all_glaciers()
        pixel_mode = "Selected pixels" if use_selected_pixels else "All pixels"
        
        # Count enhanced vs standard analysis types
        enhanced_count = 0
        standard_count = 0
        for glacier_id, _ in available_glaciers:
            glacier_config = self.glacier_sites_config['glaciers'].get(glacier_id, {})
            if glacier_config.get('data_type') == 'athabasca_multiproduct':
                enhanced_count += 1
            else:
                standard_count += 1
        
        print(f"\nProcessing {len(available_glaciers)} glacier(s) with available data ({pixel_mode})...")
        print(f"  - {enhanced_count} glacier(s) with enhanced visualization suite (7 plot types)")
        print(f"  - {standard_count} glacier(s) with standard analysis")
        if use_selected_pixels:
            print(f"  - Using optimally selected pixels closest to AWS stations")
        print("=" * 70)
        
        success_count = 0
        for glacier_id, glacier_name in available_glaciers:
            success = self.process_single_glacier_with_pixel_selection(glacier_id, glacier_name, use_selected_pixels)
            if success:
                success_count += 1
            print("\n" + "-" * 70)
        
        print(f"\nCompleted: {success_count}/{len(available_glaciers)} glaciers processed successfully.")
        if enhanced_count > 0:
            print("Enhanced visualization suites include:")
            print("  • User-style comprehensive analysis • Multi-panel summary figures")
            print("  • Time series analysis • Distribution analysis • Outlier analysis")
            print("  • Seasonal analysis • Correlation & bias analysis")
        return success_count
    
    def run(self):
        """Run the interactive interface."""
        while True:
            self.clear_screen()
            self.display_header()
            
            # Display glacier menu
            glacier_info = self.display_glacier_menu()
            
            # Display options
            self.display_options_menu()
            
            # Get user choice
            choice = self.get_user_choice(glacier_info)
            
            if choice == 'Q':
                print("\nExiting MODIS Albedo Analysis Framework. Goodbye!")
                break
            
            elif choice == 'S':
                self.show_system_status()
                continue
            
            elif choice == 'A':
                self.process_all_available_glaciers(glacier_info)
                input("\nPress Enter to continue...")
                continue
                
            elif choice == 'C':
                # Launch comparative analysis interface
                print("\n[LAUNCHING] Multi-Glacier Comparative Analysis...")
                try:
                    comparative_interface = ComparativeAnalysisInterface()
                    comparative_interface.run_interactive_session()
                except Exception as e:
                    print(f"\n[ERROR] Error in comparative analysis: {e}")
                    logging.error(f"Comparative analysis error: {e}")
                    input("Press Enter to continue...")
                continue
            
            elif choice in glacier_info:
                # Process selected glacier with analysis type menu
                selected = glacier_info[choice]
                
                if not selected['available']:
                    print(f"\n[ERROR] Cannot process {selected['config']['name']} - missing required data files:")
                    for missing in selected['missing_files']:
                        print(f"  - {missing}")
                    input("\nPress Enter to continue...")
                    continue
                
                # Show analysis type menu
                while True:
                    self.clear_screen()
                    self.display_header()
                    self.display_analysis_type_menu(selected['config']['name'], selected['id'])
                    
                    analysis_choice = self.get_analysis_type_choice()
                    
                    if analysis_choice == 'B':
                        break  # Go back to main menu
                    elif analysis_choice == '1':
                        # Standard analysis (all pixels)
                        success = self.process_single_glacier_with_pixel_selection(
                            selected['id'], selected['config']['name'], use_selected_pixels=False)
                        break
                    elif analysis_choice == '2':
                        # Best pixel analysis
                        success = self.process_single_glacier_with_pixel_selection(
                            selected['id'], selected['config']['name'], use_selected_pixels=True)
                        break
                
                # If user chose 'B', continue to main menu
                if analysis_choice == 'B':
                    continue
                
                if success:
                    # Ask if user wants to process another glacier
                    print("\nOptions:")
                    print("1. Process another glacier")
                    print("2. Return to main menu")
                    print("3. Quit")
                    
                    while True:
                        next_choice = input("Enter your choice (1-3): ").strip()
                        if next_choice == '1':
                            break
                        elif next_choice == '2':
                            break
                        elif next_choice == '3':
                            print("\nExiting MODIS Albedo Analysis Framework. Goodbye!")
                            return
                        else:
                            print("Invalid choice. Please enter 1, 2, or 3.")
                    
                    if next_choice == '3':
                        break
                else:
                    input("\nPress Enter to continue...")


def main():
    """Main function to run the interactive interface."""
    try:
        app = InteractiveGlacierAnalysis()
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()