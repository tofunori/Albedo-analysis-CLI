#!/usr/bin/env python3
"""
MODIS Albedo Analysis Framework - Unified Entry Point

This is the single entry point for the MODIS Albedo Analysis Framework.
It provides both interactive menu interface and command-line options for flexible usage.

Interactive Mode (default):
    python interactive_main.py
    
Command-Line Mode:
    python interactive_main.py --glacier haig --analysis-mode comprehensive --selected-pixels
    python interactive_main.py --all-glaciers --analysis-mode basic
    python interactive_main.py --comparative-analysis
    python interactive_main.py --help

Features:
- Interactive menu system with data availability detection
- Command-line interface for automation and scripting
- Multiple analysis modes (auto, basic, enhanced, comprehensive)
- Pixel selection algorithms for enhanced accuracy
- Batch processing capabilities
- Comparative analysis across glaciers
- System diagnostics and status monitoring
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
import logging

from utils.config.helpers import load_config, setup_logging
from utils.config.glacier_manager import GlacierConfigManager
from albedo_engine.engine import AlbedoAnalysisEngine

# Optional comparative analysis (requires additional dependencies)
try:
    from analysis.comparative.interface import ComparativeAnalysisInterface
    COMPARATIVE_ANALYSIS_AVAILABLE = True
except ImportError as e:
    ComparativeAnalysisInterface = None
    COMPARATIVE_ANALYSIS_AVAILABLE = False
    print(f"Warning: Comparative analysis not available due to missing dependencies: {e}")


class InteractiveGlacierAnalysis:
    """Interactive interface for glacier analysis."""
    
    def __init__(self, config_path: str = 'config/config.yaml'):
        """Initialize the interactive interface."""
        self.config_path = config_path
        self.config = load_config(config_path)
        self.glacier_sites_config = load_config('config/glacier_sites.yaml')
        
        # Set up minimal logging for interactive mode
        logging.basicConfig(level=logging.WARNING)
        
        # Initialize unified analysis engine
        self.analysis_engine = AlbedoAnalysisEngine(config_path)
        
        # Initialize glacier configuration manager
        self.glacier_manager = GlacierConfigManager()
    
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
        from analysis.spatial.glacier_mapping_simple import MultiGlacierMapperSimple
        try:
            mapper = MultiGlacierMapperSimple()
            all_pixels = mapper.load_original_modis_data(glacier_id, analysis_mode=False)
            selected_pixels = mapper.load_original_modis_data(glacier_id, analysis_mode=True)
            
            all_count = len(all_pixels) if all_pixels is not None else 0
            selected_count = len(selected_pixels) if selected_pixels is not None else 0
        except:
            all_count = "?"
            selected_count = "?"
        
        # Get current plot mode
        current_plot_mode = self.config.get('visualization', {}).get('plot_output', {}).get('plot_mode', 'both')
        
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
        print(f"| [P] Plot Mode Settings (Current: {current_plot_mode})                              |")
        print(f"|     >> Choose which plots to generate                                |")
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
        valid_choices = ['1', '2', 'P', 'B']
        
        while True:
            choice = input("> Select analysis mode (1, 2, P, B): ").strip().upper()
            
            if choice in valid_choices:
                return choice
            else:
                print(f"[X] Invalid choice. Please select: {', '.join(valid_choices)}")
                print()
    
    def display_plot_mode_menu(self) -> None:
        """Display plot mode selection menu."""
        current_plot_mode = self.config.get('visualization', {}).get('plot_output', {}).get('plot_mode', 'both')
        
        print()
        print("+" + "-" * 78 + "+")
        print("|" + " PLOT MODE SELECTION ".center(78) + "|")
        print("+" + "-" * 78 + "+")
        print("| [1] Original Plots (All 7 plots like before)                         |")
        print("|     >> Comprehensive summary, temporal, distribution, outlier,       |")
        print("|     >> seasonal, correlation/bias analysis + comprehensive analysis  |")
        print("+" + "-" * 78 + "+")
        print("| [2] Individual Refined Plots                                         |")
        print("|     >> Specialized plots with redundancy elimination                 |")
        print("+" + "-" * 78 + "+")
        print("| [3] Dashboard Only                                                   |")
        print("|     >> Single comprehensive dashboard plot                           |")
        print("+" + "-" * 78 + "+")
        print("| [4] Both Systems (Original + Refined)                                |")
        print("|     >> All original plots + refined features                         |")
        print("+" + "-" * 78 + "+")
        print(f"| Current mode: {current_plot_mode:<60} |")
        print("+" + "-" * 78 + "+")
        print("| [B] Back to Analysis Menu                                             |")
        print("+" + "-" * 78 + "+")
        print()
    
    def get_plot_mode_choice(self) -> str:
        """Get and validate plot mode choice."""
        valid_choices = ['1', '2', '3', '4', 'B']
        
        while True:
            choice = input("> Select plot mode (1, 2, 3, 4, B): ").strip().upper()
            
            if choice in valid_choices:
                return choice
            else:
                print(f"[X] Invalid choice. Please select: {', '.join(valid_choices)}")
                print()
    
    def update_plot_mode(self, choice: str) -> bool:
        """Update plot mode in configuration."""
        plot_mode_map = {
            '1': 'original',
            '2': 'individual', 
            '3': 'dashboard',
            '4': 'both'
        }
        
        if choice in plot_mode_map:
            new_mode = plot_mode_map[choice]
            
            # Update the configuration in memory
            if 'visualization' not in self.config:
                self.config['visualization'] = {}
            if 'plot_output' not in self.config['visualization']:
                self.config['visualization']['plot_output'] = {}
            
            self.config['visualization']['plot_output']['plot_mode'] = new_mode
            
            # Update the config file
            import yaml
            try:
                with open('config/config.yaml', 'r') as f:
                    file_config = yaml.safe_load(f)
                
                if 'visualization' not in file_config:
                    file_config['visualization'] = {}
                if 'plot_output' not in file_config['visualization']:
                    file_config['visualization']['plot_output'] = {}
                
                file_config['visualization']['plot_output']['plot_mode'] = new_mode
                
                with open('config/config.yaml', 'w') as f:
                    yaml.dump(file_config, f, default_flow_style=False, sort_keys=False)
                
                print(f"[OK] Plot mode updated to: {new_mode}")
                return True
                
            except Exception as e:
                print(f"[X] Error updating configuration: {e}")
                return False
        
        return False
    
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
        if COMPARATIVE_ANALYSIS_AVAILABLE:
            print("| [C] Comparative Analysis (Multi-Glacier)                              |")
            print("|     >> Cross-glacier statistical comparisons & visualizations       |")
        else:
            print("| [C] Comparative Analysis (UNAVAILABLE - missing dependencies)         |")
            print("|     >> Install sklearn and other deps for comparative analysis       |")
        print("+" + "-" * 78 + "+")
        print("| [N] Add New Glacier                                                   |")
        print("|     >> Configure new glacier site with guided setup wizard          |")
        print("+" + "-" * 78 + "+")
        print("| [S] Show System Status                                                |")
        print("|     >> View data paths, configuration, and system health            |")
        print("+" + "-" * 78 + "+")
        print("| [Q] Quit                                                              |")
        print("|     >> Exit the analysis framework                                   |")
        print("+" + "-" * 78 + "+")
        print()
    
    def run_glacier_setup_wizard(self):
        """Run the interactive glacier setup wizard."""
        print("\n" + "=" * 80)
        print(" GLACIER SETUP WIZARD ".center(80))
        print("=" * 80)
        print("This wizard will guide you through setting up a new glacier site.")
        print("You'll need: MODIS CSV data, AWS CSV data, glacier mask file, and coordinates.")
        print("-" * 80)
        
        try:
            # Step 1: Basic glacier information
            glacier_data = self.get_glacier_basic_info()
            if not glacier_data:
                return
            
            # Step 2: Configure MODIS data
            modis_success = self.configure_modis_data(glacier_data)
            if not modis_success:
                return
            
            # Step 3: Configure AWS data
            aws_success = self.configure_aws_data(glacier_data)
            if not aws_success:
                return
            
            # Step 4: Configure mask data
            mask_success = self.configure_mask_data(glacier_data)
            if not mask_success:
                return
            
            # Step 5: Configure AWS stations
            stations_success = self.configure_aws_stations(glacier_data)
            if not stations_success:
                return
            
            # Step 6: Preview and confirm
            confirmed = self.preview_glacier_config(glacier_data)
            if not confirmed:
                print("\n[CANCELLED] Glacier setup cancelled by user.")
                return
            
            # Step 7: Save configuration
            success = self.save_glacier_config(glacier_data)
            if success:
                print(f"\n[SUCCESS] Glacier '{glacier_data['id']}' has been added successfully!")
                print("You can now select it from the main menu for analysis.")
                
                # Reload configurations
                self.glacier_sites_config = load_config('config/glacier_sites.yaml')
            else:
                print("\n[ERROR] Failed to save glacier configuration.")
                
        except KeyboardInterrupt:
            print("\n\n[CANCELLED] Glacier setup cancelled by user.")
        except Exception as e:
            print(f"\n[ERROR] An error occurred during setup: {e}")
    
    def get_glacier_basic_info(self) -> Dict[str, Any]:
        """Get basic glacier information from user."""
        print("\n" + "-" * 40)
        print("STEP 1: Basic Glacier Information")
        print("-" * 40)
        
        glacier_data = {}
        
        # Get glacier ID
        while True:
            glacier_id = input("Enter glacier ID (lowercase, no spaces, e.g., 'mendenhall'): ").strip().lower()
            
            valid_id, error_msg = self.glacier_manager.validate_glacier_id(glacier_id, self.glacier_sites_config)
            if valid_id:
                glacier_data['id'] = glacier_id
                break
            else:
                print(f"[ERROR] {error_msg}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
        
        # Get glacier name
        glacier_name = input("Enter full glacier name (e.g., 'Mendenhall Glacier'): ").strip()
        if not glacier_name:
            print("[ERROR] Glacier name cannot be empty.")
            return None
        glacier_data['name'] = glacier_name
        
        # Get region
        region = input("Enter region (e.g., 'Alaska', 'Himalayas'): ").strip()
        if not region:
            region = "Unknown Region"
        glacier_data['region'] = region
        
        # Get coordinates
        while True:
            try:
                lat = input("Enter latitude (decimal degrees, e.g., 58.4623): ").strip()
                lon = input("Enter longitude (decimal degrees, e.g., -134.5839): ").strip()
                
                valid_coords, error_msg = self.glacier_manager.validate_coordinates(lat, lon)
                if valid_coords:
                    glacier_data['coordinates'] = {'lat': float(lat), 'lon': float(lon)}
                    break
                else:
                    print(f"[ERROR] {error_msg}")
                    retry = input("Try again? (y/n): ").strip().lower()
                    if retry != 'y':
                        return None
            except ValueError:
                print("[ERROR] Coordinates must be valid numbers.")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return None
        
        return glacier_data
    
    def configure_modis_data(self, glacier_data: Dict[str, Any]) -> bool:
        """Configure MODIS data file."""
        print("\n" + "-" * 40)
        print("STEP 2: MODIS Data Configuration")
        print("-" * 40)
        print("Supported formats:")
        print("  • Multi-product CSV (like Athabasca format with 'method' column)")
        print("  • Standard MODIS CSV (date, latitude, longitude, albedo columns)")
        
        while True:
            modis_file = input("\nEnter path to MODIS CSV file: ").strip()
            if not modis_file:
                print("[ERROR] File path cannot be empty.")
                continue
            
            # Validate file
            valid, error_msg, file_info = self.glacier_manager.validate_modis_file(modis_file)
            
            if valid:
                print(f"[SUCCESS] Valid MODIS file detected!")
                print(f"  Format: {file_info.get('data_type', 'Unknown')}")
                print(f"  Columns: {len(file_info.get('columns', []))} columns")
                if 'methods' in file_info:
                    print(f"  Available methods: {', '.join(file_info['methods'])}")
                print(f"  Total rows: {file_info.get('total_rows', file_info.get('row_count', 'Unknown'))}")
                
                glacier_data['data_files'] = {'modis': modis_file}
                
                # Determine data type
                data_type, _ = self.glacier_manager.detect_data_format(modis_file)
                glacier_data['data_type'] = data_type
                
                return True
            else:
                print(f"[ERROR] {error_msg}")
                print(f"File info: {file_info}")
                retry = input("Try a different file? (y/n): ").strip().lower()
                if retry != 'y':
                    return False
    
    def configure_aws_data(self, glacier_data: Dict[str, Any]) -> bool:
        """Configure AWS data file."""
        print("\n" + "-" * 40)
        print("STEP 3: AWS Data Configuration")
        print("-" * 40)
        print("Supported AWS formats:")
        print("  • Standard: Time, Albedo columns")
        print("  • Coropuna: Timestamp, Albedo columns")
        print("  • Haig: Year, Day, albedo columns (semicolon-separated)")
        
        while True:
            aws_file = input("\nEnter path to AWS CSV file: ").strip()
            if not aws_file:
                print("[ERROR] File path cannot be empty.")
                continue
            
            # Validate file
            valid, error_msg, file_info = self.glacier_manager.validate_aws_file(aws_file)
            
            if valid:
                print(f"[SUCCESS] Valid AWS file detected!")
                print(f"  Columns: {', '.join(file_info.get('columns', []))}")
                print(f"  Rows: {file_info.get('row_count', 'Unknown')}")
                
                glacier_data['data_files']['aws'] = aws_file
                return True
            else:
                print(f"[ERROR] {error_msg}")
                print(f"Available columns: {file_info.get('columns', [])}")
                retry = input("Try a different file? (y/n): ").strip().lower()
                if retry != 'y':
                    return False
    
    def configure_mask_data(self, glacier_data: Dict[str, Any]) -> bool:
        """Configure glacier mask file."""
        print("\n" + "-" * 40)
        print("STEP 4: Glacier Mask Configuration")
        print("-" * 40)
        print("Supported formats:")
        print("  • Shapefile (.shp with .shx, .dbf components)")
        print("  • GeoTIFF (.tif, .tiff)")
        
        while True:
            mask_file = input("\nEnter path to glacier mask file: ").strip()
            if not mask_file:
                print("[ERROR] File path cannot be empty.")
                continue
            
            # Validate file
            valid, error_msg = self.glacier_manager.validate_mask_file(mask_file)
            
            if valid:
                print(f"[SUCCESS] Valid mask file detected!")
                glacier_data['data_files']['mask'] = mask_file
                return True
            else:
                print(f"[ERROR] {error_msg}")
                retry = input("Try a different file? (y/n): ").strip().lower()
                if retry != 'y':
                    return False
    
    def configure_aws_stations(self, glacier_data: Dict[str, Any]) -> bool:
        """Configure AWS station information."""
        print("\n" + "-" * 40)
        print("STEP 5: AWS Station Configuration")
        print("-" * 40)
        
        stations = {}
        
        while True:
            station_id = input("Enter AWS station ID (e.g., 'main_station'): ").strip()
            if not station_id:
                print("[ERROR] Station ID cannot be empty.")
                continue
            
            station_name = input(f"Enter station name (e.g., '{glacier_data['name']} AWS'): ").strip()
            if not station_name:
                station_name = f"{glacier_data['name']} AWS"
            
            # Get station coordinates
            while True:
                try:
                    print("Enter AWS station coordinates:")
                    station_lat = float(input("  Latitude: ").strip())
                    station_lon = float(input("  Longitude: ").strip())
                    
                    valid_coords, error_msg = self.glacier_manager.validate_coordinates(station_lat, station_lon)
                    if valid_coords:
                        break
                    else:
                        print(f"[ERROR] {error_msg}")
                except ValueError:
                    print("[ERROR] Coordinates must be valid numbers.")
            
            # Get station elevation (optional)
            elevation = input("Enter station elevation in meters (optional): ").strip()
            try:
                elevation = int(elevation) if elevation else 1000
            except ValueError:
                elevation = 1000
                print("Using default elevation: 1000m")
            
            stations[station_id] = {
                'name': station_name,
                'lat': station_lat,
                'lon': station_lon,
                'elevation': elevation
            }
            
            # Ask if user wants to add more stations
            more_stations = input("Add another AWS station? (y/n): ").strip().lower()
            if more_stations != 'y':
                break
        
        glacier_data['aws_stations'] = stations
        return True
    
    def preview_glacier_config(self, glacier_data: Dict[str, Any]) -> bool:
        """Preview glacier configuration and get user confirmation."""
        print("\n" + "=" * 60)
        print(" CONFIGURATION PREVIEW ".center(60))
        print("=" * 60)
        
        print(f"Glacier ID: {glacier_data['id']}")
        print(f"Name: {glacier_data['name']}")
        print(f"Region: {glacier_data['region']}")
        print(f"Coordinates: {glacier_data['coordinates']['lat']}, {glacier_data['coordinates']['lon']}")
        print(f"Data Type: {glacier_data.get('data_type', 'Unknown')}")
        
        print("\nData Files:")
        print(f"  MODIS: {glacier_data['data_files']['modis']}")
        print(f"  AWS: {glacier_data['data_files']['aws']}")
        print(f"  Mask: {glacier_data['data_files']['mask']}")
        
        print("\nAWS Stations:")
        for station_id, station_info in glacier_data['aws_stations'].items():
            print(f"  {station_id}: {station_info['name']}")
            print(f"    Location: {station_info['lat']}, {station_info['lon']}")
            print(f"    Elevation: {station_info['elevation']}m")
        
        print("=" * 60)
        
        while True:
            confirm = input("Save this configuration? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return True
            elif confirm in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no.")
    
    def save_glacier_config(self, glacier_data: Dict[str, Any]) -> bool:
        """Save the glacier configuration."""
        try:
            success, message = self.glacier_manager.add_glacier(glacier_data['id'], glacier_data)
            if success:
                print(f"\n[SUCCESS] {message}")
                return True
            else:
                print(f"\n[ERROR] {message}")
                return False
        except Exception as e:
            print(f"\n[ERROR] Failed to save configuration: {e}")
            return False

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
        valid_choices = list(glacier_info.keys()) + ['A', 'C', 'N', 'S', 'Q']
        
        print("+" + "-" * 78 + "+")
        print("|" + f" Select your option: {', '.join(sorted(glacier_info.keys()))}, A, C, N, S, Q ".ljust(78) + "|")
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
            from analysis.spatial.glacier_mapping_simple import MultiGlacierMapperSimple
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
            
            # Determine analysis mode based on glacier configuration
            glacier_config = self.glacier_sites_config['glaciers'].get(glacier_id, {})
            use_enhanced_plotting = glacier_config.get('data_type') == 'athabasca_multiproduct'
            
            # Set analysis mode - engine will auto-determine the best approach
            if use_enhanced_plotting:
                analysis_mode = 'comprehensive'  # Full 7-plot suite + enhanced features
                print("Using comprehensive analysis mode (7 plot types + advanced features)...")
            else:
                analysis_mode = 'basic'  # Standard analysis
                print("Using standard analysis mode...")
            
            # Process glacier with unified engine
            result = self.analysis_engine.process_glacier(
                glacier_id=glacier_id,
                use_selected_pixels=use_selected_pixels,
                analysis_mode=analysis_mode
            )
            
            # Display results based on analysis mode
            if analysis_mode == 'comprehensive':
                print(f"\n[SUCCESS] Comprehensive analysis completed successfully!")
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
                print(f"\n[SUCCESS] Standard analysis completed successfully!")
            
            print(f"Output directory: {result['output_directory']}")
            
            # Show key statistics if available
            if 'statistics' in result and 'method_comparison' in result['statistics']:
                print("\nKey Results:")
                for method, stats in result['statistics']['method_comparison'].items():
                    if analysis_mode == 'comprehensive':
                        # Enhanced statistics display for comprehensive analysis
                        print(f"  {method}: n={stats['n_samples']}, r={stats['r']:.3f}, RMSE={stats['rmse']:.3f}, Bias={stats['bias']:.3f}")
                    else:
                        # Standard statistics display
                        print(f"  {method}: n={stats['n_samples']}, r={stats.get('correlation', stats.get('r', 0)):.3f}, RMSE={stats['rmse']:.3f}")
            
            # Show outlier analysis results if available
            if analysis_mode == 'comprehensive' and 'outlier_analysis' in result.get('statistics', {}):
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
            
            elif choice == 'N':
                # Add new glacier wizard
                self.run_glacier_setup_wizard()
                input("\nPress Enter to continue...")
                continue
            
            elif choice == 'S':
                self.show_system_status()
                continue
            
            elif choice == 'A':
                self.process_all_available_glaciers(glacier_info)
                input("\nPress Enter to continue...")
                continue
                
            elif choice == 'C':
                if COMPARATIVE_ANALYSIS_AVAILABLE:
                    # Launch comparative analysis interface
                    print("\n[LAUNCHING] Multi-Glacier Comparative Analysis...")
                    try:
                        comparative_interface = ComparativeAnalysisInterface()
                        comparative_interface.run_interactive_session()
                    except Exception as e:
                        print(f"\n[ERROR] Error in comparative analysis: {e}")
                        logging.error(f"Comparative analysis error: {e}")
                        input("Press Enter to continue...")
                else:
                    print("\n[ERROR] Comparative analysis is not available due to missing dependencies.")
                    print("Please install required packages (sklearn, pyarrow, etc.) to enable this feature.")
                    input("\nPress Enter to continue...")
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
                    elif analysis_choice == 'P':
                        # Plot mode selection
                        plot_menu_active = True
                        while plot_menu_active:
                            self.clear_screen()
                            self.display_header()
                            self.display_plot_mode_menu()
                            
                            plot_choice = self.get_plot_mode_choice()
                            
                            if plot_choice == 'B':
                                plot_menu_active = False  # Go back to analysis menu
                            else:
                                if self.update_plot_mode(plot_choice):
                                    input("\nPress Enter to continue...")
                                    plot_menu_active = False
                                else:
                                    input("\nPress Enter to try again...")
                        continue  # Back to analysis menu  
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
    """Main function to run the interactive interface or command-line mode."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='MODIS Albedo Analysis Framework - Interactive & Command Line Interface',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python interactive_main.py
    
  Command-line mode:
    python interactive_main.py --glacier haig --analysis-mode comprehensive --selected-pixels
    python interactive_main.py --all-glaciers --analysis-mode basic
    python interactive_main.py --comparative-analysis
        """
    )
    
    # Analysis target options
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument('--glacier', type=str, 
                             help='Process a specific glacier (e.g., haig, athabasca, coropuna)')
    target_group.add_argument('--all-glaciers', action='store_true',
                             help='Process all available glaciers with data')
    target_group.add_argument('--comparative-analysis', action='store_true',
                             help='Run comparative analysis across all glaciers')
    
    # Analysis options
    parser.add_argument('--analysis-mode', type=str, choices=['auto', 'basic', 'enhanced', 'comprehensive'],
                       default='auto', help='Analysis mode (default: auto - determined by glacier type)')
    parser.add_argument('--selected-pixels', action='store_true',
                       help='Use optimally selected pixels (recommended for better accuracy)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to configuration file (default: config/config.yaml)')
    
    # Output options
    parser.add_argument('--output-summary', type=str,
                       help='Save analysis summary to specified CSV file')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Suppress verbose output (for automated scripts)')
    
    args = parser.parse_args()
    
    try:
        # If no command-line arguments for processing, run interactive mode
        if not any([args.glacier, args.all_glaciers, args.comparative_analysis]):
            print("Starting Interactive Mode...")
            print("(Use --help to see command-line options)\n")
            app = InteractiveGlacierAnalysis(args.config)
            app.run()
            return
        
        # Command-line mode
        if not args.quiet:
            print("MODIS Albedo Analysis Framework - Command Line Mode")
            print("=" * 60)
        
        app = InteractiveGlacierAnalysis(args.config)
        
        # Set up logging based on quiet flag
        if args.quiet:
            logging.getLogger().setLevel(logging.ERROR)
        else:
            logging.getLogger().setLevel(logging.INFO)
        
        results = []
        
        if args.glacier:
            # Process single glacier
            if not args.quiet:
                pixel_mode = "selected pixels" if args.selected_pixels else "all pixels"
                print(f"Processing glacier: {args.glacier}")
                print(f"Analysis mode: {args.analysis_mode}")
                print(f"Pixel selection: {pixel_mode}")
                print("-" * 40)
            
            # Check data availability
            available, missing_files = app.check_data_availability(args.glacier)
            if not available:
                print(f"ERROR: Cannot process {args.glacier} - missing data files:")
                for missing in missing_files:
                    print(f"  - {missing}")
                sys.exit(1)
            
            glacier_config = app.glacier_sites_config['glaciers'].get(args.glacier, {})
            glacier_name = glacier_config.get('name', args.glacier)
            
            success = app.process_single_glacier_with_pixel_selection(
                args.glacier, glacier_name, args.selected_pixels)
            
            if success:
                results.append({'glacier_id': args.glacier, 'status': 'success'})
                if not args.quiet:
                    print(f"\nSUCCESS: {glacier_name} processed successfully!")
            else:
                if not args.quiet:
                    print(f"\nERROR: Failed to process {glacier_name}")
                sys.exit(1)
        
        elif args.all_glaciers:
            # Process all available glaciers
            if not args.quiet:
                pixel_mode = "selected pixels" if args.selected_pixels else "all pixels"
                print(f"Processing all available glaciers")
                print(f"Analysis mode: {args.analysis_mode}")
                print(f"Pixel selection: {pixel_mode}")
                print("-" * 40)
            
            glacier_info = {}
            for i, (glacier_id, config) in enumerate(app.glacier_sites_config['glaciers'].items(), 1):
                available, missing_files = app.check_data_availability(glacier_id)
                glacier_info[str(i)] = {
                    'id': glacier_id,
                    'config': config,
                    'available': available,
                    'missing_files': missing_files
                }
            
            success_count = app.process_all_available_glaciers(glacier_info)
            
            if not args.quiet:
                total_available = sum(1 for info in glacier_info.values() if info['available'])
                print(f"\nCompleted: {success_count}/{total_available} glaciers processed successfully")
        
        elif args.comparative_analysis:
            # Run comparative analysis
            if not COMPARATIVE_ANALYSIS_AVAILABLE:
                print("ERROR: Comparative analysis is not available due to missing dependencies.")
                print("Please install required packages (sklearn, pyarrow, etc.) to enable this feature.")
                sys.exit(1)
            
            if not args.quiet:
                print("Starting comparative analysis across all glaciers...")
                print("-" * 40)
            
            try:
                comparative_interface = ComparativeAnalysisInterface()
                # For command-line mode, run automated comparative analysis
                if hasattr(comparative_interface, 'run_automated_analysis'):
                    comparative_interface.run_automated_analysis()
                else:
                    # Fallback to interactive if automated method doesn't exist
                    comparative_interface.run_interactive_session()
                
                if not args.quiet:
                    print("\nSUCCESS: Comparative analysis completed!")
                    
            except Exception as e:
                print(f"ERROR: Comparative analysis failed: {e}")
                sys.exit(1)
        
        # Save summary if requested
        if args.output_summary and results:
            import pandas as pd
            summary_df = pd.DataFrame(results)
            summary_df.to_csv(args.output_summary, index=False)
            if not args.quiet:
                print(f"Analysis summary saved to: {args.output_summary}")
        
        if not args.quiet:
            print("\nCommand-line analysis completed successfully!")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        sys.exit(0)
    except Exception as e:
        print(f"\nFatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()