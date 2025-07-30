#!/usr/bin/env python3
"""
Test enhanced spatial mapping with Coropuna glacier (has many pixels).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pivot_based_main import PivotBasedAlbedoAnalysis
import logging

def test_coropuna_mapping():
    """Test the enhanced spatial mapping functionality with Coropuna."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize analyzer
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        analyzer = PivotBasedAlbedoAnalysis(config_path)
        
        print("Testing enhanced spatial mapping with Coropuna glacier...")
        print("This glacier has many pixels - perfect for testing pixel highlighting!")
        print("-" * 70)
        
        # Process with best pixel selection
        result = analyzer.process_glacier('coropuna', use_selected_pixels=True)
        
        if result:
            print(f"\n[SUCCESS] Analysis with enhanced spatial mapping completed!")
            
            # Check for maps directory
            output_dir = result.get('output_directory', '')
            if output_dir and os.path.exists(output_dir):
                maps_dir = os.path.join(output_dir, 'maps')
                if os.path.exists(maps_dir):
                    map_files = [f for f in os.listdir(maps_dir) if f.endswith(('.png', '.jpg', '.pdf'))]
                    if map_files:
                        print(f"[VERIFIED] Found {len(map_files)} map files:")
                        for map_file in sorted(map_files):
                            print(f"    + {map_file}")
                        print(f"\nMap location: {maps_dir}")
                        print("\nThe enhanced map should show:")
                        print("- Light gray dots: All available pixels")
                        print("- Large blue circles: 2 selected pixels for analysis")
                        print("- Red star: AWS station location")
                        print("- Info box: Pixel selection details")
                    else:
                        print(f"[WARNING] Maps directory exists but contains no map files")
                else:
                    print(f"[WARNING] Maps directory was not created")
            else:
                print(f"[WARNING] Output directory not found")
        else:
            print("[FAILED] No results obtained")
            return False
            
    except Exception as e:
        print(f"[FAILED] {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_coropuna_mapping()
    sys.exit(0 if success else 1)