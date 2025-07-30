#!/usr/bin/env python3
"""
Test script to verify that spatial mapping is now working.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pivot_based_main import PivotBasedAlbedoAnalysis
import logging

def test_mapping():
    """Test the spatial mapping functionality."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize analyzer
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        analyzer = PivotBasedAlbedoAnalysis(config_path)
        
        print("Testing spatial mapping functionality...")
        print("Processing Haig glacier with best pixel selection...")
        
        # Process with best pixel selection
        result = analyzer.process_glacier('haig', use_selected_pixels=True)
        
        if result:
            print("\n[SUCCESS] Analysis with spatial mapping completed!")
            
            # Check for maps directory
            output_dir = result.get('output_directory', '')
            if output_dir and os.path.exists(output_dir):
                maps_dir = os.path.join(output_dir, 'maps')
                if os.path.exists(maps_dir):
                    map_files = [f for f in os.listdir(maps_dir) if f.endswith(('.png', '.jpg', '.pdf'))]
                    if map_files:
                        print(f"\n[VERIFIED] Found {len(map_files)} map files in maps directory:")
                        for map_file in sorted(map_files):
                            print(f"    {map_file}")
                    else:
                        print("\n[WARNING] Maps directory exists but contains no map files")
                else:
                    print("\n[WARNING] Maps directory was not created")
            else:
                print("\n[WARNING] Output directory not found")
        else:
            print("[FAILED] No results obtained")
            return False
            
    except Exception as e:
        print(f"[FAILED] {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_mapping()
    sys.exit(0 if success else 1)