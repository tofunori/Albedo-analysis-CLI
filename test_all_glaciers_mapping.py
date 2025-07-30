#!/usr/bin/env python3
"""
Test script to verify spatial mapping works for all glaciers.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pivot_based_main import PivotBasedAlbedoAnalysis
import logging

def test_all_glaciers_mapping():
    """Test the spatial mapping functionality for all glaciers."""
    
    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # List of glaciers to test
    glaciers_to_test = ['athabasca', 'haig', 'coropuna']
    results = {}
    
    try:
        # Initialize analyzer
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
        analyzer = PivotBasedAlbedoAnalysis(config_path)
        
        print("="*80)
        print("TESTING SPATIAL MAPPING FOR ALL GLACIERS")
        print("="*80)
        
        for glacier_id in glaciers_to_test:
            print(f"\n[TESTING] {glacier_id.upper()} GLACIER")
            print("-" * 50)
            
            try:
                # Process with best pixel selection
                result = analyzer.process_glacier(glacier_id, use_selected_pixels=True)
                
                if result:
                    print(f"[SUCCESS] Analysis completed for {glacier_id}")
                    
                    # Check for maps directory and files
                    output_dir = result.get('output_directory', '')
                    if output_dir and os.path.exists(output_dir):
                        maps_dir = os.path.join(output_dir, 'maps')
                        if os.path.exists(maps_dir):
                            map_files = [f for f in os.listdir(maps_dir) if f.endswith(('.png', '.jpg', '.pdf'))]
                            if map_files:
                                print(f"[VERIFIED] Found {len(map_files)} map files:")
                                for map_file in sorted(map_files):
                                    print(f"    + {map_file}")
                                results[glacier_id] = {'status': 'success', 'maps': len(map_files), 'files': map_files}
                            else:
                                print(f"[WARNING] Maps directory exists but contains no map files")
                                results[glacier_id] = {'status': 'no_maps', 'maps': 0, 'files': []}
                        else:
                            print(f"[WARNING] Maps directory was not created")
                            results[glacier_id] = {'status': 'no_maps_dir', 'maps': 0, 'files': []}
                    else:
                        print(f"[WARNING] Output directory not found")
                        results[glacier_id] = {'status': 'no_output_dir', 'maps': 0, 'files': []}
                else:
                    print(f"[FAILED] No results obtained for {glacier_id}")
                    results[glacier_id] = {'status': 'failed', 'maps': 0, 'files': []}
                    
            except Exception as e:
                print(f"[FAILED] Error processing {glacier_id}: {e}")
                results[glacier_id] = {'status': 'error', 'maps': 0, 'files': [], 'error': str(e)}
        
        # Summary report
        print("\n" + "="*80)
        print("SPATIAL MAPPING TEST SUMMARY")
        print("="*80)
        
        success_count = 0
        for glacier_id, result in results.items():
            status_icon = "+" if result['status'] == 'success' else "X"
            print(f"{status_icon} {glacier_id.upper()}: {result['status']} ({result['maps']} maps)")
            if result['status'] == 'success':
                success_count += 1
            elif result['status'] == 'error':
                print(f"    Error: {result.get('error', 'Unknown error')}")
        
        print(f"\nOVERALL RESULT: {success_count}/{len(glaciers_to_test)} glaciers successfully generated spatial maps")
        
        if success_count == len(glaciers_to_test):
            print("SUCCESS: ALL GLACIERS PASSED SPATIAL MAPPING TEST!")
            return True
        else:
            print("WARNING: Some glaciers failed spatial mapping test")
            return False
            
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        return False

if __name__ == "__main__":
    success = test_all_glaciers_mapping()
    sys.exit(0 if success else 1)