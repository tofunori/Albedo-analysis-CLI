#!/usr/bin/env python3
"""
Test script to verify spatial mapping works through interactive interface.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from interactive_main import InteractiveGlacierAnalysis
import logging

def test_interactive_mapping():
    """Test the spatial mapping functionality through interactive interface."""
    
    # Set up logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise for this test
    
    try:
        # Initialize interactive analyzer
        app = InteractiveGlacierAnalysis()
        
        print("Testing interactive interface spatial mapping...")
        print("Processing Haig glacier with pixel selection...")
        
        # Test processing a single glacier with pixel selection (this should generate maps)
        success = app.process_single_glacier_with_pixel_selection('haig', 'Haig Glacier', use_selected_pixels=True)
        
        if success:
            print("\n[SUCCESS] Interactive analysis completed!")
            return True
        else:
            print("\n[FAILED] Interactive analysis failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] {e}")
        return False

if __name__ == "__main__":
    success = test_interactive_mapping()
    sys.exit(0 if success else 1)