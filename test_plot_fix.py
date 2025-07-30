#!/usr/bin/env python3
"""Test script to verify the plotting fix."""

import sys
import os
import logging

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.config.helpers import load_config, setup_logging
from albedo_engine.engine import AlbedoAnalysisEngine

def test_plotting_fix():
    """Test the plotting fix."""
    
    # Load configuration
    config = load_config('config/config.yaml')
    setup_logging(config)
    
    logger = logging.getLogger(__name__)
    logger.info("Testing plotting fix...")
    
    try:
        # Initialize engine
        engine = AlbedoAnalysisEngine('config/config.yaml')
        
        # Test with Haig glacier (smaller dataset)
        logger.info("Running analysis for Haig glacier...")
        results = engine.process_glacier('haig', analysis_mode='enhanced')
        
        if results:
            logger.info(f"Analysis completed successfully!")
            logger.info(f"Results keys: {list(results.keys())}")
            
            # Check if plots were generated
            output_dir = results.get('output_directory', '')
            if output_dir:
                plots_dir = os.path.join(output_dir, 'plots')
                if os.path.exists(plots_dir):
                    plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
                    logger.info(f"Generated {len(plot_files)} plot files:")
                    for plot_file in sorted(plot_files):
                        logger.info(f"  - {plot_file}")
                else:
                    logger.error("Plots directory not found!")
            else:
                logger.error("No output directory in results!")
        else:
            logger.error("Analysis failed - no results returned")
            
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_plotting_fix()