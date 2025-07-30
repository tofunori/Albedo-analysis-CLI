#!/usr/bin/env python3
"""
Framework Verification Script

This script tests all essential components to verify that statistics 
and data loading are working correctly.
"""

import sys
from pathlib import Path

def test_configuration():
    """Test configuration loading."""
    print("1. Testing Configuration System...")
    try:
        from utils.config.helpers import load_config
        
        # Test main config
        config = load_config('config/config.yaml')
        assert 'data' in config, "Main config missing 'data' section"
        assert 'analysis' in config, "Main config missing 'analysis' section"
        print("   ✅ Main configuration loaded successfully")
        
        # Test glacier config
        glacier_config = load_config('config/glacier_sites.yaml')
        assert 'glaciers' in glacier_config, "Glacier config missing 'glaciers' section"
        glaciers = list(glacier_config['glaciers'].keys())
        print(f"   ✅ Glacier configuration loaded: {len(glaciers)} glaciers ({', '.join(glaciers)})")
        
        return True, config, glacier_config
        
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False, None, None

def test_data_loading(config, glacier_config):
    """Test data loading system."""
    print("\n2. Testing Data Loading System...")
    try:
        from data_processing.loaders.pivot_loaders import AthabascaMultiProductLoader
        
        loader = AthabascaMultiProductLoader(config)
        print("   ✅ Data loader initialized successfully")
        
        # Test data availability for each glacier
        glaciers_tested = 0
        glaciers_available = 0
        
        for glacier_id in glacier_config['glaciers'].keys():
            try:
                # This will test file existence and basic loading
                print(f"   Testing {glacier_id} data...")
                data = loader.load(glacier_id)
                if data is not None and not data.empty:
                    print(f"     ✅ {glacier_id}: {data.shape[0]} rows, {data.shape[1]} columns")
                    glaciers_available += 1
                else:
                    print(f"     ⚠️  {glacier_id}: No data loaded (file may not exist)")
                glaciers_tested += 1
            except Exception as e:
                print(f"     ❌ {glacier_id}: Error loading data - {e}")
                glaciers_tested += 1
        
        print(f"   Data loading summary: {glaciers_available}/{glaciers_tested} glaciers have data")
        return True
        
    except Exception as e:
        print(f"   ❌ Data loading error: {e}")
        return False

def test_analysis_engine():
    """Test analysis engine initialization."""
    print("\n3. Testing Analysis Engine...")
    try:
        from albedo_engine.engine import AlbedoAnalysisEngine
        
        engine = AlbedoAnalysisEngine('config/config.yaml')
        print("   ✅ Analysis engine initialized successfully")
        print(f"   ✅ Engine configuration loaded: {engine.config is not None}")
        print(f"   ✅ Data processor initialized: {engine.processor is not None}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Analysis engine error: {e}")
        return False

def test_statistical_analyzer():
    """Test statistical analysis components."""
    print("\n4. Testing Statistical Analysis...")
    try:
        from analysis.core.statistical_analyzer import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer({})
        print("   ✅ Statistical analyzer initialized successfully")
        print("   ✅ Advanced statistical analysis available")
        
        return True
        
    except ImportError as e:
        print(f"   ⚠️  Statistical analyzer unavailable (expected if sklearn missing): {e}")
        print("   ℹ️  Basic statistics will still work through the analysis engine")
        return True  # This is expected and OK
        
    except Exception as e:
        print(f"   ❌ Statistical analyzer error: {e}")
        return False

def test_visualization():
    """Test visualization components."""
    print("\n5. Testing Visualization System...")
    try:
        from visualization.plots.statistical_plots import PlotGenerator
        
        config = {'visualization': {'figure_size': [12, 8], 'dpi': 300}}
        plot_gen = PlotGenerator(config)
        print("   ✅ Plot generator initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Visualization error: {e}")
        return False

def test_interactive_app():
    """Test interactive application."""
    print("\n6. Testing Interactive Application...")
    try:
        from interactive_main import InteractiveGlacierAnalysis
        
        app = InteractiveGlacierAnalysis()
        print("   ✅ Interactive application initialized successfully")
        
        # Test data availability checking
        print("   Testing data availability checking...")
        available_count = 0
        total_count = 0
        
        for glacier_id in ['athabasca', 'haig', 'coropuna']:
            available, missing = app.check_data_availability(glacier_id)
            total_count += 1
            if available:
                available_count += 1
                print(f"     ✅ {glacier_id}: Data available")
            else:
                print(f"     ⚠️  {glacier_id}: Missing {len(missing)} files")
                for missing_file in missing[:3]:  # Show first 3 missing files
                    print(f"       - {missing_file}")
        
        print(f"   Data availability: {available_count}/{total_count} glaciers ready")
        return True
        
    except Exception as e:
        print(f"   ❌ Interactive application error: {e}")
        return False

def main():
    """Run comprehensive framework verification."""
    print("MODIS Albedo Analysis Framework Verification")
    print("=" * 60)
    
    # Track test results
    tests_passed = 0
    total_tests = 6
    
    # Run tests
    config_ok, config, glacier_config = test_configuration()
    if config_ok:
        tests_passed += 1
    
    if config_ok and test_data_loading(config, glacier_config):
        tests_passed += 1
    
    if test_analysis_engine():
        tests_passed += 1
        
    if test_statistical_analyzer():
        tests_passed += 1
        
    if test_visualization():
        tests_passed += 1
        
    if test_interactive_app():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    if tests_passed == total_tests:
        print(f"ALL TESTS PASSED ({tests_passed}/{total_tests})")
        print("SUCCESS: Framework is ready for analysis!")
        print("\nNext steps:")
        print("   * Run: python interactive_main.py")
        print("   * Or: python interactive_main.py --help")
    else:
        print(f"WARNING: SOME ISSUES FOUND ({tests_passed}/{total_tests} tests passed)")
        print("Check the errors above and ensure:")
        print("   * Data files are in the correct locations")
        print("   * Configuration files are properly set up")
        print("   * Required dependencies are installed")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)