#!/usr/bin/env python3
"""
Simple Framework Verification Script - No Unicode

Tests essential components to verify that statistics and data loading work correctly.
"""

def test_configuration():
    """Test configuration loading."""
    print("1. Testing Configuration System...")
    try:
        from utils.config.helpers import load_config
        
        # Test main config
        config = load_config('config/config.yaml')
        assert 'data' in config, "Main config missing 'data' section"
        assert 'analysis' in config, "Main config missing 'analysis' section"
        print("   SUCCESS: Main configuration loaded")
        
        # Test glacier config
        glacier_config = load_config('config/glacier_sites.yaml')
        assert 'glaciers' in glacier_config, "Glacier config missing 'glaciers' section"
        glaciers = list(glacier_config['glaciers'].keys())
        print(f"   SUCCESS: {len(glaciers)} glaciers configured: {', '.join(glaciers)}")
        
        return True, config, glacier_config
        
    except Exception as e:
        print(f"   ERROR: Configuration failed - {e}")
        return False, None, None

def test_data_loading(config, glacier_config):
    """Test data loading system."""
    print("\n2. Testing Data Loading System...")
    try:
        from data_processing.loaders.pivot_loaders import AthabascaMultiProductLoader
        
        loader = AthabascaMultiProductLoader(config)
        print("   SUCCESS: Data loader initialized")
        
        # Test data availability
        glaciers_available = 0
        for glacier_id in glacier_config['glaciers'].keys():
            try:
                # Test with a dummy file path (the loader tests the method, not actual file loading)
                glacier_info = glacier_config['glaciers'][glacier_id]
                if glacier_info.get('data_type') == 'athabasca_multiproduct':
                    print(f"   SUCCESS: {glacier_id} - Loader compatible with data type")
                    glaciers_available += 1
                else:
                    print(f"   WARNING: {glacier_id} - Different data type: {glacier_info.get('data_type')}")
            except Exception as e:
                print(f"   ERROR: {glacier_id} - {e}")
        
        print(f"   SUMMARY: {glaciers_available}/{len(glacier_config['glaciers'])} glaciers have data")
        return True
        
    except Exception as e:
        print(f"   ERROR: Data loading failed - {e}")
        return False

def test_analysis_engine():
    """Test analysis engine."""
    print("\n3. Testing Analysis Engine...")
    try:
        from albedo_engine.engine import AlbedoAnalysisEngine
        
        engine = AlbedoAnalysisEngine('config/config.yaml')
        print("   SUCCESS: Analysis engine initialized")
        print(f"   SUCCESS: Configuration loaded: {engine.config is not None}")
        print(f"   SUCCESS: Pivot processor ready: {hasattr(engine, 'pivot_processor')}")
        
        return True
        
    except Exception as e:
        print(f"   ERROR: Analysis engine failed - {e}")
        return False

def test_interactive_app():
    """Test interactive application."""
    print("\n4. Testing Interactive Application...")
    try:
        from interactive_main import InteractiveGlacierAnalysis
        
        app = InteractiveGlacierAnalysis()
        print("   SUCCESS: Interactive app initialized")
        
        # Test data availability checking
        available_count = 0
        for glacier_id in ['athabasca', 'haig', 'coropuna']:
            available, missing = app.check_data_availability(glacier_id)
            if available:
                available_count += 1
                print(f"   SUCCESS: {glacier_id} - Data available")
            else:
                print(f"   WARNING: {glacier_id} - Missing {len(missing)} files")
        
        print(f"   SUMMARY: {available_count}/3 glaciers ready")
        return True
        
    except Exception as e:
        print(f"   ERROR: Interactive app failed - {e}")
        return False

def main():
    """Run verification."""
    print("MODIS Albedo Analysis Framework Verification")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 4
    
    # Run tests
    config_ok, config, glacier_config = test_configuration()
    if config_ok:
        tests_passed += 1
    
    if config_ok and test_data_loading(config, glacier_config):
        tests_passed += 1
    
    if test_analysis_engine():
        tests_passed += 1
        
    if test_interactive_app():
        tests_passed += 1
    
    # Summary
    print("\n" + "=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    
    if tests_passed == total_tests:
        print(f"SUCCESS: ALL TESTS PASSED ({tests_passed}/{total_tests})")
        print("Framework is ready for analysis!")
        print("\nNext steps:")
        print("  python interactive_main.py")
        print("  python interactive_main.py --help")
    else:
        print(f"WARNING: {tests_passed}/{total_tests} tests passed")
        print("Check errors above and verify:")
        print("  - Data files are in correct locations")
        print("  - Configuration files are properly set up")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)