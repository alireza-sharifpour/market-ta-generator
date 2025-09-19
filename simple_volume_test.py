#!/usr/bin/env python3
"""
Simple test to verify volume analysis components work correctly.
"""

import sys
sys.path.append('.')

def test_imports():
    """Test that all volume analysis components can be imported."""
    try:
        from app.core.volume_analyzer import VolumeAnalyzer, VolumeAnalysisResult
        from app.core.volume_chart_generator import VolumeChartGenerator
        from app.api.models import VolumeAnalysisRequest, VolumeAnalysisResponse
        from app.config import VOLUME_ANALYSIS_CONFIG, VOLUME_CHART_CONFIG
        
        print("✅ All volume analysis imports successful:")
        print("  - VolumeAnalyzer")
        print("  - VolumeChartGenerator") 
        print("  - API models")
        print("  - Configuration")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config():
    """Test volume analysis configuration."""
    try:
        from app.config import VOLUME_ANALYSIS_CONFIG, VOLUME_CHART_CONFIG
        
        print("✅ Configuration loaded successfully:")
        print(f"  - Mean+Std enabled: {VOLUME_ANALYSIS_CONFIG['enable_mean_std_detection']}")
        print(f"  - Lookback period: {VOLUME_ANALYSIS_CONFIG['mean_std_lookback_period']}")
        print(f"  - Multiplier: {VOLUME_ANALYSIS_CONFIG['mean_std_multiplier']}")
        print(f"  - Chart template: {VOLUME_CHART_CONFIG['template']}")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_analyzer_init():
    """Test volume analyzer initialization."""
    try:
        from app.core.volume_analyzer import VolumeAnalyzer
        
        analyzer = VolumeAnalyzer()
        print("✅ VolumeAnalyzer initialized successfully")
        print(f"  - Config loaded: {analyzer.config is not None}")
        print(f"  - Mean+Std detection: {analyzer.config.get('enable_mean_std_detection', False)}")
        return True
        
    except Exception as e:
        print(f"❌ Analyzer initialization error: {e}")
        return False

def test_chart_generator_init():
    """Test chart generator initialization."""
    try:
        from app.core.volume_chart_generator import VolumeChartGenerator
        
        generator = VolumeChartGenerator()
        print("✅ VolumeChartGenerator initialized successfully")
        print(f"  - Config loaded: {generator.config is not None}")
        print(f"  - Chart width: {generator.config.get('width', 'unknown')}")
        return True
        
    except Exception as e:
        print(f"❌ Chart generator initialization error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Market TA Generator - Volume Analysis Component Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config), 
        ("Analyzer Initialization", test_analyzer_init),
        ("Chart Generator Initialization", test_chart_generator_init),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY:")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, _) in enumerate(tests):
        status = "✅ PASS" if results[i] else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Volume analysis components are working.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit(main())
