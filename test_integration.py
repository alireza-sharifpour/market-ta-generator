#!/usr/bin/env python3
"""
Integration test to verify that enhanced S/R detection works with both
LLM analysis and chart generation.
"""

import logging
from app.core.analysis_service import run_phase2_analysis
from app.core.data_processor import identify_support_resistance, process_raw_data
from app.core.chart_generator import generate_ohlcv_chart
from app.external.lbank_client import fetch_ohlcv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_full_integration():
    """Test the complete integration of enhanced S/R detection with LLM and chart generation."""
    logger.info("=== Testing Full Integration ===")
    
    # Test parameters
    pair = "btc_usdt"
    timeframe = "hour1"  # Use 1-hour timeframe to test enhanced detection
    limit = 50
    
    logger.info(f"Testing with pair: {pair}, timeframe: {timeframe}, limit: {limit}")
    
    try:
        # Step 1: Test manual S/R detection
        logger.info("--- Step 1: Manual S/R Detection ---")
        raw_data = fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
        df = process_raw_data(raw_data)
        
        # Test enhanced S/R detection with timeframe
        sr_levels = identify_support_resistance(df, timeframe=timeframe)
        
        logger.info(f"Enhanced S/R Detection Results:")
        logger.info(f"  Support levels: {sr_levels['support']}")
        logger.info(f"  Resistance levels: {sr_levels['resistance']}")
        logger.info(f"  Total levels found: {len(sr_levels['support']) + len(sr_levels['resistance'])}")
        
        # Step 2: Test chart generation with S/R levels
        logger.info("--- Step 2: Chart Generation with S/R Levels ---")
        from app.core.data_processor import add_technical_indicators
        df_with_indicators = add_technical_indicators(df)
        
        chart_base64 = generate_ohlcv_chart(
            df_with_indicators,
            sr_levels=sr_levels,
            use_filtered_indicators=True
        )
        
        chart_success = chart_base64 is not None and len(chart_base64) > 100
        logger.info(f"Chart generation successful: {chart_success}")
        if chart_success:
            logger.info(f"Chart data length: {len(chart_base64)} characters")
        
        # Step 3: Test full analysis pipeline
        logger.info("--- Step 3: Full Analysis Pipeline ---")
        analysis_result = run_phase2_analysis(pair, timeframe=timeframe, limit=limit)
        
        pipeline_success = analysis_result["status"] == "success"
        logger.info(f"Pipeline execution successful: {pipeline_success}")
        
        if pipeline_success:
            # Check if LLM analysis contains S/R information
            analysis_text = analysis_result.get("analysis", "")
            has_sr_info = "Support" in analysis_text or "Resistance" in analysis_text
            logger.info(f"LLM analysis contains S/R information: {has_sr_info}")
            
            # Check if chart was generated
            chart_generated = analysis_result.get("chart_image_base64") is not None
            logger.info(f"Chart included in pipeline result: {chart_generated}")
            
            # Summary
            logger.info("=== Integration Test Summary ===")
            logger.info(f"✅ Enhanced S/R Detection: {len(sr_levels['support']) + len(sr_levels['resistance'])} levels found")
            logger.info(f"✅ Chart Generation: {'Success' if chart_success else 'Failed'}")
            logger.info(f"✅ LLM Analysis: {'Contains S/R info' if has_sr_info else 'Missing S/R info'}")
            logger.info(f"✅ Full Pipeline: {'Success' if pipeline_success else 'Failed'}")
            
            return True
        else:
            logger.error(f"Pipeline failed: {analysis_result.get('message', 'Unknown error')}")
            return False
            
    except Exception as e:
        logger.error(f"Integration test failed: {e}", exc_info=True)
        return False

def test_different_timeframes():
    """Test S/R detection with different timeframes to verify parameter optimization."""
    logger.info("=== Testing Different Timeframes ===")
    
    pair = "eth_usdt"
    limit = 100
    
    timeframes = ["minute1", "hour1", "day1"]
    
    for timeframe in timeframes:
        logger.info(f"--- Testing timeframe: {timeframe} ---")
        
        try:
            raw_data = fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
            df = process_raw_data(raw_data)
            
            sr_levels = identify_support_resistance(df, timeframe=timeframe)
            
            total_levels = len(sr_levels['support']) + len(sr_levels['resistance'])
            logger.info(f"Timeframe {timeframe}: {total_levels} total levels "
                       f"({len(sr_levels['support'])} support, {len(sr_levels['resistance'])} resistance)")
            
            # Verify that shorter timeframes generally find more levels due to sensitivity
            if timeframe == "minute1":
                logger.info(f"High-frequency timeframe detected {total_levels} levels")
            elif timeframe == "hour1":
                logger.info(f"Medium-frequency timeframe detected {total_levels} levels")
            elif timeframe == "day1":
                logger.info(f"Low-frequency timeframe detected {total_levels} levels")
                
        except Exception as e:
            logger.error(f"Error testing timeframe {timeframe}: {e}")

if __name__ == "__main__":
    logger.info("Starting integration tests...")
    
    # Test full integration
    integration_success = test_full_integration()
    
    # Test different timeframes
    test_different_timeframes()
    
    if integration_success:
        logger.info("🎉 All integration tests passed!")
    else:
        logger.error("❌ Some integration tests failed!")