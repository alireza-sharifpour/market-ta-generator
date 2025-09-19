#!/usr/bin/env python3
"""
Test script to run the volume analyzer once for testing purposes.
This uses the manual analysis method from the scheduler.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.volume_scheduler import VolumeAnalysisScheduler
from app.utils.logging_config import setup_logging

# Set up logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


async def run_single_analysis():
    """Run volume analysis once for testing."""
    
    print("🚀 Running Volume Analyzer - Single Test Run")
    print("=" * 60)
    
    try:
        # Initialize the scheduler (but don't start the scheduled job)
        scheduler = VolumeAnalysisScheduler()
        
        print("📊 Starting manual volume analysis...")
        
        # Run the analysis once manually
        result = await scheduler.run_manual_analysis()
        
        print("=" * 60)
        if result["success"]:
            print("✅ Volume analysis completed successfully!")
            print(f"⏰ Timestamp: {result['timestamp']}")
            print(f"🔧 Type: {result['type']}")
        else:
            print("❌ Volume analysis failed!")
        
        print("=" * 60)
        return result["success"]
        
    except Exception as e:
        print(f"❌ Error during volume analysis: {str(e)}")
        logger.error(f"Volume analysis failed: {str(e)}", exc_info=True)
        return False


async def main():
    """Main function."""
    success = await run_single_analysis()
    
    if success:
        print("🎉 Test completed successfully!")
        return 0
    else:
        print("💥 Test failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
