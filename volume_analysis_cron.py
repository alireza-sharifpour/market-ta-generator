#!/usr/bin/env python3
"""
Cron job script for volume analysis.
Runs every 5 minutes to analyze top 200 trading pairs for suspicious volume activity.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from app.core.batch_volume_analyzer import BatchVolumeAnalyzer
from app.utils.logging_config import setup_logging

# Set up logging
setup_logging("INFO")
logger = logging.getLogger(__name__)


async def run_volume_analysis():
    """Run the batch volume analysis."""
    try:
        logger.info("🔄 Starting scheduled volume analysis...")
        
        # Initialize batch analyzer
        analyzer = BatchVolumeAnalyzer()
        
        # Run analysis
        summary = await analyzer.analyze_all_pairs()
        
        # Log summary
        batch_summary = summary["batch_analysis_summary"]
        logger.info(f"📊 Analysis Summary:")
        logger.info(f"  - Total pairs: {batch_summary['total_pairs']}")
        logger.info(f"  - Successful: {batch_summary['successful_pairs']}")
        logger.info(f"  - Failed: {batch_summary['failed_pairs']}")
        logger.info(f"  - Suspicious: {batch_summary['pairs_with_suspicious_periods']}")
        logger.info(f"  - Duration: {batch_summary['duration_seconds']:.2f}s")
        logger.info(f"  - Success rate: {batch_summary['success_rate']:.1f}%")
        
        # Log top suspicious pairs
        top_pairs = summary["top_suspicious_pairs"]
        if top_pairs:
            logger.info(f"🚨 Top suspicious pairs:")
            for i, pair in enumerate(top_pairs[:5], 1):
                logger.info(f"  {i}. {pair['pair'].upper()}: {pair['suspicious_periods']} periods (confidence: {pair['confidence_score']:.2%})")
        
        logger.info("✅ Scheduled volume analysis completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Volume analysis failed: {str(e)}", exc_info=True)
        return False


def main():
    """Main function for cron job execution."""
    try:
        # Run the async analysis
        success = asyncio.run(run_volume_analysis())
        
        if success:
            print("Volume analysis completed successfully")
            sys.exit(0)
        else:
            print("Volume analysis failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Volume analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
