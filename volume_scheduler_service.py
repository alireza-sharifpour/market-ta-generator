#!/usr/bin/env python3
"""
Standalone Volume Analysis Scheduler Service.
Runs the volume analysis scheduler as a standalone Python service.
"""

import asyncio
import logging
import signal
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


class VolumeSchedulerService:
    """Standalone service for volume analysis scheduling."""
    
    def __init__(self):
        self.scheduler = VolumeAnalysisScheduler()
        self.shutdown_event = asyncio.Event()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.shutdown_event.set()
    
    async def start_service(self, schedule_type: str = "interval"):
        """Start the volume analysis scheduler service."""
        try:
            logger.info("🚀 Starting Volume Analysis Scheduler Service")
            logger.info(f"Schedule type: {schedule_type}")
            
            # Start the scheduler
            self.scheduler.start_scheduler(schedule_type)
            
            # Log initial status
            status = self.scheduler.get_scheduler_status()
            logger.info(f"Scheduler status: {status}")
            
            # Show next run time in IRST
            if status.get("jobs"):
                for job in status["jobs"]:
                    if job.get("next_run_time"):
                        from datetime import datetime
                        import pytz
                        next_run = datetime.fromisoformat(job["next_run_time"].replace('Z', '+00:00'))
                        irst_tz = pytz.timezone('Asia/Tehran')
                        next_run_irst = next_run.astimezone(irst_tz)
                        logger.info(f"⏰ Next analysis run: {next_run_irst.strftime('%Y-%m-%d %H:%M:%S IRST')}")
            
            # Wait for shutdown signal
            logger.info("Service running... Press Ctrl+C to stop")
            await self.shutdown_event.wait()
            
        except Exception as e:
            logger.error(f"Service error: {str(e)}", exc_info=True)
            raise
        finally:
            await self.stop_service()
    
    async def stop_service(self):
        """Stop the volume analysis scheduler service."""
        try:
            logger.info("🛑 Stopping Volume Analysis Scheduler Service")
            
            if self.scheduler.is_running:
                self.scheduler.stop_scheduler()
            
            logger.info("✅ Service stopped gracefully")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {str(e)}", exc_info=True)


async def main():
    """Main function for the scheduler service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Volume Analysis Scheduler Service")
    parser.add_argument(
        "--schedule-type",
        choices=["interval", "cron"],
        default="interval",
        help="Schedule type: interval (every N minutes) or cron (cron-like)"
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=5,
        help="Interval in minutes for interval schedule type"
    )
    
    args = parser.parse_args()
    
    # Create service
    service = VolumeSchedulerService()
    
    # Update scheduler config if needed
    if args.schedule_type == "interval":
        service.scheduler.schedule_config["interval_minutes"] = args.interval_minutes
    
    try:
        # Start the service
        await service.start_service(args.schedule_type)
        
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
    except Exception as e:
        logger.error(f"Service failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
