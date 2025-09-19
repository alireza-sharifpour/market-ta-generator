"""
Volume Analysis Scheduler using APScheduler.
Provides Python-based scheduling for volume analysis tasks.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.asyncio import AsyncIOExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from app.core.batch_volume_analyzer import BatchVolumeAnalyzer
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class VolumeAnalysisScheduler:
    """Python-based scheduler for volume analysis using APScheduler."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the volume analysis scheduler."""
        self.config = config or {}
        self.scheduler = None
        self.batch_analyzer = None
        self.is_running = False
        
        # Default configuration
        self.schedule_config = {
            "interval_minutes": 5,  # Run every 5 minutes
            "max_instances": 1,     # Only one instance at a time
            "coalesce": True,       # Combine missed runs
            "misfire_grace_time": 60,  # 1 minute grace time
            "replace_existing": True,
        }
        
        # Merge with provided config
        self.schedule_config.update(self.config.get("scheduler", {}))
        
        logger.info("VolumeAnalysisScheduler initialized")
    
    def _setup_scheduler(self):
        """Setup the APScheduler with job stores and executors."""
        jobstores = {
            'default': MemoryJobStore()
        }
        
        executors = {
            'default': AsyncIOExecutor()
        }
        
        job_defaults = {
            'coalesce': self.schedule_config["coalesce"],
            'max_instances': self.schedule_config["max_instances"],
            'misfire_grace_time': self.schedule_config["misfire_grace_time"],
        }
        
        self.scheduler = AsyncIOScheduler(
            jobstores=jobstores,
            executors=executors,
            job_defaults=job_defaults,
            timezone='UTC'
        )
        
        # Add event listeners for monitoring
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
        
        logger.info("APScheduler configured with job stores and executors")
    
    def _job_executed(self, event):
        """Handle successful job execution."""
        logger.info(f"Volume analysis job executed successfully: {event.job_id}")
    
    def _job_error(self, event):
        """Handle job execution errors."""
        logger.error(f"Volume analysis job failed: {event.job_id}, Exception: {event.exception}")
    
    async def _run_volume_analysis(self):
        """Execute the volume analysis batch job."""
        try:
            logger.info("🔄 Starting scheduled volume analysis...")
            
            if not self.batch_analyzer:
                self.batch_analyzer = BatchVolumeAnalyzer()
            
            # Run analysis
            summary = await self.batch_analyzer.analyze_all_pairs()
            
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
    
    def start_scheduler(self, schedule_type: str = "interval"):
        """
        Start the volume analysis scheduler.
        
        Args:
            schedule_type: "interval" for every N minutes, "cron" for cron-like scheduling
        """
        if self.is_running:
            logger.warning("Scheduler is already running")
            return
        
        self._setup_scheduler()
        
        if schedule_type == "interval":
            # Run every N minutes
            trigger = IntervalTrigger(
                minutes=self.schedule_config["interval_minutes"]
            )
            logger.info(f"Starting interval scheduler: every {self.schedule_config['interval_minutes']} minutes")
            
        elif schedule_type == "cron":
            # Run every 5 minutes (cron-like)
            trigger = CronTrigger(
                minute="*/5",  # Every 5 minutes
                second=0       # At the start of the minute
            )
            logger.info("Starting cron scheduler: every 5 minutes")
            
        else:
            raise ValueError(f"Invalid schedule_type: {schedule_type}. Use 'interval' or 'cron'")
        
        # Add the job
        self.scheduler.add_job(
            func=self._run_volume_analysis,
            trigger=trigger,
            id='volume_analysis_job',
            name='Volume Analysis Batch Job',
            replace_existing=self.schedule_config["replace_existing"]
        )
        
        # Start the scheduler
        self.scheduler.start()
        self.is_running = True
        
        logger.info("✅ Volume analysis scheduler started successfully")
    
    def stop_scheduler(self):
        """Stop the volume analysis scheduler."""
        if not self.is_running:
            logger.warning("Scheduler is not running")
            return
        
        if self.scheduler:
            self.scheduler.shutdown(wait=True)
            self.is_running = False
            logger.info("✅ Volume analysis scheduler stopped")
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Get current scheduler status and job information."""
        if not self.scheduler or not self.is_running:
            return {
                "status": "stopped",
                "is_running": False,
                "jobs": []
            }
        
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })
        
        return {
            "status": "running",
            "is_running": True,
            "jobs": jobs,
            "scheduler_config": self.schedule_config
        }
    
    async def run_manual_analysis(self) -> Dict[str, Any]:
        """Run volume analysis manually (outside of schedule)."""
        logger.info("🔄 Running manual volume analysis...")
        success = await self._run_volume_analysis()
        
        return {
            "success": success,
            "timestamp": datetime.now().isoformat(),
            "type": "manual"
        }
    
    def update_schedule(self, new_config: Dict[str, Any]):
        """Update scheduler configuration and restart if needed."""
        logger.info(f"Updating scheduler configuration: {new_config}")
        
        # Update config
        self.schedule_config.update(new_config)
        
        # If scheduler is running, restart it with new config
        if self.is_running:
            logger.info("Restarting scheduler with new configuration...")
            self.stop_scheduler()
            self.start_scheduler()
        else:
            logger.info("Scheduler not running, configuration updated for next start")
