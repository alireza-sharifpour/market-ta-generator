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
from app.core.telegram_notifier import TelegramNotifier
from app.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class VolumeAnalysisScheduler:
    """Python-based scheduler for volume analysis using APScheduler with multiple timeframe support."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the volume analysis scheduler."""
        self.config = config or {}
        self.scheduler = None
        self.batch_analyzer = None
        self.telegram_notifier = TelegramNotifier()
        self.is_running = False
        
        # Available timeframes and their configurations
        self.available_timeframes = {
            "5m": {"interval_minutes": 5, "display_name": "5 Minutes"},
            "1h": {"interval_minutes": 60, "display_name": "1 Hour"},
            "4h": {"interval_minutes": 240, "display_name": "4 Hours"},
            "1d": {"interval_minutes": 1440, "display_name": "1 Day"},
        }
        
        # Track active jobs by timeframe
        self.active_jobs = {}
        
        # Default configuration
        self.schedule_config = {
            "max_instances": 1,     # Only one instance at a time
            "coalesce": True,       # Combine missed runs
            "misfire_grace_time": 60,  # 1 minute grace time
            "replace_existing": True,
        }
        
        # Merge with provided config
        self.schedule_config.update(self.config.get("scheduler", {}))
        
        logger.info("VolumeAnalysisScheduler initialized with multi-timeframe support")
    
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
            timezone='Asia/Tehran'  # IRST timezone
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
            
            # Send Telegram notification
            await self.telegram_notifier.send_batch_summary(summary)
            
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
            # Run every N minutes using cron for precise timing
            interval_minutes = self.schedule_config["interval_minutes"]
            trigger = CronTrigger(
                minute=f"*/{interval_minutes}",  # Every N minutes
                second=0,                        # At the start of the minute
                timezone='Asia/Tehran'           # IRST timezone
            )
            logger.info(f"Starting interval scheduler: every {interval_minutes} minutes on IRST timezone (Asia/Tehran)")
            
        elif schedule_type == "cron":
            # Run every 5 minutes (cron-like) on IRST
            trigger = CronTrigger(
                minute="*/5",                    # Every 5 minutes
                second=0,                        # At the start of the minute
                timezone='Asia/Tehran'           # IRST timezone
            )
            logger.info("Starting cron scheduler: every 5 minutes on IRST timezone (Asia/Tehran)")
            
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
    
    def start_timeframe_job(self, timeframe: str):
        """
        Start a volume analysis job for a specific timeframe.
        
        Args:
            timeframe: One of the available timeframes (minute5, minute15, hour1, etc.)
        """
        if timeframe not in self.available_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}. Available: {list(self.available_timeframes.keys())}")
        
        # Setup scheduler if not already running
        if not self.scheduler:
            self._setup_scheduler()
            self.scheduler.start()
            self.is_running = True
        
        # Check if job already exists
        job_id = f"volume_analysis_{timeframe}"
        if job_id in self.active_jobs:
            logger.warning(f"Job for timeframe {timeframe} is already running")
            return
        
        # Get timeframe config
        tf_config = self.available_timeframes[timeframe]
        interval_minutes = tf_config["interval_minutes"]
        
        # Create cron trigger for the specific interval
        from apscheduler.triggers.cron import CronTrigger
        
        # Create appropriate cron expression based on interval
        if interval_minutes < 60:
            # Minutes: */5, */15, */30
            trigger = CronTrigger(
                minute=f"*/{interval_minutes}",
                second=0,
                timezone='Asia/Tehran'
            )
        elif interval_minutes == 60:
            # Every hour at minute 0
            trigger = CronTrigger(
                minute=0,
                second=0,
                timezone='Asia/Tehran'
            )
        elif interval_minutes < 1440:
            # Hours: */4, */12
            hour_interval = interval_minutes // 60
            trigger = CronTrigger(
                hour=f"*/{hour_interval}",
                minute=0,
                second=0,
                timezone='Asia/Tehran'
            )
        else:
            # Daily
            trigger = CronTrigger(
                hour=0,
                minute=0,
                second=0,
                timezone='Asia/Tehran'
            )
        
        # Add the job
        self.scheduler.add_job(
            func=self._run_volume_analysis_with_timeframe,
            trigger=trigger,
            args=[timeframe],
            id=job_id,
            name=f'Volume Analysis - {tf_config["display_name"]}',
            replace_existing=self.schedule_config["replace_existing"],
            max_instances=self.schedule_config["max_instances"],
            coalesce=self.schedule_config["coalesce"],
            misfire_grace_time=self.schedule_config["misfire_grace_time"]
        )
        
        # Track the job
        self.active_jobs[job_id] = {
            "timeframe": timeframe,
            "interval_minutes": interval_minutes,
            "display_name": tf_config["display_name"],
            "job_id": job_id
        }
        
        logger.info(f"✅ Started {tf_config['display_name']} analysis job (every {interval_minutes} minutes)")
    
    def stop_timeframe_job(self, timeframe: str):
        """
        Stop a volume analysis job for a specific timeframe.
        
        Args:
            timeframe: The timeframe to stop
        """
        job_id = f"volume_analysis_{timeframe}"
        
        if job_id not in self.active_jobs:
            logger.warning(f"No active job found for timeframe: {timeframe}")
            return
        
        # Remove the job
        if self.scheduler:
            try:
                self.scheduler.remove_job(job_id)
                logger.info(f"✅ Stopped {self.active_jobs[job_id]['display_name']} analysis job")
            except Exception as e:
                logger.error(f"Error stopping job {job_id}: {e}")
        
        # Remove from tracking
        del self.active_jobs[job_id]
    
    def stop_all_jobs(self):
        """Stop all active timeframe jobs."""
        if not self.active_jobs:
            logger.info("No active jobs to stop")
            return
        
        jobs_to_stop = list(self.active_jobs.keys())
        for job_id in jobs_to_stop:
            timeframe = self.active_jobs[job_id]["timeframe"]
            self.stop_timeframe_job(timeframe)
        
        logger.info("✅ All volume analysis jobs stopped")
    
    def get_active_jobs(self) -> Dict[str, Any]:
        """Get information about all active jobs."""
        active_jobs_info = {}
        
        for job_id, job_info in self.active_jobs.items():
            # Get next run time from scheduler
            next_run = None
            if self.scheduler:
                try:
                    job = self.scheduler.get_job(job_id)
                    if job and job.next_run_time:
                        next_run = job.next_run_time.isoformat()
                except Exception:
                    pass
            
            active_jobs_info[job_info["timeframe"]] = {
                "display_name": job_info["display_name"],
                "interval_minutes": job_info["interval_minutes"],
                "next_run_time": next_run,
                "job_id": job_id,
                "status": "running"
            }
        
        return active_jobs_info
    
    def get_available_timeframes(self) -> Dict[str, Any]:
        """Get all available timeframes and their configurations."""
        return self.available_timeframes.copy()
    
    async def _run_volume_analysis_with_timeframe(self, timeframe: str):
        """Execute volume analysis for a specific timeframe."""
        try:
            tf_config = self.available_timeframes[timeframe]
            logger.info(f"🔄 Starting {tf_config['display_name']} volume analysis...")
            
            if not self.batch_analyzer:
                self.batch_analyzer = BatchVolumeAnalyzer()
            
            # Run analysis
            summary = await self.batch_analyzer.analyze_all_pairs()
            
            # Log summary with timeframe info
            batch_summary = summary["batch_analysis_summary"]
            logger.info(f"📊 {tf_config['display_name']} Analysis Summary:")
            logger.info(f"  - Total pairs: {batch_summary['total_pairs']}")
            logger.info(f"  - Successful: {batch_summary['successful_pairs']}")
            logger.info(f"  - Failed: {batch_summary['failed_pairs']}")
            logger.info(f"  - Suspicious: {batch_summary['pairs_with_suspicious_periods']}")
            logger.info(f"  - Duration: {batch_summary['duration_seconds']:.2f}s")
            logger.info(f"  - Success rate: {batch_summary['success_rate']:.1f}%")
            
            # Log top suspicious pairs
            top_pairs = summary["top_suspicious_pairs"]
            if top_pairs:
                logger.info(f"🚨 {tf_config['display_name']} Top suspicious pairs:")
                for i, pair in enumerate(top_pairs[:5], 1):
                    logger.info(f"  {i}. {pair['pair'].upper()}: {pair['suspicious_periods']} periods (confidence: {pair['confidence_score']:.2%})")
            
            # Send Telegram notification with timeframe info
            await self.telegram_notifier.send_batch_summary(summary, timeframe_info=tf_config)
            
            logger.info(f"✅ {tf_config['display_name']} volume analysis completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ {tf_config['display_name']} volume analysis failed: {str(e)}", exc_info=True)
            return False
    
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
