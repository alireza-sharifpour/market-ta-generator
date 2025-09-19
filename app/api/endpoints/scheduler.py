"""
Scheduler API endpoints for volume analysis.
Provides REST API to control the volume analysis scheduler.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from app.core.volume_scheduler import VolumeAnalysisScheduler

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Volume Analysis Scheduler"])

# Global scheduler instance
volume_scheduler: Optional[VolumeAnalysisScheduler] = None


class SchedulerConfigRequest(BaseModel):
    """Request model for scheduler configuration."""
    
    interval_minutes: Optional[int] = Field(None, ge=1, le=1440, description="Interval in minutes (1-1440)")
    max_instances: Optional[int] = Field(None, ge=1, le=10, description="Maximum concurrent instances")
    coalesce: Optional[bool] = Field(None, description="Combine missed runs")
    misfire_grace_time: Optional[int] = Field(None, ge=0, le=3600, description="Grace time in seconds")


class SchedulerStatusResponse(BaseModel):
    """Response model for scheduler status."""
    
    status: str = Field(..., description="Scheduler status (running/stopped)")
    is_running: bool = Field(..., description="Whether scheduler is active")
    jobs: list = Field(default_factory=list, description="List of scheduled jobs")
    scheduler_config: Optional[Dict[str, Any]] = Field(None, description="Current configuration")


class ManualAnalysisResponse(BaseModel):
    """Response model for manual analysis."""
    
    success: bool = Field(..., description="Whether analysis completed successfully")
    timestamp: str = Field(..., description="Analysis timestamp")
    type: str = Field(..., description="Analysis type (manual)")


def get_scheduler() -> VolumeAnalysisScheduler:
    """Get or create the global scheduler instance."""
    global volume_scheduler
    if volume_scheduler is None:
        volume_scheduler = VolumeAnalysisScheduler()
    return volume_scheduler


@router.post("/scheduler/start")
async def start_scheduler(
    schedule_type: str = "interval",
    background_tasks: BackgroundTasks = None
) -> Dict[str, str]:
    """
    Start the volume analysis scheduler.
    
    Args:
        schedule_type: "interval" for every N minutes, "cron" for cron-like scheduling
    """
    try:
        scheduler = get_scheduler()
        
        if scheduler.is_running:
            raise HTTPException(status_code=400, detail="Scheduler is already running")
        
        # Start scheduler in background
        if background_tasks:
            background_tasks.add_task(scheduler.start_scheduler, schedule_type)
        else:
            scheduler.start_scheduler(schedule_type)
        
        logger.info(f"Volume analysis scheduler started with {schedule_type} schedule")
        
        return {
            "status": "success",
            "message": f"Scheduler started with {schedule_type} schedule",
            "schedule_type": schedule_type
        }
        
    except Exception as e:
        logger.error(f"Failed to start scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to start scheduler: {str(e)}")


@router.post("/scheduler/stop")
async def stop_scheduler() -> Dict[str, str]:
    """Stop the volume analysis scheduler."""
    try:
        scheduler = get_scheduler()
        
        if not scheduler.is_running:
            raise HTTPException(status_code=400, detail="Scheduler is not running")
        
        scheduler.stop_scheduler()
        
        logger.info("Volume analysis scheduler stopped")
        
        return {
            "status": "success",
            "message": "Scheduler stopped successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to stop scheduler: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to stop scheduler: {str(e)}")


@router.get("/scheduler/status", response_model=SchedulerStatusResponse)
async def get_scheduler_status() -> SchedulerStatusResponse:
    """Get current scheduler status and job information."""
    try:
        scheduler = get_scheduler()
        status = scheduler.get_scheduler_status()
        
        return SchedulerStatusResponse(**status)
        
    except Exception as e:
        logger.error(f"Failed to get scheduler status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get scheduler status: {str(e)}")


@router.post("/scheduler/config")
async def update_scheduler_config(
    config: SchedulerConfigRequest
) -> Dict[str, str]:
    """Update scheduler configuration."""
    try:
        scheduler = get_scheduler()
        
        # Convert Pydantic model to dict, excluding None values
        config_dict = {k: v for k, v in config.dict().items() if v is not None}
        
        if not config_dict:
            raise HTTPException(status_code=400, detail="No configuration provided")
        
        scheduler.update_schedule(config_dict)
        
        logger.info(f"Scheduler configuration updated: {config_dict}")
        
        return {
            "status": "success",
            "message": "Scheduler configuration updated successfully",
            "updated_config": config_dict
        }
        
    except Exception as e:
        logger.error(f"Failed to update scheduler config: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update scheduler config: {str(e)}")


@router.post("/scheduler/run-manual", response_model=ManualAnalysisResponse)
async def run_manual_analysis() -> ManualAnalysisResponse:
    """Run volume analysis manually (outside of schedule)."""
    try:
        scheduler = get_scheduler()
        result = await scheduler.run_manual_analysis()
        
        return ManualAnalysisResponse(**result)
        
    except Exception as e:
        logger.error(f"Failed to run manual analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to run manual analysis: {str(e)}")


@router.get("/scheduler/info")
async def get_scheduler_info() -> Dict[str, Any]:
    """Get information about the scheduler and its capabilities."""
    return {
        "scheduler_type": "APScheduler",
        "supported_schedules": ["interval", "cron"],
        "features": [
            "Interval-based scheduling",
            "Cron-like scheduling",
            "Job monitoring and logging",
            "Manual execution",
            "Configuration updates",
            "Graceful shutdown",
            "Misfire handling"
        ],
        "default_config": {
            "interval_minutes": 5,
            "max_instances": 1,
            "coalesce": True,
            "misfire_grace_time": 60
        }
    }
