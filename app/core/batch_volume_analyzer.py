"""
Batch Volume Analyzer for processing multiple trading pairs.
Runs volume analysis on all configured pairs and saves results to files.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import aiofiles

from app.core.volume_analyzer import VolumeAnalyzer
from app.core.volume_chart_generator import VolumeChartGenerator
from app.core.telegram_notifier import TelegramNotifier
from app.core.volume_pairs_config import BATCH_CONFIG
from app.utils.logging_config import setup_logging

# Set up logging
logger = logging.getLogger(__name__)


class BatchVolumeAnalyzer:
    """Batch processor for volume analysis of multiple trading pairs."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the batch analyzer with configuration."""
        self.config = config or BATCH_CONFIG
        self.analyzer = VolumeAnalyzer()
        self.chart_generator = VolumeChartGenerator()
        self.telegram_notifier = TelegramNotifier()
        self.results = []
        self.failed_pairs = []
        
        # Create output directory
        self.output_dir = Path(self.config["output_dir"])
        self.output_dir.mkdir(exist_ok=True)
        
        logger.debug(f"BatchVolumeAnalyzer initialized with {len(self.config['pairs'])} pairs")
    
    async def analyze_all_pairs(self) -> Dict[str, Any]:
        """
        Analyze all configured trading pairs and save results to files.
        
        Returns:
            Dictionary with batch analysis summary
        """
        start_time = datetime.now()
        logger.info(f"🚀 Starting batch volume analysis for {len(self.config['pairs'])} pairs")
        logger.info(f"Timeframe: {self.config['timeframe']}, Periods: {self.config['periods']}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        
        # Clear previous results to avoid accumulation between runs
        self.results = []
        self.failed_pairs = []
        
        # Create semaphore for concurrent processing
        semaphore = asyncio.Semaphore(self.config["max_concurrent"])
        
        # Process pairs concurrently
        tasks = []
        for i, pair in enumerate(self.config["pairs"]):
            task = self._analyze_pair_with_semaphore(semaphore, pair, i + 1, len(self.config["pairs"]))
            tasks.append(task)
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Task failed with exception: {result}")
            elif result:
                self.results.append(result)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Generate summary
        summary = await self._generate_summary(start_time, end_time, duration)
        
        # Save summary to file
        await self._save_summary(summary)
        
        logger.info(f"✅ Batch analysis completed in {duration:.2f} seconds")
        logger.info(f"Successful: {len(self.results)}, Failed: {len(self.failed_pairs)}")
        
        return summary
    
    async def _analyze_pair_with_semaphore(self, semaphore: asyncio.Semaphore, pair: str, index: int, total: int) -> Optional[Dict[str, Any]]:
        """Analyze a single pair with semaphore for concurrency control."""
        async with semaphore:
            return await self._analyze_single_pair(pair, index, total)
    
    async def _analyze_single_pair(self, pair: str, index: int, total: int) -> Optional[Dict[str, Any]]:
        """Analyze a single trading pair with retry logic."""
        
        for attempt in range(self.config["retry_attempts"]):
            try:
                # Perform volume analysis
                result = await self.analyzer.analyze_pair(
                    pair=pair,
                    timeframe=self.config["timeframe"],
                    periods=self.config["periods"]
                )
                
                # Skip file saving if configured to do so
                if not self.config.get("save_files", False):
                    logger.debug(f"Analysis completed for {pair} - skipping file generation to save space")
                else:
                    # Generate outputs (original file saving logic would go here)
                    logger.debug(f"Analysis completed for {pair} - file saving enabled but not implemented")
                
                # Create result summary (no files generated)
                pair_result = {
                    "pair": pair,
                    "success": True,
                    "suspicious_periods_count": len(result.suspicious_periods),
                    "confidence_score": result.confidence_score,
                    "analysis_timestamp": result.analysis_timestamp.isoformat(),
                    "files_generated": {
                        "json": None,
                        "chart_html": None,
                        "chart_png": None,
                        "report_html": None,
                    },
                    "metrics": result.metrics,
                    "alerts": result.alerts
                }
                
                # Single line summary for successful analysis
                if result.suspicious_periods:
                    logger.info(f"✅ {pair.upper()}: {len(result.suspicious_periods)} suspicious periods (confidence: {result.confidence_score:.2%})")
                    
                    # Send individual Telegram notification for pairs with suspicious activity
                    await self.telegram_notifier.send_analysis_notification(result)
                else:
                    logger.info(f"✅ {pair.upper()}: No suspicious periods detected")
                
                return pair_result
                
            except Exception as e:
                # Log retry attempts as debug to reduce noise
                logger.debug(f"Attempt {attempt + 1} failed for {pair}: {str(e)}")
                if attempt < self.config["retry_attempts"] - 1:
                    await asyncio.sleep(self.config["retry_delay"])
                else:
                    # Only show error details on final failure
                    logger.error(f"❌ {pair.upper()}: Failed after {self.config['retry_attempts']} attempts - {str(e)}")
                    self.failed_pairs.append({
                        "pair": pair,
                        "error": str(e),
                        "attempts": self.config["retry_attempts"]
                    })
                    return None
    
    async def _generate_summary(self, start_time: datetime, end_time: datetime, duration: float) -> Dict[str, Any]:
        """Generate batch analysis summary."""
        
        # Calculate statistics
        total_pairs = len(self.config["pairs"])
        successful_pairs = len(self.results)
        failed_pairs = len(self.failed_pairs)
        
        # Count pairs with suspicious periods
        pairs_with_suspicious = [r for r in self.results if r.get("suspicious_periods_count", 0) > 0]
        suspicious_count = len(pairs_with_suspicious)
        
        # Calculate average confidence score
        confidence_scores = [r.get("confidence_score", 0) for r in self.results if r.get("confidence_score", 0) > 0]
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Get top pairs by confidence score
        top_pairs = sorted(
            [r for r in self.results if r.get("confidence_score", 0) > 0],
            key=lambda x: x.get("confidence_score", 0),
            reverse=True
        )[:10]
        
        summary = {
            "batch_analysis_summary": {
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": duration,
                "total_pairs": total_pairs,
                "successful_pairs": successful_pairs,
                "failed_pairs": failed_pairs,
                "success_rate": (successful_pairs / total_pairs) * 100 if total_pairs > 0 else 0,
                "pairs_with_suspicious_periods": suspicious_count,
                "suspicious_rate": (suspicious_count / successful_pairs) * 100 if successful_pairs > 0 else 0,
                "average_confidence_score": avg_confidence,
                "analysis_config": {
                    "timeframe": self.config["timeframe"],
                    "periods": self.config["periods"],
                    "max_concurrent": self.config["max_concurrent"]
                }
            },
            "top_suspicious_pairs": [
                {
                    "pair": pair["pair"],
                    "suspicious_periods": pair["suspicious_periods_count"],
                    "confidence_score": pair["confidence_score"]
                }
                for pair in top_pairs
            ],
            "failed_pairs": self.failed_pairs,
            "all_results": self.results
        }
        
        return summary
    
    async def _save_summary(self, summary: Dict[str, Any]) -> None:
        """Skip saving batch analysis summary to file to save disk space."""
        logger.debug("📊 Batch summary generated - skipping file save to conserve disk space")
    
    def get_recent_results(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get results from recent analysis runs."""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        recent_results = []
        
        for result in self.results:
            try:
                analysis_time = datetime.fromisoformat(result["analysis_timestamp"]).timestamp()
                if analysis_time > cutoff_time:
                    recent_results.append(result)
            except (ValueError, KeyError):
                continue
        
        return recent_results


