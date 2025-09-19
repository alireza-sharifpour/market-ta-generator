"""
Volume Analysis module for detecting suspicious volume activity in crypto pairs.
Integrates with market-ta-generator infrastructure, reusing existing components.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import numpy as np
import pandas as pd

from app.external.lbank_client import fetch_ohlcv, LBankAPIError, LBankConnectionError
from app.core.data_processor import process_raw_data
from app.config import VOLUME_ANALYSIS_CONFIG, DEFAULT_TIMEFRAME, DEFAULT_SIZE

# Set up logging
logger = logging.getLogger(__name__)


class VolumeAnalysisResult:
    """Data class to hold volume analysis results."""
    
    def __init__(self):
        self.pair: str = ""
        self.timeframe: str = ""
        self.analysis_timestamp: datetime = datetime.now()
        self.data: pd.DataFrame = pd.DataFrame()
        self.suspicious_periods: List[Dict] = []
        self.metrics: Dict[str, Any] = {}
        self.alerts: List[Dict] = []
        self.confidence_score: float = 0.0


class VolumeAnalyzer:
    """Main class for analyzing suspicious volume activity in crypto pairs using mean+std method."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the VolumeAnalyzer with configuration."""
        self.config = config or VOLUME_ANALYSIS_CONFIG
        logger.info("VolumeAnalyzer initialized with mean+std detection method")
    
    async def analyze_pair(
        self, 
        pair: str, 
        timeframe: str = None, 
        periods: int = None
    ) -> VolumeAnalysisResult:
        """
        Analyze a trading pair for suspicious volume activity using mean+std method.
        
        Args:
            pair: Trading pair symbol (e.g., "btc_usdt")
            timeframe: Timeframe for analysis (defaults to market-ta-generator default)
            periods: Number of periods to analyze (defaults to market-ta-generator default)
        
        Returns:
            VolumeAnalysisResult with analysis data and findings
        """
        # Use market-ta-generator defaults if not specified
        timeframe_to_use = timeframe or DEFAULT_TIMEFRAME
        periods_to_use = periods or DEFAULT_SIZE
        
        logger.info(f"Starting volume analysis for {pair} ({timeframe_to_use}, {periods_to_use} periods)")
        
        result = VolumeAnalysisResult()
        result.pair = pair
        result.timeframe = timeframe_to_use
        
        try:
            # Fetch OHLCV data using market-ta-generator's LBank client
            raw_data = await fetch_ohlcv(pair, timeframe_to_use, periods_to_use)
            
            # Convert to DataFrame using market-ta-generator's data processor
            df = process_raw_data(raw_data)
            
            # Calculate volume indicators (focusing on mean+std method)
            df = self._calculate_volume_indicators(df)
            
            # Detect suspicious activity using mean+std method only
            suspicious_periods = self._detect_suspicious_volume(df)
            
            # Calculate metrics and confidence
            metrics = self._calculate_metrics(df, suspicious_periods)
            confidence = self._calculate_confidence_score(df, suspicious_periods)
            
            # Generate alerts
            alerts = self._generate_alerts(df, suspicious_periods, confidence)
            
            # Populate result
            result.data = df
            result.suspicious_periods = suspicious_periods
            result.metrics = metrics
            result.alerts = alerts
            result.confidence_score = confidence
            
            logger.info(f"Analysis completed. Found {len(suspicious_periods)} suspicious periods with confidence {confidence:.2f}")
            
            return result
            
        except (LBankAPIError, LBankConnectionError) as e:
            logger.error(f"Failed to fetch data for {pair}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error during volume analysis: {e}")
            raise
    
    def _calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate volume indicators focusing on mean+std method."""
        logger.info("Calculating volume indicators (mean+std method)")
        
        # Mean and Standard Deviation Method (primary method)
        if self.config.get("enable_mean_std_detection", True):
            lookback_period = self.config.get("mean_std_lookback_period", 25)
            multiplier = self.config.get("mean_std_multiplier", 4.0)
            
            logger.info(f"Using mean+std method with lookback={lookback_period}, multiplier={multiplier}")
            
            # Calculate rolling mean and standard deviation
            df["volume_mean"] = df["Volume"].rolling(window=lookback_period).mean()
            df["volume_std"] = df["Volume"].rolling(window=lookback_period).std()
            
            # Calculate spike threshold: Mean + (Multiplier * Standard_Deviation)
            df["volume_spike_threshold"] = df["volume_mean"] + (multiplier * df["volume_std"])
            
            # Detect spikes: current volume > threshold
            df["volume_spike_detected"] = df["Volume"] > df["volume_spike_threshold"]
            
            # Calculate spike ratio for analysis
            df["volume_spike_ratio"] = df["Volume"] / df["volume_spike_threshold"]
            df["volume_spike_ratio"] = df["volume_spike_ratio"].fillna(1.0)  # Fill NaN with 1.0
        
        logger.info("Volume indicators calculated successfully")
        return df
    
    def _detect_suspicious_volume(self, df: pd.DataFrame) -> List[Dict]:
        """Detect suspicious volume patterns using mean+std method only."""
        logger.info("Detecting suspicious volume patterns using mean+std method")
        
        suspicious_periods = []
        
        for i in range(len(df)):
            timestamp = df.index[i]
            alerts = []
            score = 0
            
            # Mean and Standard Deviation Spike Detection (primary method)
            if self.config.get("enable_mean_std_detection", True) and pd.notna(df["volume_spike_detected"].iloc[i]):
                if df["volume_spike_detected"].iloc[i]:
                    alerts.append("mean_std_volume_spike")
                    
                    # Calculate score based on how extreme the spike is
                    spike_ratio = df["volume_spike_ratio"].iloc[i] if pd.notna(df["volume_spike_ratio"].iloc[i]) else 1.0
                    
                    if spike_ratio > 2.0:  # Very extreme spike
                        score += 4
                    elif spike_ratio > 1.5:  # High spike
                        score += 3
                    else:  # Standard spike
                        score += 2
            
            # Mark as suspicious if we have any alerts (since we only use one method)
            if score >= 2 and alerts:
                suspicious_periods.append({
                    "timestamp": timestamp,
                    "index": i,
                    "alerts": alerts,
                    "score": score,
                    "volume": df["Volume"].iloc[i],
                    "volume_mean": df["volume_mean"].iloc[i] if pd.notna(df["volume_mean"].iloc[i]) else 0,
                    "volume_spike_threshold": df["volume_spike_threshold"].iloc[i] if pd.notna(df["volume_spike_threshold"].iloc[i]) else 0,
                    "volume_spike_ratio": df["volume_spike_ratio"].iloc[i] if pd.notna(df["volume_spike_ratio"].iloc[i]) else 1.0,
                    "price": df["Close"].iloc[i],
                })
        
        logger.info(f"Found {len(suspicious_periods)} suspicious periods using mean+std method")
        return suspicious_periods
    
    def _calculate_metrics(self, df: pd.DataFrame, suspicious_periods: List[Dict]) -> Dict[str, Any]:
        """Calculate overall metrics for the analysis."""
        metrics = {}
        
        # Basic statistics
        metrics["total_periods"] = len(df)
        metrics["suspicious_periods_count"] = len(suspicious_periods)
        metrics["suspicious_percentage"] = (len(suspicious_periods) / len(df)) * 100 if len(df) > 0 else 0
        
        # Volume statistics
        metrics["avg_volume"] = float(df["Volume"].mean())
        metrics["max_volume"] = float(df["Volume"].max())
        metrics["volume_std"] = float(df["Volume"].std())
        
        # Mean+std specific metrics
        if "volume_spike_ratio" in df.columns:
            metrics["max_spike_ratio"] = float(df["volume_spike_ratio"].max())
            metrics["avg_spike_ratio"] = float(df["volume_spike_ratio"].mean())
        
        # Alert type frequency
        alert_counts = {}
        for period in suspicious_periods:
            for alert in period["alerts"]:
                alert_counts[alert] = alert_counts.get(alert, 0) + 1
        metrics["alert_frequency"] = alert_counts
        
        return metrics
    
    def _calculate_confidence_score(self, df: pd.DataFrame, suspicious_periods: List[Dict]) -> float:
        """Calculate confidence score for the analysis."""
        if len(suspicious_periods) == 0:
            return 0.0
        
        # Base score from suspicious period ratio
        base_score = min(len(suspicious_periods) / len(df) * 10, 1.0)
        
        # Boost score based on severity of spikes
        severity_boost = 0
        for period in suspicious_periods:
            spike_ratio = period.get("volume_spike_ratio", 1.0)
            if spike_ratio > 2.0:
                severity_boost += 0.3
            elif spike_ratio > 1.5:
                severity_boost += 0.2
            else:
                severity_boost += 0.1
        
        # Normalize severity boost
        severity_boost = min(severity_boost / len(suspicious_periods), 0.5)
        
        # Final confidence score
        confidence = min(base_score + severity_boost, 1.0)
        
        return confidence
    
    def _generate_alerts(self, df: pd.DataFrame, suspicious_periods: List[Dict], confidence: float) -> List[Dict]:
        """Generate human-readable alerts based on analysis."""
        alerts = []
        
        if confidence < self.config.get("confidence_threshold", 0.7):
            return alerts
        
        # High-level summary alert
        alerts.append({
            "type": "summary",
            "level": "info" if confidence < 0.8 else "warning" if confidence < 0.9 else "critical",
            "message": f"Detected {len(suspicious_periods)} suspicious volume periods using mean+std method with {confidence:.1%} confidence",
            "timestamp": datetime.now()
        })
        
        # Specific alerts for mean+std spikes
        mean_std_periods = [sp for sp in suspicious_periods if 'mean_std_volume_spike' in sp['alerts']]
        if mean_std_periods:
            max_ratio = max(p.get("volume_spike_ratio", 1.0) for p in mean_std_periods)
            alerts.append({
                "type": "volume_spike",
                "level": "critical" if max_ratio > 2.0 else "warning",
                "message": f"Volume spikes detected using mean+4×std method in {len(mean_std_periods)} periods (max ratio: {max_ratio:.1f}x)",
                "count": len(mean_std_periods),
                "timestamp": datetime.now()
            })
        
        return alerts
