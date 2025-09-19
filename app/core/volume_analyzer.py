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
        """Calculate volume indicators and RSI for enhanced detection."""
        logger.info("Calculating volume indicators and RSI for enhanced detection")
        
        # Calculate RSI if not already present
        if "RSI_14" not in df.columns:
            from app.core.data_processor import add_technical_indicators
            df = add_technical_indicators(df)
        
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
        
        # RSI-based market condition detection
        if self.config.get("enable_rsi_volume_alerts", True):
            rsi_period = self.config.get("rsi_period", 14)
            rsi_col = f"RSI_{rsi_period}"
            
            if rsi_col in df.columns:
                # Market condition flags
                df["rsi_overbought"] = df[rsi_col] > self.config.get("rsi_overbought_threshold", 70)
                df["rsi_oversold"] = df[rsi_col] < self.config.get("rsi_oversold_threshold", 30)
                df["rsi_neutral"] = ~(df["rsi_overbought"] | df["rsi_oversold"])
                
                logger.info(f"RSI analysis enabled with overbought={self.config.get('rsi_overbought_threshold', 70)}, oversold={self.config.get('rsi_oversold_threshold', 30)}")
            else:
                logger.warning(f"RSI column {rsi_col} not found in DataFrame")
        
        logger.info("Volume indicators and RSI calculated successfully")
        return df
    
    def _detect_suspicious_volume(self, df: pd.DataFrame) -> List[Dict]:
        """Detect suspicious volume patterns with RSI-enhanced intelligence."""
        logger.info("Detecting suspicious volume patterns with RSI enhancement")
        
        suspicious_periods = []
        
        for i in range(len(df)):
            timestamp = df.index[i]
            alerts = []
            score = 0
            alert_type = "volume_spike"  # Default alert type
            
            # Mean and Standard Deviation Spike Detection
            if self.config.get("enable_mean_std_detection", True) and pd.notna(df["volume_spike_detected"].iloc[i]):
                if df["volume_spike_detected"].iloc[i]:
                    # Calculate base score based on spike severity
                    spike_ratio = df["volume_spike_ratio"].iloc[i] if pd.notna(df["volume_spike_ratio"].iloc[i]) else 1.0
                    
                    if spike_ratio > 2.0:  # Very extreme spike
                        base_score = 4
                    elif spike_ratio > 1.5:  # High spike
                        base_score = 3
                    else:  # Standard spike
                        base_score = 2
                    
                    # RSI-Enhanced Intelligence
                    if self.config.get("enable_rsi_volume_alerts", True):
                        rsi_period = self.config.get("rsi_period", 14)
                        rsi_col = f"RSI_{rsi_period}"
                        
                        if rsi_col in df.columns and pd.notna(df[rsi_col].iloc[i]):
                            current_rsi = df[rsi_col].iloc[i]
                            
                            # Alert 1: Potential Market Top (Bearish Signal) 🐻
                            if (self.config.get("enable_bearish_volume_alerts", True) and 
                                current_rsi > self.config.get("rsi_overbought_threshold", 70)):
                                
                                alerts.append("bearish_volume_spike")
                                alert_type = "potential_market_top"
                                score = base_score + 2  # Boost score for RSI confirmation
                                
                                logger.info(f"🐻 Bearish alert: Volume spike + RSI {current_rsi:.1f} > 70 at {timestamp}")
                            
                            # Alert 2: Potential Market Bottom (Bullish Signal) 🐂
                            elif (self.config.get("enable_bullish_volume_alerts", True) and 
                                  current_rsi < self.config.get("rsi_oversold_threshold", 30)):
                                
                                alerts.append("bullish_volume_spike")
                                alert_type = "potential_market_bottom"
                                score = base_score + 2  # Boost score for RSI confirmation
                                
                                logger.info(f"🐂 Bullish alert: Volume spike + RSI {current_rsi:.1f} < 30 at {timestamp}")
                            
                            # Standard volume spike (no RSI extreme)
                            else:
                                alerts.append("mean_std_volume_spike")
                                score = base_score
                        else:
                            # Fallback to standard volume spike if RSI not available
                            alerts.append("mean_std_volume_spike")
                            score = base_score
                    else:
                        # Standard volume spike without RSI enhancement
                        alerts.append("mean_std_volume_spike")
                        score = base_score
            
            # Mark as suspicious if we have any alerts
            if score >= 2 and alerts:
                suspicious_periods.append({
                    "timestamp": timestamp,
                    "index": i,
                    "alerts": alerts,
                    "alert_type": alert_type,
                    "score": score,
                    "volume": df["Volume"].iloc[i],
                    "volume_mean": df["volume_mean"].iloc[i] if pd.notna(df["volume_mean"].iloc[i]) else 0,
                    "volume_spike_threshold": df["volume_spike_threshold"].iloc[i] if pd.notna(df["volume_spike_threshold"].iloc[i]) else 0,
                    "volume_spike_ratio": df["volume_spike_ratio"].iloc[i] if pd.notna(df["volume_spike_ratio"].iloc[i]) else 1.0,
                    "price": df["Close"].iloc[i],
                    "rsi": df[f"RSI_{self.config.get('rsi_period', 14)}"].iloc[i] if f"RSI_{self.config.get('rsi_period', 14)}" in df.columns else None,
                })
        
        logger.info(f"Found {len(suspicious_periods)} suspicious periods with RSI enhancement")
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
        """Generate intelligent alerts based on RSI-enhanced volume analysis."""
        alerts = []
        
        if confidence < self.config.get("confidence_threshold", 0.7):
            return alerts
        
        # Separate alerts by type
        bearish_alerts = [sp for sp in suspicious_periods if "bearish_volume_spike" in sp["alerts"]]
        bullish_alerts = [sp for sp in suspicious_periods if "bullish_volume_spike" in sp["alerts"]]
        standard_alerts = [sp for sp in suspicious_periods if "mean_std_volume_spike" in sp["alerts"] and "bearish_volume_spike" not in sp["alerts"] and "bullish_volume_spike" not in sp["alerts"]]
        
        # High-level summary alert
        alerts.append({
            "type": "summary",
            "level": "info" if confidence < 0.8 else "warning" if confidence < 0.9 else "critical",
            "message": f"Detected {len(suspicious_periods)} suspicious volume periods with {confidence:.1%} confidence",
            "timestamp": datetime.now()
        })
        
        # Bearish Alerts (Potential Market Top) 🐻
        if bearish_alerts:
            max_ratio = max(p.get("volume_spike_ratio", 1.0) for p in bearish_alerts)
            rsi_values = [p.get("rsi") for p in bearish_alerts if p.get("rsi") is not None]
            avg_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else 0
            
            alerts.append({
                "type": "bearish_volume_spike",
                "level": "critical",
                "message": f"🐻 POTENTIAL MARKET TOP: {len(bearish_alerts)} volume spikes during overbought conditions (RSI avg: {avg_rsi:.1f}, max volume ratio: {max_ratio:.1f}x)",
                "count": len(bearish_alerts),
                "avg_rsi": avg_rsi,
                "max_volume_ratio": max_ratio,
                "timestamp": datetime.now()
            })
        
        # Bullish Alerts (Potential Market Bottom) 🐂
        if bullish_alerts:
            max_ratio = max(p.get("volume_spike_ratio", 1.0) for p in bullish_alerts)
            rsi_values = [p.get("rsi") for p in bullish_alerts if p.get("rsi") is not None]
            avg_rsi = sum(rsi_values) / len(rsi_values) if rsi_values else 0
            
            alerts.append({
                "type": "bullish_volume_spike",
                "level": "critical",
                "message": f"🐂 POTENTIAL MARKET BOTTOM: {len(bullish_alerts)} volume spikes during oversold conditions (RSI avg: {avg_rsi:.1f}, max volume ratio: {max_ratio:.1f}x)",
                "count": len(bullish_alerts),
                "avg_rsi": avg_rsi,
                "max_volume_ratio": max_ratio,
                "timestamp": datetime.now()
            })
        
        # Standard Volume Spike Alerts
        if standard_alerts:
            max_ratio = max(p.get("volume_spike_ratio", 1.0) for p in standard_alerts)
            alerts.append({
                "type": "volume_spike",
                "level": "warning" if max_ratio > 2.0 else "info",
                "message": f"Volume spikes detected in {len(standard_alerts)} periods (max ratio: {max_ratio:.1f}x) - No RSI extremes",
                "count": len(standard_alerts),
                "timestamp": datetime.now()
            })
        
        return alerts


