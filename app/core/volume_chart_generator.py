"""
Volume Chart Generator for market-ta-generator.
Creates interactive charts for volume analysis with suspicious period highlighting.
"""

import logging
import base64
from typing import Optional, List, Dict, Any
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import io

from app.core.volume_analyzer import VolumeAnalysisResult
from app.config import VOLUME_CHART_CONFIG

# Set up logging
logger = logging.getLogger(__name__)


class VolumeChartGenerator:
    """Generate interactive charts for volume analysis with suspicious period highlighting."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the chart generator with configuration."""
        self.config = config or VOLUME_CHART_CONFIG
        logger.debug("VolumeChartGenerator initialized")
    
    def _get_severity_info(self, suspicious_periods: List[Dict]) -> Dict[str, str]:
        """Get severity information for chart title and display."""
        if not suspicious_periods:
            return {
                "title": "No Suspicious Activity Detected",
                "severity": "none",
                "threshold_info": ""
            }
        
        # Get the highest severity level
        severities = [period.get("severity", "unknown") for period in suspicious_periods]
        severity_counts = {}
        for severity in severities:
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Determine primary severity (highest level)
        severity_levels = {"low": 1, "medium": 2, "high": 3}
        primary_severity = max(severities, key=lambda x: severity_levels.get(x, 0))
        
        # Create title and threshold info
        if primary_severity == "high":
            title = "HIGH Suspicious Volume Detected"
            threshold_info = "Triggered: High Threshold (6.0σ)"
        elif primary_severity == "medium":
            title = "MEDIUM Suspicious Volume Detected"
            threshold_info = "Triggered: Medium Threshold (4.0σ)"
        elif primary_severity == "low":
            title = "LOW Suspicious Volume Detected"
            threshold_info = "Triggered: Low Threshold (2.0σ)"
        else:
            title = "Suspicious Volume Detected"
            threshold_info = "Unknown Severity"
        
        # Add severity breakdown if multiple levels
        if len(severity_counts) > 1:
            breakdown = ", ".join([f"{count} {sev}" for sev, count in severity_counts.items()])
            title += f" ({breakdown})"
        
        return {
            "title": title,
            "severity": primary_severity,
            "threshold_info": threshold_info,
            "severity_breakdown": severity_counts
        }
    
    def create_analysis_chart(self, result: VolumeAnalysisResult) -> str:
        """
        Create a comprehensive chart showing price, volume, RSI, and three-level threshold analysis.
        
        Args:
            result: VolumeAnalysisResult containing analysis data
            
        Returns:
            HTML string of the interactive chart
        """
        logger.debug(f"Creating three-level volume analysis chart for {result.pair}")
        
        df = result.data
        suspicious_periods = result.suspicious_periods
        
        # Determine severity level and threshold info for title
        severity_info = self._get_severity_info(suspicious_periods)
        
        # Create subplot layout with RSI
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(
                f'{result.pair.upper()} Price Chart',
                f'Volume Analysis - {severity_info["title"]}',
                'RSI (14) - Market Conditions'
            ),
            vertical_spacing=0.08,
            row_heights=[0.5, 0.3, 0.2],
            specs=[
                [{"secondary_y": False}],  # Price chart
                [{"secondary_y": False}],  # Volume chart with three-level thresholds
                [{"secondary_y": False}]   # RSI chart
            ]
        )
        
        # 1. Main Price Chart (Row 1)
        self._add_price_chart(fig, df, suspicious_periods, row=1)
        
        # 2. Volume Chart with Three-Level Thresholds (Row 2)
        self._add_volume_chart_three_levels(fig, df, suspicious_periods, row=2)
        
        # 3. RSI Chart (Row 3)
        self._add_rsi_chart(fig, df, suspicious_periods, row=3)
        
        # Update layout
        fig.update_layout(
            height=self.config["height"] + 200,  # Increase height for RSI subplot
            template=self.config["template"],
            title=f"{result.pair.upper()} RSI-Enhanced Volume Analysis - {result.timeframe} - Mean+4×Std + RSI Intelligence",
            showlegend=True,
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        # Update all x-axes to show datetime properly
        for i in range(1, 4):
            fig.update_xaxes(
                type='date',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.3)',
                row=i, col=1
            )
        
        logger.debug("RSI-enhanced volume analysis chart created successfully")
        return fig.to_html(include_plotlyjs=True, div_id="rsi-volume-analysis-chart")
    
    def create_analysis_chart_base64(self, result: VolumeAnalysisResult) -> str:
        """
        Create chart and return as base64 encoded PNG similar to market-ta-generator.
        
        Args:
            result: VolumeAnalysisResult containing analysis data
            
        Returns:
            Base64 encoded PNG image
        """
        logger.debug(f"Creating volume analysis chart as base64 for {result.pair}")
        
        df = result.data
        suspicious_periods = result.suspicious_periods
        
        # Create subplot layout
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f'{result.pair.upper()} Price Chart',
                'Volume Analysis (Mean + 4×Std Detection)'
            ),
            vertical_spacing=0.12,
            row_heights=[0.6, 0.4],
            specs=[
                [{"secondary_y": False}],  # Price chart
                [{"secondary_y": False}]   # Volume chart with mean/std lines
            ]
        )
        
        # 1. Main Price Chart (Row 1)
        self._add_price_chart(fig, df, suspicious_periods, row=1)
        
        # 2. Volume Chart with Mean+Std Analysis (Row 2)
        self._add_volume_chart_three_levels(fig, df, suspicious_periods, row=2)
        
        # Update layout
        fig.update_layout(
            height=self.config["height"],
            template=self.config["template"],
            title=f"{result.pair.upper()} Volume Analysis - {result.timeframe} - Mean+4×Std Method",
            showlegend=True,
            hovermode='x unified',
            xaxis_rangeslider_visible=False
        )
        
        # Update all x-axes to show datetime properly
        for i in range(1, 3):
            fig.update_xaxes(
                type='date',
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(128,128,128,0.3)',
                row=i, col=1
            )
        
        # Convert to base64 PNG
        img_bytes = fig.to_image(format="png", width=self.config["width"], height=self.config["height"])
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')
        
        logger.debug("Volume analysis chart created as base64 PNG")
        return f"data:image/png;base64,{img_base64}"
    
    def _add_price_chart(self, fig, df: pd.DataFrame, suspicious_periods: List[Dict], row: int):
        """Add price candlestick chart with suspicious period highlighting."""
        
        # Check if we have OHLC data for candlestick chart
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        if all(col in df.columns for col in ohlc_columns):
            # Use candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name='Price',
                    increasing_line_color='#26a69a',
                    decreasing_line_color='#ef5350'
                ),
                row=row, col=1
            )
            logger.debug("Added candlestick chart")
        elif 'Close' in df.columns:
            # Fallback to line chart if OHLC data is not available
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['Close'],
                    mode='lines',
                    name='Price',
                    line=dict(color='#26a69a', width=2)
                ),
                row=row, col=1
            )
            logger.debug("Added line chart (OHLC data not available)")
        else:
            logger.warning("No price data available for chart")
            return
        
        # Add suspicious period highlights to price chart
        self._add_suspicious_highlights(fig, df, suspicious_periods, row)
        
        # Update y-axes with proper price range
        if not df.empty and 'High' in df.columns and 'Low' in df.columns:
            # Calculate price range with some padding
            min_price = df['Low'].min()
            max_price = df['High'].max()
            price_range = max_price - min_price
            padding = price_range * 0.05  # 5% padding
            
            fig.update_yaxes(
                title_text="Price",
                range=[min_price - padding, max_price + padding],
                row=row, col=1
            )
        else:
            fig.update_yaxes(title_text="Price", row=row, col=1)
    
    def _add_volume_chart_three_levels(self, fig, df: pd.DataFrame, suspicious_periods: List[Dict], row: int):
        """Add volume chart with three-level threshold analysis."""
        
        # Check if we have valid data
        if df.empty or 'Volume' not in df.columns:
            logger.warning("No valid volume data to display")
            return
        
        # Volume bars with color coding for severity levels
        volume_colors = []
        for i in range(len(df)):
            # Find the suspicious period for this index
            period = next((sp for sp in suspicious_periods if sp['index'] == i), None)
            if period:
                # Color based on severity level
                severity = period.get('severity', 'unknown')
                if severity == 'high':
                    volume_colors.append("#FF0000")  # Red for high severity
                elif severity == 'medium':
                    volume_colors.append("#FF8800")  # Orange for medium severity
                elif severity == 'low':
                    volume_colors.append("#FFDD00")  # Yellow for low severity
                else:
                    volume_colors.append("#FF00FF")  # Magenta for unknown
            else:
                volume_colors.append(self.config["volume_color"])  # Blue for normal volume
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['Volume'],
                name='Volume',
                marker_color=volume_colors,
                hovertemplate='Volume: %{y:,.0f}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # Volume mean line (if available)
        if 'volume_mean' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['volume_mean'],
                    mode='lines',
                    name='Volume Mean',
                    line=dict(color='green', width=2, dash='solid'),
                    hovertemplate='Volume Mean: %{y:,.0f}<extra></extra>'
                ),
                row=row, col=1
            )
        
        # Three-level threshold lines
        if 'volume_threshold_low' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['volume_threshold_low'],
                    mode='lines',
                    name='Low Threshold (2.0σ)',
                    line=dict(color='yellow', width=2, dash='dot'),
                    hovertemplate='Low Threshold: %{y:,.0f}<extra></extra>'
                ),
                row=row, col=1
            )
        
        if 'volume_threshold_medium' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['volume_threshold_medium'],
                    mode='lines',
                    name='Medium Threshold (4.0σ)',
                    line=dict(color='orange', width=2, dash='dash'),
                    hovertemplate='Medium Threshold: %{y:,.0f}<extra></extra>'
                ),
                row=row, col=1
            )
        
        if 'volume_threshold_high' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['volume_threshold_high'],
                    mode='lines',
                    name='High Threshold (6.0σ)',
                    line=dict(color='red', width=2, dash='solid'),
                    hovertemplate='High Threshold: %{y:,.0f}<extra></extra>'
                ),
                row=row, col=1
            )
        
        # Add annotations for three-level suspicious periods on volume chart
        for period in suspicious_periods:
            spike_ratio = period.get('volume_spike_ratio', 1.0)
            rsi_value = period.get('rsi', None)
            severity = period.get('severity', 'unknown')
            
            # Create annotation text with severity information
            severity_text = severity.upper() if severity != 'unknown' else 'UNKNOWN'
            
            if "bearish_volume_spike" in period['alerts']:
                text = f"🐻 Bearish Alert<br>{severity_text} Severity<br>Spike: {spike_ratio:.1f}x<br>RSI: {rsi_value:.1f}" if rsi_value else f"🐻 Bearish Alert<br>{severity_text} Severity<br>Spike: {spike_ratio:.1f}x"
                arrowcolor = "#FF4444"
                bordercolor = "#FF4444"
            elif "bullish_volume_spike" in period['alerts']:
                text = f"🐂 Bullish Alert<br>{severity_text} Severity<br>Spike: {spike_ratio:.1f}x<br>RSI: {rsi_value:.1f}" if rsi_value else f"🐂 Bullish Alert<br>{severity_text} Severity<br>Spike: {spike_ratio:.1f}x"
                arrowcolor = "#44FF44"
                bordercolor = "#44FF44"
            else:
                text = f"📊 Volume Spike<br>{severity_text} Severity<br>Ratio: {spike_ratio:.1f}x<br>RSI: {rsi_value:.1f}" if rsi_value else f"📊 Volume Spike<br>{severity_text} Severity<br>Ratio: {spike_ratio:.1f}x"
                arrowcolor = "#FFFF00"
                bordercolor = "#FFFF00"
            
            fig.add_annotation(
                x=period['timestamp'],
                y=period['volume'],
                text=text,
                showarrow=True,
                arrowhead=2,
                arrowcolor=arrowcolor,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=bordercolor,
                borderwidth=2,
                row=row, col=1
            )
        
        # Update volume y-axis with proper range
        if not df.empty and 'Volume' in df.columns:
            max_volume = df['Volume'].max()
            # Set volume range from 0 to max volume with some padding
            fig.update_yaxes(
                title_text="Volume",
                range=[0, max_volume * 1.1],  # 10% padding above max volume
                row=row, col=1
            )
        else:
            fig.update_yaxes(title_text="Volume", row=row, col=1)
    
    def _add_rsi_chart(self, fig, df: pd.DataFrame, suspicious_periods: List[Dict], row: int):
        """Add RSI chart with overbought/oversold levels and alert highlights."""
        
        # Check if RSI data is available
        rsi_col = "RSI_14"
        if rsi_col not in df.columns:
            logger.warning("RSI data not available for chart")
            return
        
        # RSI line
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[rsi_col],
                mode='lines',
                name='RSI (14)',
                line=dict(color='purple', width=2),
                hovertemplate='RSI: %{y:.1f}<extra></extra>'
            ),
            row=row, col=1
        )
        
        # Overbought level (70)
        fig.add_hline(
            y=70,
            line_dash="dash",
            line_color="red",
            annotation_text="Overbought (70)",
            annotation_position="top right",
            row=row, col=1
        )
        
        # Oversold level (30)
        fig.add_hline(
            y=30,
            line_dash="dash",
            line_color="green",
            annotation_text="Oversold (30)",
            annotation_position="bottom right",
            row=row, col=1
        )
        
        # Highlight RSI-enhanced alerts
        for period in suspicious_periods:
            if period.get('rsi') is not None:
                rsi_value = period['rsi']
                timestamp = period['timestamp']
                
                # Color based on alert type
                if "bearish_volume_spike" in period['alerts']:
                    color = "red"
                    symbol = "🔻"
                    text = f"🐻 Bearish Alert<br>RSI: {rsi_value:.1f}"
                elif "bullish_volume_spike" in period['alerts']:
                    color = "green"
                    symbol = "🔺"
                    text = f"🐂 Bullish Alert<br>RSI: {rsi_value:.1f}"
                else:
                    color = "yellow"
                    symbol = "📊"
                    text = f"📊 Volume Spike<br>RSI: {rsi_value:.1f}"
                
                # Add annotation
                fig.add_annotation(
                    x=timestamp,
                    y=rsi_value,
                    text=text,
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    bgcolor="rgba(255,255,255,0.9)",
                    bordercolor=color,
                    borderwidth=2,
                    row=row, col=1
                )
        
        # Set RSI y-axis range
        fig.update_yaxes(
            title_text="RSI",
            range=[0, 100],
            row=row, col=1
        )
    
    def _add_suspicious_highlights(self, fig, df: pd.DataFrame, suspicious_periods: List[Dict], row: int):
        """Add visual highlights for suspicious periods."""
        
        for period in suspicious_periods:
            timestamp = period['timestamp']
            spike_ratio = period.get('volume_spike_ratio', 1.0)
            
            # Color intensity based on spike ratio
            if spike_ratio > 2.0:
                line_color = '#FF0000'  # Red for extreme spikes
                annotation = "Extreme"
            elif spike_ratio > 1.5:
                line_color = '#FF8800'  # Orange for high spikes
                annotation = "High"
            else:
                line_color = '#FFFF00'  # Yellow for standard spikes
                annotation = "Standard"
            
            # Add vertical line
            fig.add_shape(
                type="line",
                x0=timestamp,
                x1=timestamp,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(
                    color=line_color,
                    width=2,
                    dash="dash"
                ),
                row=row, col=1
            )
    
    def create_analysis_report(self, result: VolumeAnalysisResult) -> str:
        """Create an HTML analysis report with charts and findings."""
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Volume Analysis Report - {result.pair.upper()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #1e1e1e; color: #ffffff; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .metrics {{ display: flex; justify-content: space-around; margin: 20px 0; }}
                .metric {{ background-color: #2e2e2e; padding: 15px; border-radius: 8px; text-align: center; }}
                .alerts {{ margin: 20px 0; }}
                .alert {{ padding: 10px; margin: 5px 0; border-radius: 5px; }}
                .alert.critical {{ background-color: #ff4444; }}
                .alert.warning {{ background-color: #ff8800; }}
                .alert.info {{ background-color: #4488ff; }}
                .chart-container {{ margin: 30px 0; }}
                .method-info {{ background-color: #2e2e2e; padding: 15px; border-radius: 8px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Volume Analysis Report</h1>
                <h2>{result.pair.upper()} - {result.timeframe}</h2>
                <p>Analysis completed on {result.analysis_timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Confidence Score: <strong>{result.confidence_score:.1%}</strong></p>
            </div>
            
            <div class="method-info">
                <h3>Detection Method: RSI-Enhanced Mean + 4×Standard Deviation</h3>
                <p>This analysis combines statistical volume spike detection with RSI market context for intelligent alerts:</p>
                <ul>
                    <li><strong>🐻 Bearish Alerts:</strong> Volume spike + RSI > 70 (potential market top)</li>
                    <li><strong>🐂 Bullish Alerts:</strong> Volume spike + RSI < 30 (potential market bottom)</li>
                    <li><strong>📊 Standard Alerts:</strong> Volume spikes without RSI extremes</li>
                </ul>
                <p>A volume is considered suspicious when it exceeds the rolling mean plus 4 times the standard deviation, calculated over a 25-period window.</p>
            </div>
            
            <div class="metrics">
                <div class="metric">
                    <h3>Total Periods</h3>
                    <p><strong>{result.metrics.get('total_periods', 0)}</strong></p>
                </div>
                <div class="metric">
                    <h3>Suspicious Periods</h3>
                    <p><strong>{result.metrics.get('suspicious_periods_count', 0)}</strong></p>
                </div>
                <div class="metric">
                    <h3>Suspicious %</h3>
                    <p><strong>{result.metrics.get('suspicious_percentage', 0):.1f}%</strong></p>
                </div>
                <div class="metric">
                    <h3>Max Spike Ratio</h3>
                    <p><strong>{result.metrics.get('max_spike_ratio', 0):.1f}x</strong></p>
                </div>
            </div>
            
            <div class="alerts">
                <h3>Alerts</h3>
        """
        
        for alert in result.alerts:
            alert_class = alert.get('level', 'info')
            html_content += f"""
                <div class="alert {alert_class}">
                    <strong>{alert['type'].upper()}</strong>: {alert['message']}
                </div>
            """
        
        html_content += """
            </div>
            
            <div class="chart-container">
        """
        
        # Add the comprehensive chart
        chart_html = self.create_analysis_chart(result)
        html_content += chart_html
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        return html_content


