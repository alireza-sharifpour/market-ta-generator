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
        logger.info("VolumeChartGenerator initialized")
    
    def create_analysis_chart(self, result: VolumeAnalysisResult) -> str:
        """
        Create a comprehensive chart showing price, volume, and mean+std analysis.
        
        Args:
            result: VolumeAnalysisResult containing analysis data
            
        Returns:
            HTML string of the interactive chart
        """
        logger.info(f"Creating volume analysis chart for {result.pair}")
        
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
        self._add_volume_chart_mean_std(fig, df, suspicious_periods, row=2)
        
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
        
        logger.info("Volume analysis chart created successfully")
        return fig.to_html(include_plotlyjs=True, div_id="volume-analysis-chart")
    
    def create_analysis_chart_base64(self, result: VolumeAnalysisResult) -> str:
        """
        Create chart and return as base64 encoded PNG similar to market-ta-generator.
        
        Args:
            result: VolumeAnalysisResult containing analysis data
            
        Returns:
            Base64 encoded PNG image
        """
        logger.info(f"Creating volume analysis chart as base64 for {result.pair}")
        
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
        self._add_volume_chart_mean_std(fig, df, suspicious_periods, row=2)
        
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
        
        logger.info("Volume analysis chart created as base64 PNG")
        return f"data:image/png;base64,{img_base64}"
    
    def _add_price_chart(self, fig, df: pd.DataFrame, suspicious_periods: List[Dict], row: int):
        """Add price candlestick chart with suspicious period highlighting."""
        
        # Candlestick chart
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
        
        # Add suspicious period highlights to price chart
        self._add_suspicious_highlights(fig, df, suspicious_periods, row)
        
        # Update y-axes
        fig.update_yaxes(title_text="Price", row=row, col=1)
    
    def _add_volume_chart_mean_std(self, fig, df: pd.DataFrame, suspicious_periods: List[Dict], row: int):
        """Add volume chart with mean+std analysis."""
        
        # Check if we have valid data
        if df.empty or 'Volume' not in df.columns:
            logger.warning("No valid volume data to display")
            return
        
        # Volume bars with color coding for suspicious periods
        volume_colors = []
        for i in range(len(df)):
            if any(sp['index'] == i for sp in suspicious_periods):
                volume_colors.append(self.config["suspicious_color"])
            else:
                volume_colors.append(self.config["volume_color"])
        
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
        
        # Volume spike threshold line (Mean + 4×Std)
        if 'volume_spike_threshold' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['volume_spike_threshold'],
                    mode='lines',
                    name='Spike Threshold (Mean + 4×Std)',
                    line=dict(color='red', width=2, dash='dash'),
                    hovertemplate='Spike Threshold: %{y:,.0f}<extra></extra>'
                ),
                row=row, col=1
            )
        
        # Add annotations for suspicious periods on volume chart
        for period in suspicious_periods:
            fig.add_annotation(
                x=period['timestamp'],
                y=period['volume'],
                text=f"Spike: {period.get('volume_spike_ratio', 1.0):.1f}x",
                showarrow=True,
                arrowhead=2,
                arrowcolor=self.config["suspicious_color"],
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=self.config["suspicious_color"],
                borderwidth=1,
                row=row, col=1
            )
        
        fig.update_yaxes(title_text="Volume", row=row, col=1)
    
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
                <h3>Detection Method: Mean + 4×Standard Deviation</h3>
                <p>This analysis uses statistical volume spike detection. A volume is considered suspicious when it exceeds the rolling mean plus 4 times the standard deviation, calculated over a 25-period window.</p>
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


