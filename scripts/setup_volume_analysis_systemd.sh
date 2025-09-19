#!/bin/bash

# Setup script for volume analysis systemd service and timer
# This script sets up a systemd timer to run volume analysis every 5 minutes

set -e

echo "Setting up volume analysis systemd service and timer..."

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_FILE="$PROJECT_DIR/scripts/volume-analysis.service"
TIMER_FILE="$PROJECT_DIR/scripts/volume-analysis.timer"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"
CRON_SCRIPT="$PROJECT_DIR/volume_analysis_cron.py"

# Check if virtual environment exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo "❌ Virtual environment not found at $VENV_PYTHON"
    echo "Please create a virtual environment first:"
    echo "  cd $PROJECT_DIR"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Check if cron script exists
if [ ! -f "$CRON_SCRIPT" ]; then
    echo "❌ Cron script not found at $CRON_SCRIPT"
    exit 1
fi

# Update service file with correct paths
sed -i "s|/media/mra/w/w/amiram/projects/market-ta-generator|$PROJECT_DIR|g" "$SERVICE_FILE"

# Copy service and timer files to systemd directory
sudo cp "$SERVICE_FILE" /etc/systemd/system/
sudo cp "$TIMER_FILE" /etc/systemd/system/

# Reload systemd daemon
sudo systemctl daemon-reload

# Enable and start the timer
sudo systemctl enable volume-analysis.timer
sudo systemctl start volume-analysis.timer

echo "✅ Volume analysis systemd service and timer setup complete!"
echo ""
echo "Service details:"
echo "  - Service: volume-analysis.service"
echo "  - Timer: volume-analysis.timer"
echo "  - Schedule: Every 5 minutes"
echo "  - Script: $CRON_SCRIPT"
echo "  - Python: $VENV_PYTHON"
echo ""
echo "Useful commands:"
echo "  # Check timer status"
echo "  sudo systemctl status volume-analysis.timer"
echo ""
echo "  # Check service status"
echo "  sudo systemctl status volume-analysis.service"
echo ""
echo "  # View logs"
echo "  sudo journalctl -u volume-analysis.service -f"
echo ""
echo "  # Run service manually"
echo "  sudo systemctl start volume-analysis.service"
echo ""
echo "  # Stop timer"
echo "  sudo systemctl stop volume-analysis.timer"
echo ""
echo "  # Disable timer"
echo "  sudo systemctl disable volume-analysis.timer"
echo ""
echo "  # Remove service and timer"
echo "  sudo systemctl stop volume-analysis.timer"
echo "  sudo systemctl disable volume-analysis.timer"
echo "  sudo rm /etc/systemd/system/volume-analysis.service"
echo "  sudo rm /etc/systemd/system/volume-analysis.timer"
echo "  sudo systemctl daemon-reload"
