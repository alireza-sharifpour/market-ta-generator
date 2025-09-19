#!/bin/bash

# Setup script for volume analysis cron job
# This script sets up a cron job to run volume analysis every 5 minutes

set -e

echo "Setting up volume analysis cron job..."

# Get the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CRON_SCRIPT="$PROJECT_DIR/volume_analysis_cron.py"
VENV_PYTHON="$PROJECT_DIR/venv/bin/python"

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

# Make cron script executable
chmod +x "$CRON_SCRIPT"

# Create log directory
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"

# Create cron job entry
CRON_ENTRY="*/5 * * * * cd $PROJECT_DIR && $VENV_PYTHON $CRON_SCRIPT >> $LOG_DIR/volume_analysis_cron.log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "volume_analysis_cron.py"; then
    echo "⚠️  Volume analysis cron job already exists"
    echo "Current cron jobs:"
    crontab -l | grep "volume_analysis_cron.py"
    echo ""
    read -p "Do you want to replace it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Remove existing cron job
        (crontab -l 2>/dev/null | grep -v "volume_analysis_cron.py") | crontab -
        echo "✅ Removed existing cron job"
    else
        echo "❌ Setup cancelled"
        exit 1
    fi
fi

# Add new cron job
(crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -

echo "✅ Volume analysis cron job added successfully!"
echo ""
echo "Cron job details:"
echo "  - Script: $CRON_SCRIPT"
echo "  - Schedule: Every 5 minutes"
echo "  - Log file: $LOG_DIR/volume_analysis_cron.log"
echo "  - Python: $VENV_PYTHON"
echo ""
echo "Current cron jobs:"
crontab -l | grep "volume_analysis_cron.py"
echo ""
echo "To view logs:"
echo "  tail -f $LOG_DIR/volume_analysis_cron.log"
echo ""
echo "To remove the cron job:"
echo "  crontab -e  # and delete the volume analysis line"
echo ""
echo "To test the script manually:"
echo "  cd $PROJECT_DIR && $VENV_PYTHON $CRON_SCRIPT"
