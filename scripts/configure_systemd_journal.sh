#!/bin/bash

# Configure systemd journal for better log retention
# Based on 2024 best practices research

echo "Configuring systemd journal retention..."

# Create backup of original config
sudo cp /etc/systemd/journald.conf /etc/systemd/journald.conf.backup

# Create new journald configuration
sudo tee /etc/systemd/journald.conf > /dev/null << 'EOF'
#  This file is part of systemd.
#
#  systemd is free software; you can redistribute it and/or modify it under the
#  terms of the GNU Lesser General Public License as published by the Free
#  Software Foundation; either version 2.1 of the License, or (at your option)
#  any later version.
#
# Entries in this file show the compile time defaults. Local configuration
# should be created by either modifying this file, or by creating "drop-ins" in
# the journald.conf.d/ subdirectory. The latter is generally recommended.
# Defaults can be restored by simply deleting this file and all drop-ins.
#
# Use 'systemd-analyze cat-config systemd/journald.conf' to display the full config.
#
# See journald.conf(5) for details.

[Journal]
# Enable persistent storage
Storage=persistent

# Limit total journal size to 1GB (down from ~3.3GB current usage)
SystemMaxUse=1G

# Keep at least 500MB free space
SystemKeepFree=500M

# Maximum size per journal file (20MB)
SystemMaxFileSize=20M

# Rotate journal files weekly instead of monthly
MaxFileSec=1week

# Keep logs for maximum 2 months
MaxRetentionSec=2month

# Enable compression to save space
Compress=yes

# Forward to syslog for compatibility
ForwardToSyslog=yes
ForwardToKMsg=no
ForwardToConsole=no
ForwardToWall=yes

# Set reasonable limits
TTYPath=/dev/console
MaxLevelStore=debug
MaxLevelSyslog=debug
MaxLevelKMsg=notice
MaxLevelConsole=info
MaxLevelWall=emerg
LineMax=48K
ReadKMsg=yes
Audit=no
EOF

echo "Systemd journal configuration created."
echo "Restarting systemd-journald service..."

# Restart journald to apply new settings
sudo systemctl daemon-reload
sudo systemctl restart systemd-journald

echo "Checking current journal disk usage..."
sudo journalctl --disk-usage

echo "Cleaning up old journal files to respect new limits..."
sudo journalctl --vacuum-size=1G
sudo journalctl --vacuum-time=2months

echo "Final journal disk usage:"
sudo journalctl --disk-usage

echo "Systemd journal configuration complete!"
echo ""
echo "Configuration summary:"
echo "- Maximum total size: 1GB (was ~3.3GB)"
echo "- File rotation: Weekly"
echo "- Maximum retention: 2 months"  
echo "- Compression: Enabled"
echo "- Minimum free space: 500MB"