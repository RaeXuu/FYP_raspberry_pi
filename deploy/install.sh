#!/bin/bash
# 安装 heartbeat 和 watchdog 服务到 systemd
# 在 Pi 上执行：bash deploy/install.sh

set -e

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing heartbeat.service..."
sudo cp "$DEPLOY_DIR/heartbeat.service" /etc/systemd/system/heartbeat.service

echo "Installing watchdog.service..."
sudo cp "$DEPLOY_DIR/watchdog.service" /etc/systemd/system/watchdog.service

sudo systemctl daemon-reload

sudo systemctl enable heartbeat
sudo systemctl enable watchdog

sudo systemctl start heartbeat
sudo systemctl start watchdog

echo ""
echo "Done. Status:"
sudo systemctl status heartbeat --no-pager
echo ""
sudo systemctl status watchdog --no-pager
