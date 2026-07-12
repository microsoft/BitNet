#!/usr/bin/env bash
set -euo pipefail

# Usage: sudo ./deploy/setup_system.sh
# Creates system user 'bitnet', log dir, env file and installs systemd unit.

SERVICE_SRC="$(cd "$(dirname "$0")" && pwd)/bitnet.service"
SERVICE_DST="/etc/systemd/system/bitnet.service"
ENV_DIR="/etc/bitnet"
ENV_FILE="$ENV_DIR/bitnet.env"
LOG_DIR="/var/log/bitnet"

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root: sudo $0"
  exit 1
fi

if ! id -u bitnet >/dev/null 2>&1; then
  echo "Creating system user 'bitnet' (no login)"
  useradd --system --no-create-home --shell /usr/sbin/nologin --user-group bitnet
else
  echo "User 'bitnet' already exists"
fi

echo "Creating log dir: $LOG_DIR"
mkdir -p "$LOG_DIR"
chown bitnet:bitnet "$LOG_DIR"
chmod 0755 "$LOG_DIR"

echo "Creating env dir: $ENV_DIR"
mkdir -p "$ENV_DIR"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'EOF'
# Example BitNet environment file
MODEL_PATH=/workspaces/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf
THREADS=4
PROMPT="You are a helpful assistant."
# LOG_TO_JOURNAL=1
# Resource profile defaults (choose one: small, medium, large)
# For small machines (e.g., 2 CPU, 4GB RAM):
# MEMORY_MAX=2G
# CPU_QUOTA=50%
# LIMIT_NOFILE=4096
# For medium machines (4 CPU, 8GB RAM):
# MEMORY_MAX=4G
# CPU_QUOTA=80%
# LIMIT_NOFILE=65536
# For large machines (8+ CPU, 16+GB RAM):
# MEMORY_MAX=12G
# CPU_QUOTA=100%
# LIMIT_NOFILE=1048576
EOF
  chown root:root "$ENV_FILE"
  chmod 0644 "$ENV_FILE"
  echo "Created $ENV_FILE"
else
  echo "$ENV_FILE already exists, leaving unchanged"
fi

echo "Installing systemd unit to $SERVICE_DST"
cp "$SERVICE_SRC" "$SERVICE_DST"
chmod 0644 "$SERVICE_DST"

echo "Reloading systemd daemon"
systemctl daemon-reload

echo "Enable service (but not starting automatically): systemctl enable bitnet.service"
echo "To start now: systemctl start bitnet.service"

echo "Setup complete."
