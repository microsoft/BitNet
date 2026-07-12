## Deployment and system setup

This document describes how to set up the `bitnet` system user, systemd unit, logrotate and environment files.

1. As root, run the provided setup script:

```bash
sudo bash deploy/setup_system.sh
```

This will:
- create a system user `bitnet` (no login)
- create `/var/log/bitnet` owned by `bitnet`
- create `/etc/bitnet/bitnet.env` with example variables
- install `/etc/systemd/system/bitnet.service` and reload systemd

2. Edit `/etc/bitnet/bitnet.env` to point `MODEL_PATH` to the correct model file and set `LOG_TO_JOURNAL=0` if you prefer file logging.

3. Enable and start the service:

```bash
sudo systemctl enable --now bitnet.service
sudo journalctl -u bitnet -f
```

4. Install logrotate config (already included in `deploy/logrotate/bitnet`):

```bash
sudo cp deploy/logrotate/bitnet /etc/logrotate.d/bitnet
```

Notes:
- The systemd unit sets `MemoryMax=4G` and `CPUQuota=80%` by default; adjust for your host.
- To run the service as file-logging, set `LOG_TO_JOURNAL=0` in `/etc/bitnet/bitnet.env`.
