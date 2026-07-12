Run scripts for BitNet

Shell runner

Usage:

```bash
chmod +x scripts/run_bitnet.sh
./scripts/run_bitnet.sh models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf "Hello" 4 128
```

Python runner

```bash
python3 scripts/run_bitnet.py --model models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf --prompt "Hello" --threads 4 --tokens 128
```

Systemd

To run as a service (requires root):

```bash
sudo cp deploy/bitnet.service /etc/systemd/system/bitnet.service
sudo systemctl daemon-reload
sudo systemctl enable --now bitnet.service
```
