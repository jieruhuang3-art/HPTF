#!/usr/bin/env bash
set -e

python -m hptf.evaluation.evaluate \
  --config configs/finetune_tor.yaml \
  --checkpoint checkpoints/tor/best_model.bin \
  --output results/tor_metrics.json
