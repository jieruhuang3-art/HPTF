#!/usr/bin/env bash
set -e

python -m hptf.training.finetune \
  --config configs/finetune_tor.yaml
