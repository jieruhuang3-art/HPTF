#!/usr/bin/env bash
set -e

python -m hptf.training.pretrain \
  --config configs/pretrain.yaml
