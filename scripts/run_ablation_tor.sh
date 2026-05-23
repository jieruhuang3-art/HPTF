#!/usr/bin/env bash
# TOR ablation study script for HPTF.
# Adjust the configuration for the target ablation setting.
set -e

python -m hptf.training.finetune --config configs/finetune_tor.yaml --flowsem_no_field_axis_transformer
python -m hptf.training.finetune --config configs/finetune_tor.yaml --flowsem_no_temporal_transformer
python -m hptf.training.finetune --config configs/finetune_tor.yaml --flowsem_no_dual_axis_transformer
