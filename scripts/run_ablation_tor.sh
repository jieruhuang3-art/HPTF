#!/usr/bin/env bash
# This script is a template for TOR ablation experiments.
# Some ablation switches are not directly implemented in the migrated codebase.
# Please refer to docs/code_mapping.md and migration_report.md before running ablation experiments.
set -e

# Directly supported switches found in the migrated source:
python -m hptf.training.finetune --config configs/finetune_tor.yaml --flowsem_no_field_axis_transformer
python -m hptf.training.finetune --config configs/finetune_tor.yaml --flowsem_no_temporal_transformer
python -m hptf.training.finetune --config configs/finetune_tor.yaml --flowsem_no_dual_axis_transformer

# Not executed here because no exact direct switch was found in the source code:
# - w/o Gated Side-channel Feature Fusion
# - w/o Time-Length Side-channel Features
# - w/o Multi-objective Masked Pre-training
# - w/o Contextual Encoder and Residual Path
