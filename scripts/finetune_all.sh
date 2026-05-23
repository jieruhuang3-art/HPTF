#!/usr/bin/env bash
set -e

bash scripts/finetune_tor.sh
python -m hptf.training.finetune --config configs/finetune_cstnet_tls13.yaml
python -m hptf.training.finetune --config configs/finetune_cesnet_w2021_40.yaml
python -m hptf.training.finetune --config configs/finetune_cesnet_w2021_41.yaml
python -m hptf.training.finetune --config configs/finetune_crossplatform_android.yaml
python -m hptf.training.finetune --config configs/finetune_crossplatform_ios.yaml
