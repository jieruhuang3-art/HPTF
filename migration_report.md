# Migration Report

## Paths

- Source project path: `/root/ET-BERT-main`
- New project path: `/root/HPTF`

## Copied Files

| New file | Source file |
|---|---|
| `hptf/models/hptf_encoder.py` | `/root/ET-BERT-main/fine-tuning/run_classifier.py` |
| `hptf/training/finetune.py` | `/root/ET-BERT-main/fine-tuning/run_classifier.py` |
| `hptf/training/pretrain.py` | `/root/ET-BERT-main/pretrain_traffic_mlm_aux.py` |
| `hptf/data/pcap_to_flow.py` | `/root/ET-BERT-main/data_process/pcap_to_flow_tsv.py` |
| `hptf/data/tokenizer.py` | `/root/ET-BERT-main/uer/utils/tokenizers.py` |
| `hptf/uer/` | `/root/ET-BERT-main/uer/` |
| `configs/bert_base_config.json` | `/root/ET-BERT-main/bert_base_config.json` |
| `LICENSE` | `/root/ET-BERT-main/LICENSE` |

## Generated Wrapper Files

- `hptf/models/embeddings.py`
- `hptf/models/gated_fusion.py`
- `hptf/models/intra_packet_transformer.py`
- `hptf/models/inter_packet_transformer.py`
- `hptf/models/classification_head.py`
- `hptf/data/dataset.py`
- `hptf/data/collate.py`
- `hptf/training/losses.py`
- `hptf/training/trainer.py`
- `hptf/evaluation/metrics.py`
- `hptf/evaluation/evaluate.py`
- `hptf/evaluation/visualization.py`
- `hptf/utils/*.py`

## Found Real Implementations

- HPTF / enhanced FlowSem classifier: `FlowSemMAEClassifier` in `/root/ET-BERT-main/fine-tuning/run_classifier.py`.
- Contextual Traffic Transformer Encoder: optional `self.encoder = str2encoder[args.encoder](args)` used by `--flowsem_use_context_encoder`.
- Gated Side-channel Feature Fusion: `_fuse_aux_features` with `aux_gate`.
- Intra-packet Field-axis Transformer Encoder: `self.field_encoder`.
- Inter-packet Temporal Transformer Encoder: `self.temporal_encoder`.
- Multi-branch Representation Aggregation: packet, statistics, direct token, and CLS branches in `FlowSemMAEClassifier.forward`.
- Masked Token Prediction: `mlm_head` and `F.cross_entropy` in `/root/ET-BERT-main/pretrain_traffic_mlm_aux.py`.
- Inter-arrival Time Prediction: `time_head` and `F.mse_loss`.
- Packet Length Prediction: `length_head` and `F.mse_loss`.
- Supervised fine-tuning: `CrossEntropyLoss` through `classification_loss`.
- PCAP/flow TSV preprocessing: `/root/ET-BERT-main/data_process/pcap_to_flow_tsv.py`.

## Missing or Not Directly Found

- No source directory `/root/ET-BERT-main/model`.
- No source directory `/root/ET-BERT-main/mamba_flow`.
- No source directory `/root/ET-BERT-main/preprocess`.
- No source directory `/root/ET-BERT-main/scripts`.
- No standalone source file named `HPTFEncoder` was found; the implementation is integrated as `FlowSemMAEClassifier`.
- No direct CLI switch named `--ablation`.
- No direct fine-tuning switch named `remove_gated_fusion`.
- No direct fine-tuning switch named `remove_time_length_features`.
- No single direct switch named `remove_contextual_residual`; contextual encoder and CLS residual are controlled by opt-in flags.

## Import Changes

- Replaced `from uer...` imports with `from hptf.uer...`.
- Replaced dynamic loading of `fine-tuning/run_classifier.py` in pretraining with `from hptf.training import finetune`.
- Added YAML `--config` expansion around the original training and pretraining argument parsers; model forward logic and default module behavior were not changed.

## Generated Scripts

- `scripts/preprocess.sh`
- `scripts/pretrain.sh`
- `scripts/finetune_tor.sh`
- `scripts/finetune_all.sh`
- `scripts/evaluate.sh`
- `scripts/run_ablation_tor.sh`

## Required Statements

- 本复现仓库仅整理和复制当前项目中的真实 HPTF 相关实现。
- 未复制原始数据集、模型权重、训练日志和临时缓存文件。
- 未编造实验结果。
- README 中的结果表来自论文中已有实验结果。
- 如果某个消融开关在当前代码中不存在，本文档已明确记录。

## Tested Commands

| Command | Result |
|---|---|
| `python -m compileall hptf` | PASS |
| `python -m hptf.training.pretrain --help` | PASS |
| `python -m hptf.training.finetune --help` | PASS |
| `python -m hptf.evaluation.evaluate --help` | PASS |
| `bash -n scripts/preprocess.sh` | PASS |
| `bash -n scripts/pretrain.sh` | PASS |
| `bash -n scripts/finetune_tor.sh` | PASS |
| `bash -n scripts/finetune_all.sh` | PASS |
| `bash -n scripts/evaluate.sh` | PASS |
| `bash -n scripts/run_ablation_tor.sh` | PASS |
| `tree -L 4 /root/HPTF` | SKIPPED: tree not installed |
| `find /root/HPTF -maxdepth 4 -type f | sort > /root/HPTF/project_tree.txt` | PASS |

## Second Round Cleanup - 2026-05-23 UTC

### Scope

- Continued refining `/root/HPTF` only.
- Did not modify `/root/ET-BERT-main`.
- Kept `FlowSemMAEClassifier` as the canonical real model class.

### Model File Organization

- `hptf/models/hptf_encoder.py` remains the canonical copied implementation from `/root/ET-BERT-main/fine-tuning/run_classifier.py`.
- `hptf/models/embeddings.py` was updated as a re-export wrapper for copied UER embeddings.
- `hptf/models/gated_fusion.py` was updated as a thin helper around `FlowSemMAEClassifier._fuse_aux_features`.
- `hptf/models/intra_packet_transformer.py` was updated as a thin helper exposing `FlowSemMAEClassifier.field_encoder`.
- `hptf/models/inter_packet_transformer.py` was updated as a thin helper exposing `FlowSemMAEClassifier.temporal_encoder`.
- `hptf/models/classification_head.py` was updated as a thin helper exposing `output_layer_1`, `output_layer_2`, and `classification_loss`.
- `hptf/models/__init__.py` now exports the copied classifier classes.

### Files That Are Wrappers

- `hptf/models/embeddings.py`
- `hptf/models/gated_fusion.py`
- `hptf/models/intra_packet_transformer.py`
- `hptf/models/inter_packet_transformer.py`
- `hptf/models/classification_head.py`

These wrappers do not replace or alter the real forward computation. They document and expose modules that remain implemented inside `FlowSemMAEClassifier`.

### Real Implementations Confirmed

- Masked Token Prediction: `TrafficMlmAuxPretrainer.mlm_head` and `F.cross_entropy` in `hptf/training/pretrain.py`.
- Inter-arrival Time Prediction: `TrafficMlmAuxPretrainer.time_head` and `F.mse_loss`.
- Packet Length Prediction: `TrafficMlmAuxPretrainer.length_head` and `F.mse_loss`.
- Pretraining objective: `mlm_loss + args.aux_loss_weight * (time_loss + len_loss)`, corresponding to equal `lambda_t` and `lambda_l` when configured that way.
- Checkpoint loading for fine-tuning: `load_or_initialize_parameters` and `--pretrained_model_path`.
- Classification head training: classifier `output_layer_1`, `output_layer_2`, and `classification_loss`.
- CrossEntropyLoss: `F.cross_entropy` in `classification_loss`.
- Accuracy and confusion matrix: original `finetune.evaluate`.
- Precision, Recall, Macro-F1, and Confusion Matrix export: `hptf/evaluation/metrics.py` and `hptf/evaluation/evaluate.py`.

### Modules Without Independent Standalone Implementation

- No independent source class named `HPTFEncoder` was found.
- No independent standalone module file for gated fusion was found; fusion is a method inside `FlowSemMAEClassifier`.
- No independent standalone module file for field-axis or temporal-axis Transformer was found; both are attributes inside `FlowSemMAEClassifier`.
- No direct `--ablation` argument was found.
- No exact direct switches were found for `remove_gated_fusion`, `remove_time_length_features`, or `remove_contextual_residual`.

### New Documentation

- Added `docs/code_mapping.md`.
- Updated `docs/model_overview.md`.
- Updated `README.md`.
- Updated `scripts/run_ablation_tor.sh` as a template script.

### Second Round Retest

| Command | Result |
|---|---|
| `python -m compileall hptf` | PASS |
| `python -m hptf.training.pretrain --help` | PASS |
| `python -m hptf.training.finetune --help` | PASS |
| `python -m hptf.evaluation.evaluate --help` | PASS |
| `bash -n scripts/preprocess.sh` | PASS |
| `bash -n scripts/pretrain.sh` | PASS |
| `bash -n scripts/finetune_tor.sh` | PASS |
| `bash -n scripts/finetune_all.sh` | PASS |
| `bash -n scripts/evaluate.sh` | PASS |
| `bash -n scripts/run_ablation_tor.sh` | PASS |
| `find /root/HPTF -maxdepth 4 -type f | sort > /root/HPTF/project_tree.txt` | PASS |

## GitHub Release Cleanup - 2026-05-23 UTC

### Files Updated

- `README.md`
- `docs/code_mapping.md`
- `docs/model_overview.md`
- `docs/reproduction.md`
- `docs/ablation.md`
- `docs/github_release_checklist.md`
- `scripts/run_ablation_tor.sh`
- `.gitignore`

### Dependency Notes

- `torch`, `six`, and `packaging` come from the original project requirements.
- `numpy`, `tqdm`, and `scapy` are required by migrated preprocessing/training code.
- `pyyaml` is required by the migrated YAML config wrapper.
- `scikit-learn`, `matplotlib`, and `seaborn` are used by evaluation metrics and visualization helpers.
- `pandas` is retained as a common tabular-data dependency for reproducibility workflows, although the core migrated TSV readers use standard-library CSV parsing.

### Release Policy Notes

- Data and checkpoints remain excluded from the repository.
- No nonexistent `--ablation` parameter was added.
- Result tables were not modified except for explanatory wording; no experiment values were fabricated.

### GitHub Release Retest

| Command | Result |
|---|---|
| `python -m compileall hptf` | PASS |
| `python -m hptf.training.pretrain --help` | PASS |
| `python -m hptf.training.finetune --help` | PASS |
| `python -m hptf.evaluation.evaluate --help` | PASS |
| `bash -n scripts/preprocess.sh` | PASS |
| `bash -n scripts/pretrain.sh` | PASS |
| `bash -n scripts/finetune_tor.sh` | PASS |
| `bash -n scripts/finetune_all.sh` | PASS |
| `bash -n scripts/evaluate.sh` | PASS |
| `bash -n scripts/run_ablation_tor.sh` | PASS |
| `find /root/HPTF -maxdepth 4 -type f | sort > /root/HPTF/project_tree.txt` | PASS |
