# HPTF: Hierarchical Packet-Temporal Fusion for Encrypted Traffic Classification

## Overview

HPTF is a hierarchical traffic representation learning framework for encrypted traffic classification. It models contextual traffic token dependencies, intra-packet local structures, inter-packet temporal dependencies, and side-channel transport features such as inter-arrival time and packet length.

本仓库提供论文《面向加密流量分类的层级流量表示学习方法》的复现代码，包括数据预处理、预训练、监督微调、评估和消融实验。

This repository is organized from real project code in `/root/ET-BERT-main`. It does not include raw datasets, processed datasets, model weights, training logs, temporary caches, or fabricated experiment outputs.

## Implementation Provenance

The source project implements the paper HPTF model under the real class name `FlowSemMAEClassifier`. This class corresponds to the HPTF encoder plus classification model used in the paper. The repository keeps that class name and does not rename it to a nonexistent `HPTFEncoder`.

The structural files under `hptf/models/` are transparent wrappers or re-exports around the copied implementation:

- `hptf/models/hptf_encoder.py`: canonical copied model implementation.
- `hptf/models/embeddings.py`: traffic embedding re-export from the copied UER/ET-BERT dependency.
- `hptf/models/gated_fusion.py`: helper access to the gated side-channel fusion method inside `FlowSemMAEClassifier`.
- `hptf/models/intra_packet_transformer.py`: helper access to `FlowSemMAEClassifier.field_encoder`.
- `hptf/models/inter_packet_transformer.py`: helper access to `FlowSemMAEClassifier.temporal_encoder`.
- `hptf/models/classification_head.py`: helper access to the original two-layer classification head.

The wrappers exist for paper-module navigation and GitHub readability; they do not replace or alter the real model forward logic.

## Framework

1. Data Preprocessing
   - PCAP / Flow TSV input
   - Flow Aggregation
   - Sliding-window Packet Segmentation
   - Traffic Tokenization
   - Packet Boundary, Inter-arrival Time, Packet Length

2. Self-supervised Pre-training
   - Masked Token Prediction
   - Inter-arrival Time Prediction
   - Packet Length Prediction
   - `L_pre = L_MLM + lambda_t L_time + lambda_l L_len`

3. HPTF Encoder
   - Traffic Token Embedding
   - Contextual Traffic Transformer Encoder
   - Gated Side-channel Feature Fusion
   - Intra-packet Field-axis Transformer Encoder
   - Inter-packet Temporal Transformer Encoder
   - Multi-branch Representation Aggregation

4. Supervised Fine-tuning
   - HPTF Encoder
   - Classification Head
   - Cross-Entropy Loss
   - Application / Traffic Classification

## Repository Structure

- `hptf/models/`: HPTF/FlowSem model implementation copied from the current ET-BERT project plus transparent module wrappers.
- `hptf/data/`: PCAP-to-TSV preprocessing, tokenizer exports, dataset readers, and batch collation helpers.
- `hptf/training/`: traffic MLM+aux pre-training and supervised fine-tuning entry points.
- `hptf/evaluation/`: metric calculation, evaluation entry point, and visualization helpers.
- `hptf/utils/`: configuration, logging, seed, and file IO helpers.
- `configs/`: reproducible pre-training and fine-tuning configuration templates.
- `scripts/`: shell scripts for preprocessing, pre-training, fine-tuning, evaluation, and TOR ablation templates.
- `data/`: empty dataset layout only. Raw and processed datasets are not included.
- `docs/`: model, data format, reproduction, ablation, code mapping, and release checklist notes.
- `results/`: result-table template and generated evaluation outputs.

## Installation

```bash
conda create -n hptf python=3.9 -y
conda activate hptf
pip install -r requirements.txt
```

## Data Preparation

Datasets are not released with this repository. Please download the public datasets according to their licenses and place them under the recommended layout:

- CSTNET-TLS 1.3
- TOR
- CESNET-TLS22 W-2021-40
- CESNET-TLS22 W-2021-41
- ISCX-NONVPN
- ISCX-VPN
- CrossPlatform Android
- CrossPlatform IOS

Recommended directories:

```text
data/raw/
data/processed/
data/splits/
```

Run preprocessing:

```bash
bash scripts/preprocess.sh
```

## Pre-training

```bash
bash scripts/pretrain.sh
```

The implemented pre-training code contains:

- Masked Token Prediction through `mlm_head` and `F.cross_entropy`.
- Inter-arrival Time Prediction through `time_head` and `F.mse_loss`.
- Packet Length Prediction through `length_head` and `F.mse_loss`.

The implemented objective is:

```text
L_pre = L_MLM + lambda_t L_time + lambda_l L_len
```

In the migrated implementation, `lambda_t` and `lambda_l` share the original `--aux_loss_weight` argument when they use the same value.

## Fine-tuning

```bash
bash scripts/finetune_tor.sh
bash scripts/finetune_all.sh
```

Fine-tuning supports loading a pre-trained checkpoint through the original `--pretrained_model_path` path, exposed in YAML as `model.pretrained_checkpoint`. Classification heads and `CrossEntropyLoss` are implemented in the copied source. Accuracy and a confusion matrix are printed by the original fine-tuning entry point; Precision, Recall, Macro-F1, and Confusion Matrix export are provided by `hptf.evaluation.metrics` and `hptf.evaluation.evaluate`.

Some functionality, including token embeddings, Transformer encoders, optimizers, schedulers, tokenizers, and model saving, comes from the copied UER/ET-BERT dependency under `hptf/uer/`.

## Evaluation

```bash
bash scripts/evaluate.sh
```

Evaluation reports:

- Accuracy
- Precision
- Recall
- Macro-F1
- Confusion Matrix

## Ablation Study

TOR core ablations:

- w/o Intra-packet Field-axis Transformer Encoder
- w/o Inter-packet Temporal Transformer Encoder
- w/o Gated Side-channel Feature Fusion
- w/o Time-Length Side-channel Features
- w/o Multi-objective Masked Pre-training
- w/o Contextual Encoder and Residual Path

```bash
bash scripts/run_ablation_tor.sh
```

The ablation script is a template. Some paper ablations are not directly implemented as CLI switches in the migrated codebase. Do not assume a paper ablation is runnable unless the corresponding switch is documented in `docs/code_mapping.md` and `migration_report.md`.

## Module-Code Mapping

The module-to-code map is maintained in `docs/code_mapping.md`. It identifies the exact code locations for `FlowSemMAEClassifier.encoder`, `_fuse_aux_features`, `field_encoder`, `temporal_encoder`, `forward`, the classification head, pre-training heads, and supervised fine-tuning loss.

## Expected Results

The table below is copied from existing paper results, not newly generated or fabricated.

| Dataset | ET-BERT Macro-F1 | HPTF Macro-F1 |
|---|---:|---:|
| CSTNET-TLS 1.3 | 0.8083 | 0.8231 |
| CrossPlatform Android | 0.7160 | 0.7527 |
| CrossPlatform IOS | 0.7268 | 0.7149 |
| TOR | 0.9365 | 0.9795 |
| CESNET-TLS22 W-2021-40 | 0.9568 | 0.9675 |
| CESNET-TLS22 W-2021-41 | 0.8993 | 0.9701 |
| ISCX-NONVPN | 0.7787 | 0.7973 |
| ISCX-VPN | 0.9736 | 0.9816 |

CrossPlatform IOS is retained as reported: HPTF Macro-F1 is lower than ET-BERT on this dataset.

## Citation

```bibtex
@article{hptf2026,
  title={Hierarchical Packet-Temporal Fusion for Encrypted Traffic Classification},
  author={Anonymous},
  journal={},
  year={2026}
}
```

## License

This repository follows the MIT License used by the source ET-BERT project.
