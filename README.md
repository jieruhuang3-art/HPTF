# HPTF: Hierarchical Packet-Temporal Fusion for Encrypted Traffic Classification

## Overview

HPTF is a hierarchical traffic representation learning framework for encrypted traffic classification. It organizes encrypted traffic into packet-aware token sequences and jointly models contextual token dependencies, intra-packet local structures, inter-packet temporal dependencies, and time-length side-channel features. The framework supports self-supervised pre-training and supervised fine-tuning for encrypted traffic classification tasks.

HPTF 是一种面向加密流量分类的层级流量表示学习框架。该方法将加密流量组织为具备 packet 边界感知的 token 序列，并联合建模流量 token 上下文、包内局部结构、包间时序依赖以及时间—长度传输侧信道特征。该仓库提供 HPTF 的预训练、监督微调和评估流程，用于论文方法复现。

Flat token sequences are limited in distinguishing intra-packet structures and inter-packet temporal dependencies. Time-length side-channel features provide useful transport-level cues for encrypted traffic classification. HPTF addresses these issues through hierarchical packet-temporal modeling and gated feature fusion.

## Framework

1. Data Preprocessing
   - Flow aggregation
   - Sliding-window packet segmentation
   - Traffic tokenization
   - Packet boundary and time-length feature construction

2. Self-supervised Pre-training
   - Masked token prediction
   - Inter-arrival time prediction
   - Packet length prediction

3. HPTF Encoder
   - Traffic token embedding
   - Contextual Traffic Transformer Encoder
   - Gated Side-channel Feature Fusion
   - Intra-packet Field-axis Transformer Encoder
   - Inter-packet Temporal Transformer Encoder
   - Multi-branch Representation Aggregation

4. Supervised Fine-tuning
   - Flow-level representation
   - Classification head
   - Cross-entropy optimization

## Installation

```bash
conda create -n hptf python=3.9 -y
conda activate hptf
pip install -r requirements.txt
```

## Data Preparation

Please prepare encrypted traffic datasets in PCAP or TSV format and organize them under the `data` directory. The preprocessing script converts raw traffic flows into packet-aware traffic token sequences with auxiliary time-length features.

Example datasets:

- CSTNET-TLS 1.3
- TOR
- CESNET-TLS22
- ISCX-VPN / ISCX-NONVPN
- CrossPlatform Android / IOS

```bash
bash scripts/preprocess.sh
```

## Pre-training

```bash
bash scripts/pretrain.sh
```

The self-supervised objective combines masked token prediction and time-length behavior prediction:

```text
L_pre = L_MLM + lambda_t L_time + lambda_l L_len
```

## Fine-tuning

```bash
bash scripts/finetune_tor.sh
bash scripts/finetune_all.sh
```

## Evaluation

```bash
bash scripts/evaluate.sh
```

Evaluation metrics include Accuracy, Precision, Recall, and Macro-F1.

## Ablation Study

The ablation settings are designed to analyze the contribution of each major component in HPTF:

- w/o Intra-packet Field-axis Transformer Encoder
- w/o Inter-packet Temporal Transformer Encoder
- w/o Gated Side-channel Feature Fusion
- w/o Time-Length Side-channel Features
- w/o Multi-objective Pre-training
- w/o Contextual Encoder and Residual Path

```bash
bash scripts/run_ablation_tor.sh
```

## Results

HPTF achieves competitive performance across multiple encrypted traffic classification benchmarks. The results show that hierarchical packet-temporal modeling and gated time-length feature fusion improve the representation ability of encrypted traffic, especially on TLS service classification, anonymous traffic classification, VPN/non-VPN traffic classification, and mobile application traffic classification tasks.

Performance may vary across datasets due to platform-specific traffic behavior and distribution shifts.

## Code Mapping

The correspondence between paper modules and implementation components is summarized in `docs/code_mapping.md`.

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

This project is released under the MIT License.
