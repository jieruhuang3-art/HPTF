# HPTF: Hierarchical Packet-Temporal Fusion for Encrypted Traffic Classification

HPTF is a hierarchical traffic representation learning framework for encrypted traffic classification. It represents encrypted traffic as packet-aware token sequences and jointly models contextual token dependencies, intra-packet local structures, inter-packet temporal dependencies, and time-length side-channel features.

This repository provides the implementation of HPTF, including data preprocessing, self-supervised pre-training, supervised fine-tuning, evaluation, and ablation study utilities.

HPTF 是一种面向加密流量分类的层级流量表示学习框架。该方法将加密流量表示为具有 packet 边界感知的 token 序列，并联合建模流量 token 上下文、包内局部结构、包间时序依赖以及时间—长度传输侧信道特征。

本仓库提供 HPTF 的数据预处理、预训练、监督微调、评估和消融实验相关代码，用于加密流量分类任务复现。

## Highlights

- Packet-aware encrypted traffic representation.
- Hierarchical modeling of token context, intra-packet structure, and inter-packet temporal dependency.
- Gated fusion of inter-arrival time and packet length features.
- Unified pre-training and fine-tuning pipeline for encrypted traffic classification.

## Framework

### 1. Data Preprocessing

- Flow aggregation
- Sliding-window packet segmentation
- Traffic tokenization
- Packet boundary construction
- Time-length feature construction

### 2. Self-supervised Pre-training

HPTF uses multi-objective pre-training to learn contextual token representations and transport behavior priors.

Pre-training objectives:

- Masked Token Prediction
- Inter-arrival Time Prediction
- Packet Length Prediction

Loss:

```text
L_pre = L_MLM + lambda_t L_time + lambda_l L_len
```

### 3. HPTF Encoder

Core modules:

- Traffic Token Embedding
- Contextual Traffic Transformer Encoder
- Gated Side-channel Feature Fusion
- Intra-packet Field-axis Transformer Encoder
- Inter-packet Temporal Transformer Encoder
- Multi-branch Representation Aggregation

### 4. Supervised Fine-tuning

The learned flow-level representation is fed into a classification head and optimized with cross-entropy loss for downstream encrypted traffic classification.

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

## Fine-tuning

```bash
bash scripts/finetune_tor.sh
bash scripts/finetune_all.sh
```

## Evaluation

```bash
bash scripts/evaluate.sh
```

The evaluation reports Accuracy, Precision, Recall, and Macro-F1.

## Ablation Study

The ablation settings are designed to analyze the contribution of each major component in HPTF.

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
