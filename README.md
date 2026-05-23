# HPTF: Hierarchical Packet-Temporal Fusion for Encrypted Traffic Classification

HPTF is a hierarchical traffic representation learning framework for encrypted traffic classification. It represents encrypted traffic as packet-aware token sequences and jointly models contextual token dependencies, intra-packet local structures, inter-packet temporal dependencies, and time-length side-channel features.

This repository provides the implementation of HPTF, including data preprocessing, self-supervised pre-training, supervised fine-tuning, evaluation, and ablation study utilities.

## Highlights

- Packet-aware encrypted traffic representation
- Hierarchical modeling of token context, intra-packet structure, and inter-packet temporal dependency
- Gated fusion of inter-arrival time and packet length features
- Unified pre-training and fine-tuning pipeline for encrypted traffic classification

## Framework

### 1. Data Preprocessing

1.1 Flow aggregation  
1.2 Sliding-window packet segmentation  
1.3 Traffic tokenization  
1.4 Packet boundary construction  
1.5 Time-length feature construction  

### 2. Self-supervised Pre-training

2.1 Masked token prediction  
2.2 Inter-arrival time prediction  
2.3 Packet length prediction  

Loss:

```text
L_pre = L_MLM + lambda_t L_time + lambda_l L_len
```

### 3. HPTF Encoder

3.1 Traffic Token Embedding  
3.2 Contextual Traffic Transformer Encoder  
3.3 Gated Side-channel Feature Fusion  
3.4 Intra-packet Field-axis Transformer Encoder  
3.5 Inter-packet Temporal Transformer Encoder  
3.6 Multi-branch Representation Aggregation  

### 4. Supervised Fine-tuning

4.1 Flow-level representation  
4.2 Classification head  
4.3 Cross-entropy optimization  

## Installation

```bash
conda create -n hptf python=3.9 -y
conda activate hptf
pip install -r requirements.txt
```

## Data Preparation

Prepare encrypted traffic datasets in PCAP or TSV format and place them under the `data` directory. The preprocessing pipeline converts traffic flows into packet-aware token sequences with packet boundary, inter-arrival time, and packet length features.

Example datasets include CSTNET-TLS 1.3, TOR, CESNET-TLS22, ISCX-VPN / ISCX-NONVPN, and CrossPlatform Android / IOS.

```bash
bash scripts/preprocess.sh
```

## Pre-training

Run self-supervised pre-training before downstream fine-tuning:

```bash
bash scripts/pretrain.sh
```

## Fine-tuning

Fine-tune HPTF on labeled encrypted traffic classification datasets:

```bash
bash scripts/finetune_tor.sh
bash scripts/finetune_all.sh
```

## Evaluation

Evaluate a fine-tuned checkpoint with standard classification metrics:

```bash
bash scripts/evaluate.sh
```

Reported metrics include Accuracy, Precision, Recall, and Macro-F1.

## Ablation Study

The ablation study analyzes the contribution of the main HPTF components:

- w/o Intra-packet Field-axis Transformer Encoder
- w/o Inter-packet Temporal Transformer Encoder
- w/o Gated Side-channel Feature Fusion
- w/o Time-Length Side-channel Features
- w/o Multi-objective Pre-training
- w/o Contextual Encoder and Residual Path

```bash
bash scripts/run_ablation_tor.sh
```

## Repository Layout

- `hptf/`: HPTF package for modeling, data processing, training, and evaluation
- `configs/`: experiment configuration files
- `scripts/`: runnable command-line scripts
- `docs/`: documentation for the model, data format, reproduction, and ablation study
- `data/`: workspace for traffic datasets and processed samples
- `results/`: workspace for evaluation outputs

## Results

HPTF achieves competitive performance across multiple encrypted traffic classification benchmarks. The results show that hierarchical packet-temporal modeling and gated time-length feature fusion improve encrypted traffic representation for TLS service classification, anonymous traffic classification, VPN/non-VPN traffic classification, and mobile application traffic classification tasks.

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
