# Reproduction Guide

This guide describes how to reproduce the main training and evaluation pipeline of HPTF.

## 1. Environment Setup

```bash
conda create -n hptf python=3.9 -y
conda activate hptf
pip install -r requirements.txt
```

## 2. Data Preparation

Prepare PCAP or TSV files and convert them into packet-aware traffic token sequences.

```bash
bash scripts/preprocess.sh
```

The expected processed samples contain traffic tokens, packet-boundary indicators, inter-arrival time features, packet length features, and class labels.

## 3. Pre-training

```bash
bash scripts/pretrain.sh
```

The pre-training stage optimizes masked token prediction, inter-arrival time prediction, and packet length prediction.

## 4. Fine-tuning

```bash
bash scripts/finetune_tor.sh
bash scripts/finetune_all.sh
```

The fine-tuning stage trains HPTF for supervised encrypted traffic classification.

## 5. Evaluation

```bash
bash scripts/evaluate.sh
```

The evaluation stage reports Accuracy, Precision, Recall, and Macro-F1.

## 6. Ablation Study

```bash
bash scripts/run_ablation_tor.sh
```

The ablation study evaluates the contribution of major HPTF components.
