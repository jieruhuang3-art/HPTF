# Reproduction Guide

This guide describes the main training and evaluation pipeline of HPTF.

## 1. Environment Setup

```bash
conda create -n hptf python=3.9 -y
conda activate hptf
pip install -r requirements.txt
```

## 2. Data Preparation

Prepare PCAP or TSV files and convert them into packet-aware traffic token sequences with packet boundary, inter-arrival time, and packet length features.

```bash
bash scripts/preprocess.sh
```

## 3. Pre-training

Run self-supervised pre-training with masked token prediction and time-length behavior prediction.

```bash
bash scripts/pretrain.sh
```

## 4. Fine-tuning

Fine-tune HPTF on labeled encrypted traffic classification datasets.

```bash
bash scripts/finetune_tor.sh
bash scripts/finetune_all.sh
```

## 5. Evaluation

Evaluate the fine-tuned model with standard classification metrics.

```bash
bash scripts/evaluate.sh
```

## 6. Ablation Study

Run component-level ablation experiments for HPTF.

```bash
bash scripts/run_ablation_tor.sh
```
