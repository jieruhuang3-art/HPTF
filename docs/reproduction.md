# Reproduction Guide

This guide describes the main training and evaluation pipeline of HPTF.

## 1. Environment Setup

```bash
conda create -n hptf python=3.9 -y
conda activate hptf
pip install -r requirements.txt
```

## 2. Data Preparation

Prepare encrypted traffic datasets in PCAP or TSV format and convert them into packet-aware traffic token sequences.

```bash
bash scripts/preprocess.sh
```

## 3. Pre-training

Run self-supervised pre-training with masked token prediction, inter-arrival time prediction, and packet length prediction.

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

Evaluate a fine-tuned model with Accuracy, Precision, Recall, and Macro-F1.

```bash
bash scripts/evaluate.sh
```

## 6. Ablation Study

Run component-level ablation experiments to analyze the contribution of HPTF modules.

```bash
bash scripts/run_ablation_tor.sh
```
