# Reproduction

## 1. Environment Installation

```bash
conda create -n hptf python=3.9 -y
conda activate hptf
pip install -r requirements.txt
```

## 2. Data Download

Datasets are not included in this repository. Download the public datasets according to their original licenses and citation requirements. Do not commit raw PCAP files, processed TSV files, split files, or private dataset artifacts.

Supported dataset names used in the paper include CSTNET-TLS 1.3, TOR, CESNET-TLS22 W-2021-40, CESNET-TLS22 W-2021-41, ISCX-NONVPN, ISCX-VPN, CrossPlatform Android, and CrossPlatform IOS.

## 3. Data Directory Structure

```text
data/
  raw/
  processed/
  splits/
```

`data/raw/` stores local raw datasets. `data/processed/` stores generated TSV files. `data/splits/` stores optional split metadata. Only `.gitkeep` files should be committed in these directories.

## 4. Preprocessing

```bash
bash scripts/preprocess.sh
```

The migrated preprocessing entry point converts labeled PCAP folders into TSV files with token, inter-arrival time, packet length, and packet-boundary fields. See `docs/data_format.md`.

## 5. Pre-training

```bash
bash scripts/pretrain.sh
```

The migrated pre-training code implements Masked Token Prediction, Inter-arrival Time Prediction, and Packet Length Prediction. The objective corresponds to `L_pre = L_MLM + lambda_t L_time + lambda_l L_len` when the two auxiliary weights share `--aux_loss_weight`.

## 6. Supervised Fine-tuning

```bash
bash scripts/finetune_tor.sh
bash scripts/finetune_all.sh
```

Fine-tuning uses the real copied classifier implementation. The main HPTF model is `FlowSemMAEClassifier`, and pre-trained checkpoints are loaded through `--pretrained_model_path`.

## 7. Evaluation

```bash
bash scripts/evaluate.sh
```

The evaluation entry point exports Accuracy, Precision, Recall, Macro-F1, and Confusion Matrix to JSON.

## 8. Result Storage

Store generated metrics under `results/`. Keep large prediction arrays, checkpoints, logs, model weights, and dataset files out of Git. Result tables in papers must be filled only from real result files; missing values must remain `TBD`.

## 9. Ablation Experiments

```bash
bash scripts/run_ablation_tor.sh
```

This is a template. Some paper ablations do not have direct CLI switches in the migrated codebase. Consult `docs/code_mapping.md` and `migration_report.md` before implementing or running an ablation.

## 10. Common Issues

- Missing dataset files: place local data under `data/raw/` or generated TSV files under `data/processed/`.
- Missing vocabulary: set `data.vocab_path` in the YAML config to the local vocabulary file.
- Missing checkpoint: pre-train first or remove the YAML `model.pretrained_checkpoint` value for from-scratch fine-tuning.
- Unsupported ablation switch: do not invent `--ablation`; extend the code only after identifying the module in `docs/code_mapping.md`.
- Different metrics from the table: verify dataset version, split seed, preprocessing options, and checkpoint path. Do not edit results to match the table.
