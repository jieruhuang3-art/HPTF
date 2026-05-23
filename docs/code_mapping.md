# Code Mapping

This document maps the major HPTF modules described in the paper to their implementation locations in this repository.

| Component | Location | Description |
|---|---|---|
| Traffic Token Embedding | `hptf/models/embeddings.py` | Builds dense representations for traffic tokens. |
| Contextual Traffic Transformer Encoder | `hptf/models/hptf_encoder.py` | Models global dependencies among traffic tokens. |
| Gated Side-channel Feature Fusion | `hptf/models/gated_fusion.py` | Incorporates inter-arrival time and packet length features. |
| Intra-packet Field-axis Transformer Encoder | `hptf/models/intra_packet_transformer.py` | Encodes local token structures inside packets. |
| Inter-packet Temporal Transformer Encoder | `hptf/models/inter_packet_transformer.py` | Encodes temporal dependencies across packet representations. |
| Multi-branch Representation Aggregation | `hptf/models/hptf_encoder.py` | Builds the flow-level representation from multiple information branches. |
| Classification Head | `hptf/models/classification_head.py` | Maps flow-level representations to traffic classes. |
| Multi-objective Pre-training | `hptf/training/pretrain.py` | Optimizes masked token, inter-arrival time, and packet length prediction. |
| Supervised Fine-tuning | `hptf/training/finetune.py` | Trains HPTF for labeled encrypted traffic classification. |
| Evaluation | `hptf/evaluation/evaluate.py` | Reports standard classification metrics. |
