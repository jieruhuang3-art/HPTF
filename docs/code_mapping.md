# Code Mapping

This document maps the major HPTF modules described in the paper to their implementation locations in this repository.

Some modular files provide lightweight interfaces to the main HPTF implementation.

| Paper Module | Implementation | Main Class / Function | Description |
|---|---|---|---|
| Traffic Token Embedding | `hptf/models/embeddings.py` | `str2embedding` | Builds traffic token embeddings used by the HPTF encoder. |
| Contextual Traffic Transformer Encoder | `hptf/models/hptf_encoder.py` | `FlowSemMAEClassifier.encoder` | Models contextual dependencies among traffic tokens. |
| Gated Side-channel Feature Fusion | `hptf/models/hptf_encoder.py` | `FlowSemMAEClassifier._fuse_aux_features` | Fuses inter-arrival time and packet length features with token embeddings. |
| Intra-packet Field-axis Transformer Encoder | `hptf/models/hptf_encoder.py` | `FlowSemMAEClassifier.field_encoder` | Encodes local token structures inside each packet. |
| Inter-packet Temporal Transformer Encoder | `hptf/models/hptf_encoder.py` | `FlowSemMAEClassifier.temporal_encoder` | Encodes temporal dependencies across packet representations. |
| Multi-branch Representation Aggregation | `hptf/models/hptf_encoder.py` | `FlowSemMAEClassifier.forward` | Aggregates token, packet, side-channel, and contextual representations into a flow-level representation. |
| Classification Head | `hptf/models/classification_head.py` | `output_layer_1 / output_layer_2` | Maps the flow-level representation to traffic classes. |
| Multi-objective Pre-training | `hptf/training/pretrain.py` | `mlm_head / time_head / length_head` | Implements masked token, inter-arrival time, and packet length prediction. |
| Supervised Fine-tuning | `hptf/training/finetune.py` | `classification_loss` | Optimizes the classification objective for encrypted traffic classification. |
| Evaluation | `hptf/evaluation/evaluate.py` | `compute_metrics` | Reports standard classification metrics. |
