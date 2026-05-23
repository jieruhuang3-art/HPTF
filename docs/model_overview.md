# Model Overview

## Core Problems

HPTF targets three limitations in encrypted traffic representation learning:

1. Flat traffic-token sequences make it difficult to distinguish intra-packet local structure from inter-packet temporal dependency.
2. Transport side-channel features such as inter-arrival time, packet length, and packet boundary markers are often weakly fused or ignored.
3. Cross-dataset and cross-platform traffic classification settings introduce generalization challenges.

## HPTF Solution

The migrated implementation keeps the source project class name `FlowSemMAEClassifier`. This class corresponds to the paper's HPTF encoder and classification model. The class name is preserved because no standalone source class named `HPTFEncoder` was found.

HPTF addresses the problems above through:

- Contextual Traffic Transformer Encoder: learns global traffic-token context through `FlowSemMAEClassifier.encoder` when `--flowsem_use_context_encoder` is enabled.
- Gated Side-channel Feature Fusion: adaptively injects inter-arrival time and packet length features through `FlowSemMAEClassifier._fuse_aux_features`.
- Intra-packet Field-axis Transformer Encoder: models local token arrangement inside packets through `FlowSemMAEClassifier.field_encoder`.
- Inter-packet Temporal Transformer Encoder: models temporal dependency across packet representations through `FlowSemMAEClassifier.temporal_encoder`.
- Multi-branch Representation Aggregation: fuses packet, direct-token, flow-statistics, and optional CLS residual branches in `FlowSemMAEClassifier.forward`.
- Multi-objective Masked Pre-training: jointly trains Masked Token Prediction, Inter-arrival Time Prediction, and Packet Length Prediction in `hptf/training/pretrain.py`.

## Implementation Note

The wrapper files in `hptf/models/` document and expose the relevant pieces, but the canonical forward computation remains inside `FlowSemMAEClassifier.forward`.

Evidence source: `/root/ET-BERT-main/fine-tuning/run_classifier.py`, especially `FlowSemMAEClassifier`, and `/root/ET-BERT-main/pretrain_traffic_mlm_aux.py`.
