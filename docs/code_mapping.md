# Code Mapping

This file maps paper modules to migrated source code. It records only modules found in the current project; missing standalone implementations are noted explicitly.

| Paper Module | Code Location | Class / Function | Implementation Note |
|---|---|---|---|
| Traffic Token Embedding | `hptf/models/embeddings.py`; `hptf/uer/layers/embeddings.py` | `str2embedding[...]`; copied UER embedding classes | Used by `FlowSemMAEClassifier.embedding`; copied from the source ET-BERT/UER dependency. |
| Contextual Traffic Transformer Encoder | `hptf/models/hptf_encoder.py`; `hptf/uer/encoders/transformer_encoder.py` | `FlowSemMAEClassifier.encoder` | Enabled by `--flowsem_use_context_encoder`; constructed through `str2encoder[args.encoder]`; no standalone `HPTFEncoder` source class was found. |
| Gated Side-channel Feature Fusion | `hptf/models/hptf_encoder.py`; `hptf/models/gated_fusion.py` | `FlowSemMAEClassifier._fuse_aux_features` | Uses `time_feature_proj`, `length_feature_proj`, `aux_feature_norm`, and `aux_gate` when `--use_aux_features --aux_feature_fusion gate` are active. |
| Intra-packet Field-axis Transformer Encoder | `hptf/models/hptf_encoder.py`; `hptf/models/intra_packet_transformer.py` | `FlowSemMAEClassifier.field_encoder` | Builds packet-field tables with `_flowsem_table` and applies `nn.TransformerEncoder` over field slots. |
| Inter-packet Temporal Transformer Encoder | `hptf/models/hptf_encoder.py`; `hptf/models/inter_packet_transformer.py` | `FlowSemMAEClassifier.temporal_encoder` | Applies `nn.TransformerEncoder` over packet representations. |
| Multi-branch Representation Aggregation | `hptf/models/hptf_encoder.py` | `FlowSemMAEClassifier.forward` | Fuses packet, statistics, optional direct-token, and optional CLS residual branches through `flow_stats_proj`, `direct_token_attention`, `cls_fusion`, `branch_gate`, and `branch_fusion_proj`. |
| Classification Head | `hptf/models/hptf_encoder.py`; `hptf/models/classification_head.py` | `output_layer_1 / output_layer_2` | Two linear layers with `tanh`, then class logits; loss comes from `classification_loss`. |
| Multi-objective Pre-training | `hptf/training/pretrain.py` | `mlm_head / time_head / length_head` | Implements Masked Token Prediction, Inter-arrival Time Prediction, and Packet Length Prediction. |
| Supervised Fine-tuning | `hptf/training/finetune.py` | `F.cross_entropy` | `classification_loss` uses `F.cross_entropy`; `load_or_initialize_parameters` loads `--pretrained_model_path`. |

## Ablation Switch Support

| Paper Ablation | Direct Switch Found | Notes |
|---|---|---|
| w/o Intra-packet Field-axis Transformer Encoder | Yes: `--flowsem_no_field_axis_transformer` | Bypasses field-axis Transformer only. |
| w/o Inter-packet Temporal Transformer Encoder | Yes: `--flowsem_no_temporal_transformer` | Bypasses temporal-axis Transformer only. |
| w/o Gated Side-channel Feature Fusion | No direct `--ablation` switch | Source supports changing `--aux_feature_fusion`, but no exact remove-gate ablation switch was found. |
| w/o Time-Length Side-channel Features | No direct remove switch | Source enables features with `--use_aux_features`; omission disables them at launch. |
| w/o Multi-objective Masked Pre-training | Partial | Pretraining has `--disable_mlm_loss`; fine-tuning without pretraining is done by omitting `--pretrained_model_path`. |
| w/o Contextual Encoder and Residual Path | No single direct switch | Context encoder and CLS residual are opt-in flags: `--flowsem_use_context_encoder`, `--flowsem_cls_residual`. |
