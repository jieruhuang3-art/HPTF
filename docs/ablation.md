# Ablation Study

This document explains the TOR ablation items used in the paper and the support status in the migrated codebase.

## TOR Ablation Items

1. w/o Intra-packet Field-axis Transformer Encoder
   - Paper meaning: verifies whether modeling local field/token arrangements inside each packet improves flow representation.
   - Direct source switch found: `--flowsem_no_field_axis_transformer`.

2. w/o Inter-packet Temporal Transformer Encoder
   - Paper meaning: verifies whether packet-to-packet temporal dependency modeling improves classification.
   - Direct source switch found: `--flowsem_no_temporal_transformer`.

3. w/o Gated Side-channel Feature Fusion
   - Paper meaning: verifies whether the gate helps adaptive fusion of inter-arrival time and packet length side-channel features.
   - No exact direct source switch named `remove_gated_fusion` was found. The source supports `--aux_feature_fusion {add,gate,cross_attention}`.

4. w/o Time-Length Side-channel Features
   - Paper meaning: verifies whether inter-arrival time and packet length information provide useful auxiliary transport behavior.
   - No exact direct source switch named `remove_time_length_features` was found. Auxiliary features are enabled by `--use_aux_features`.

5. w/o Multi-objective Masked Pre-training
   - Paper meaning: verifies whether joint token-context and transport-behavior pre-training improves downstream fine-tuning.
   - Partial source support exists: pre-training has `--disable_mlm_loss`; fine-tuning without pretraining can be represented by omitting `--pretrained_model_path`.

6. w/o Contextual Encoder and Residual Path
   - Paper meaning: verifies whether global contextual encoding and residual CLS fusion stabilize flow-level representation.
   - No single direct source switch was found. Contextual encoder and CLS residual are opt-in flags: `--flowsem_use_context_encoder` and `--flowsem_cls_residual`.

## Important Note

The current repository does not fabricate a nonexistent `--ablation` parameter. If an ablation does not have a direct code switch, researchers should identify the corresponding module in `docs/code_mapping.md` and manually implement or extend the code in a transparent, documented way.
