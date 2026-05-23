# Ablation Study

The ablation study is designed to analyze the contribution of the main HPTF components.

## Intra-packet Field-axis Transformer Encoder

This ablation verifies the role of intra-packet local structure modeling. It studies whether token organization within the same packet contributes to flow-level representation learning.

## Inter-packet Temporal Transformer Encoder

This ablation verifies the role of inter-packet temporal dependency modeling. It studies whether packet-level sequential behavior improves encrypted traffic classification.

## Gated Side-channel Feature Fusion

This ablation verifies the effect of gated fusion for time-length side-channel features. It evaluates whether adaptive feature injection improves the use of inter-arrival time and packet length information.

## Time-Length Side-channel Features

This ablation verifies the auxiliary contribution of inter-arrival time and packet length features. These transport-level cues can remain informative when payload content is encrypted.

## Multi-objective Pre-training

This ablation verifies the effect of pre-training on token context and transport behavior prior learning.

## Contextual Encoder and Residual Path

This ablation verifies the contribution of global contextual modeling and residual representation paths to stable flow-level representation learning.
