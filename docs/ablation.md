# Ablation Study

The ablation study is designed to analyze the contribution of the main HPTF components.

## Intra-packet Field-axis Transformer Encoder

This ablation evaluates the role of local structure modeling within each packet. Removing this component tests whether packet-internal token organization contributes to discriminative flow representations.

## Inter-packet Temporal Transformer Encoder

This ablation evaluates the role of temporal dependency modeling across packets. Removing this component tests whether packet-level sequence dynamics improve encrypted traffic classification.

## Gated Side-channel Feature Fusion

This ablation evaluates the effect of adaptively fusing time-length transport side-channel features. The gate controls how inter-arrival time and packet length information are injected into token representations.

## Time-Length Side-channel Features

This ablation evaluates the auxiliary contribution of inter-arrival time and packet length. These features provide observable transport-level cues even when payload content is encrypted.

## Multi-objective Pre-training

This ablation evaluates whether self-supervised pre-training helps HPTF learn token context and transport behavior priors before supervised fine-tuning.

## Contextual Encoder and Residual Path

This ablation evaluates the role of global contextual modeling and residual representation paths in producing stable flow-level representations.
