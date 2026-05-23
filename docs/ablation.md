# Ablation Study

The ablation study analyzes the contribution of the main HPTF components.

## Intra-packet Field-axis Transformer Encoder

This setting evaluates the effect of modeling local token structures inside each packet.

## Inter-packet Temporal Transformer Encoder

This setting evaluates the effect of modeling temporal dependencies across packet representations.

## Gated Side-channel Feature Fusion

This setting evaluates the effect of adaptive fusion for inter-arrival time and packet length features.

## Time-Length Side-channel Features

This setting evaluates the auxiliary contribution of transport-level time and length information.

## Multi-objective Pre-training

This setting evaluates the benefit of learning contextual token representations and transport behavior priors before supervised fine-tuning.

## Contextual Encoder and Residual Path

This setting evaluates the role of global contextual modeling and residual representation paths in stable flow-level representation learning.
