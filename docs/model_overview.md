# HPTF Model Overview

## Motivation

1. Flat traffic token sequences cannot explicitly distinguish intra-packet local structure and inter-packet temporal dependency.
2. Encrypted traffic still exposes observable transport-side features such as inter-arrival time and packet length.
3. A hierarchical representation is needed to integrate token context, packet structure, temporal behavior, and auxiliary side-channel features.

## Architecture

HPTF organizes encrypted traffic into packet-aware token sequences and builds a flow-level representation through the following components:

- Traffic Token Embedding
- Contextual Traffic Transformer Encoder
- Gated Side-channel Feature Fusion
- Intra-packet Field-axis Transformer Encoder
- Inter-packet Temporal Transformer Encoder
- Multi-branch Representation Aggregation

The contextual encoder captures global token dependencies. The gated fusion module injects time-length side-channel features. The field-axis encoder models local token structures within packets, while the temporal encoder captures dependencies across packet representations. The final aggregation stage combines token-level, packet-level, side-channel, and contextual representations into a flow-level representation.

## Pre-training Objectives

HPTF supports multi-objective self-supervised pre-training:

- Masked Token Prediction
- Inter-arrival Time Prediction
- Packet Length Prediction

These objectives encourage the model to learn both token context and transport behavior patterns before supervised classification.

## Fine-tuning

For downstream encrypted traffic classification, HPTF uses the learned flow-level representation with a classification head and cross-entropy optimization.
