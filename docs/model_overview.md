# HPTF Model Overview

## Motivation

Encrypted traffic classification requires representations that preserve packet structure while remaining effective under payload encryption.

1. Flat traffic token sequences cannot explicitly distinguish intra-packet local structure and inter-packet temporal dependency.
2. Encrypted traffic still exposes observable transport-side features such as inter-arrival time and packet length.
3. A hierarchical representation is needed to integrate token context, packet structure, temporal behavior, and auxiliary side-channel features.

## Architecture

HPTF represents each flow as a packet-aware token sequence and constructs a flow-level representation through hierarchical packet-temporal modeling.

Traffic Token Embedding maps traffic tokens into dense representations. Contextual Traffic Transformer Encoder captures global token dependencies. Gated Side-channel Feature Fusion injects inter-arrival time and packet length features. Intra-packet Field-axis Transformer Encoder models local structures within packets, while Inter-packet Temporal Transformer Encoder models temporal dependencies across packets. Multi-branch Representation Aggregation combines contextual, packet-level, temporal, and side-channel representations.

## Pre-training Objectives

HPTF supports multi-objective self-supervised pre-training:

1. Masked Token Prediction
2. Inter-arrival Time Prediction
3. Packet Length Prediction

These objectives encourage HPTF to learn contextual token semantics and transport behavior priors before supervised fine-tuning.

## Fine-tuning

For downstream encrypted traffic classification, HPTF feeds the learned flow-level representation into a classification head and optimizes the model using cross-entropy loss.
