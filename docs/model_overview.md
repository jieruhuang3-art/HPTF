# HPTF Model Overview

## Motivation

Encrypted traffic classification requires representations that preserve packet-level structure while remaining effective under payload encryption.

1. Flat traffic token sequences cannot explicitly distinguish intra-packet local structure and inter-packet temporal dependency.
2. Encrypted traffic still exposes observable transport-side features such as inter-arrival time and packet length.
3. A hierarchical representation is needed to integrate token context, packet structure, temporal behavior, and auxiliary side-channel features.

## Architecture

HPTF represents each encrypted flow as a packet-aware token sequence and builds a flow-level representation through hierarchical packet-temporal modeling.

The encoder consists of Traffic Token Embedding, Contextual Traffic Transformer Encoder, Gated Side-channel Feature Fusion, Intra-packet Field-axis Transformer Encoder, Inter-packet Temporal Transformer Encoder, and Multi-branch Representation Aggregation. Together, these modules capture token semantics, packet-local structure, packet-level temporal behavior, and time-length transport cues.

## Pre-training Objectives

HPTF uses multi-objective self-supervised pre-training to learn contextual token representations and transport behavior priors:

1. Masked Token Prediction
2. Inter-arrival Time Prediction
3. Packet Length Prediction

## Fine-tuning

For downstream encrypted traffic classification, HPTF feeds the learned flow-level representation into a classification head and optimizes the model with cross-entropy loss.
