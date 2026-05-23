"""Inter-packet Temporal Transformer mapping.

The real temporal Transformer is ``FlowSemMAEClassifier.temporal_encoder``.
This helper returns that module from an initialized classifier instance.
"""
from hptf.models.hptf_encoder import FlowSemMAEClassifier


def get_inter_packet_temporal_transformer(model):
    if not isinstance(model, FlowSemMAEClassifier):
        raise TypeError("model must be a FlowSemMAEClassifier instance")
    return model.temporal_encoder


__all__ = ["FlowSemMAEClassifier", "get_inter_packet_temporal_transformer"]
