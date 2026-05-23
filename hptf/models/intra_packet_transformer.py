"""Intra-packet Field-axis Transformer mapping.

The real field-axis Transformer is ``FlowSemMAEClassifier.field_encoder``.
This helper returns that module from an initialized classifier instance.
"""
from hptf.models.hptf_encoder import FlowSemMAEClassifier


def get_intra_packet_field_axis_transformer(model):
    if not isinstance(model, FlowSemMAEClassifier):
        raise TypeError("model must be a FlowSemMAEClassifier instance")
    return model.field_encoder


__all__ = ["FlowSemMAEClassifier", "get_intra_packet_field_axis_transformer"]
