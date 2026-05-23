"""Classification-head mapping for the migrated HPTF implementation.

The real classification head is kept inside the copied classifier classes as
``output_layer_1`` and ``output_layer_2``. This module exposes transparent
helpers for documentation and inspection.
"""
from hptf.models.hptf_encoder import (
    Classifier,
    FlowSemMAEClassifier,
    HybridPacketFlowTransformerClassifier,
    PacketContextFlowTransformerClassifier,
    PacketFlowTransformerClassifier,
    classification_loss,
)


def get_classification_head(model):
    return model.output_layer_1, model.output_layer_2


__all__ = [
    "Classifier",
    "FlowSemMAEClassifier",
    "HybridPacketFlowTransformerClassifier",
    "PacketContextFlowTransformerClassifier",
    "PacketFlowTransformerClassifier",
    "classification_loss",
    "get_classification_head",
]
