"""Gated side-channel feature fusion mapping.

The real fusion computation is the private method
``FlowSemMAEClassifier._fuse_aux_features`` copied from the source project.
This module exposes a thin helper without changing the model forward path.
"""
from hptf.models.hptf_encoder import FlowSemMAEClassifier


def apply_gated_side_channel_fusion(model, emb, seg, time_delta, pkt_len):
    """Call the copied fusion implementation on a FlowSemMAEClassifier instance."""
    if not isinstance(model, FlowSemMAEClassifier):
        raise TypeError("model must be a FlowSemMAEClassifier instance")
    return model._fuse_aux_features(emb, seg, time_delta, pkt_len)


__all__ = ["FlowSemMAEClassifier", "apply_gated_side_channel_fusion"]
