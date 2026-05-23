"""Traffic embedding exports used by the migrated HPTF implementation.

The canonical embedding modules are copied from the original UER/ET-BERT
dependency. FlowSemMAEClassifier instantiates them through ``str2embedding``.
"""
from hptf.uer.layers.embeddings import *  # noqa: F401,F403
from hptf.uer.layers.embeddings import str2embedding

__all__ = ["str2embedding"]
