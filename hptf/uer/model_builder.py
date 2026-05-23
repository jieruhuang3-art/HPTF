import torch
from hptf.uer.layers import *
from hptf.uer.encoders import *
from hptf.uer.targets import *
from hptf.uer.models.model import Model


def build_model(args):
    """
    Build universial encoder representations models.
    The combinations of different embedding, encoder, 
    and target layers yield pretrained models of different 
    properties. 
    We could select suitable one for downstream tasks.
    """

    embedding = str2embedding[args.embedding](args, len(args.vocab))
    encoder = str2encoder[args.encoder](args)
    target = str2target[args.target](args, len(args.vocab))
    model = Model(args, embedding, encoder, target)

    return model
