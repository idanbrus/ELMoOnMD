from elmoformanylangs import Embedder
import torch.nn as nn
import os


def get_pretrained_elmo() -> Embedder:
    """
    Create a pretrained ELMo embedder
    Returns: an ELMo Embedder object.

    """
    embedder = Embedder('..\..\..\..\ELMoForManyLangs\hebrew')
    model = embedder.model

    # open the entire model to fine tuning
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    return embedder
