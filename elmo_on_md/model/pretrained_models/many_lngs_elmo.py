from ELMoForManyLangs.elmoformanylangs.elmo import Embedder
import torch.nn as nn
import os


def get_pretrained_elmo() -> Embedder:
    """
    Create a pretrained ELMo embedder
    Returns: an ELMo Embedder object.

    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    embedder = Embedder(os.path.join(root_dir, 'ELMoForManyLangs', 'hebrew'))
    model = embedder.model

    # open the entire model to fine tuning
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    return embedder
