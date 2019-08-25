from ELMoForManyLangs.elmoformanylangs.elmo import Embedder
import torch.nn as nn
import os


def get_pretrained_elmo(batch_size:int = 64) -> Embedder:
    """
    Create a pretrained ELMo embedder
    Returns: an ELMo Embedder object.

    """
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    embedder = Embedder(os.path.join(root_dir, 'ELMoForManyLangs', 'hebrew'), batch_size=batch_size)
    model = embedder.model

    # open the entire model to fine tuning
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    return embedder
