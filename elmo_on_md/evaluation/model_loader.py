import os
import pickle

from ELMoForManyLangs.elmoformanylangs import Embedder
from elmo_on_md.model.pretrained_models.many_lngs_elmo import get_pretrained_elmo


def load_model(model_name: str) -> Embedder:
    """
    load an elmo model
    Args:
        model_name: the name of the model to be loaded.
        if the original (pretrained) is desired, the name should be 'original'
    Returns:
        an ELMo Embedder
    """
    if model_name == 'original':
        return get_pretrained_elmo()
    else:
        root_dir = os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
        with open(os.path.join(root_dir, 'elmo_on_md', 'model', 'trained_models', f'{model_name}.pkl'), 'rb') as f:
            model = pickle.load(f)
        return model
