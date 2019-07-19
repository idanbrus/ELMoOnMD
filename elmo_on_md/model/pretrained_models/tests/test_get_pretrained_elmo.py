from unittest import TestCase
from elmo_on_md.model.pretrained_models.many_lngs_elmo import get_pretrained_elmo
import torch.nn as nn

class TestGet_pretrained_elmo(TestCase):
    def test_get_pretrained_elmo(self):
        embedder = get_pretrained_elmo()
        self.assertIsInstance(embedder.model, nn.Module)

        corpus = [['שלום', 'עולם']]
        embedding = embedder.sents2elmo(corpus)
        self.assertTupleEqual(embedding[0].shape, (2,1024))
