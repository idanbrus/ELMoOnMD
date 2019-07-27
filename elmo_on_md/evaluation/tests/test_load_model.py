from unittest import TestCase

from ELMoForManyLangs.elmoformanylangs import Embedder
from elmo_on_md.evaluation.model_loader import load_model


class TestLoad_model(TestCase):
    def test_load_model(self):
        embedder = load_model('new_model')
        self.assertIsInstance(embedder, Embedder)

