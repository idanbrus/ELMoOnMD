from unittest import TestCase

from elmo_on_md.data_loaders.tree_bank_loader import Token_loader


class TestToken_loader(TestCase):
    def test_load_data(self):
        token_loader = Token_loader()
        corpus = token_loader.load_data()
        self.assertGreater(len(corpus['train']), 0)
