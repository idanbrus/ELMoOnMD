from unittest import TestCase

from elmo_on_md.data_loaders.tree_bank_loader import DependencyTreesLoader


class TestDependencyTreesLoader(TestCase):
    def test_load_data(self):
        data = DependencyTreesLoader().load_data()
