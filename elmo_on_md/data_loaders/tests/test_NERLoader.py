from unittest import TestCase
import pandas as pd

from elmo_on_md.data_loaders.ner_loader import NERLoader


class TestNERLoader(TestCase):
    def test_load_data(self):
        data = NERLoader().load_data()
        self.assertIsInstance(data[0], pd.DataFrame)
