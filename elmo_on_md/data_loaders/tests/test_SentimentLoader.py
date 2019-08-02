from unittest import TestCase
import pandas as pd

from elmo_on_md.data_loaders.sentiment_loader import SentimentLoader

class TestSentimentLoader(TestCase):
    def test_load_data(self):
        data = SentimentLoader().load_data()
        self.assertGreater(len(data['train']['sentences']), 0)
        self.assertEqual(len(data['train']['labels']), len(data['train']['sentences']))

    def test__read_sentence(self):
        sentence ='שלום עולם!\t1'
        loader = SentimentLoader()
        (tokens,label)=loader._read_sentence(sentence)
        self.assertEqual(len(tokens),2)
        self.assertEqual(label,1)

