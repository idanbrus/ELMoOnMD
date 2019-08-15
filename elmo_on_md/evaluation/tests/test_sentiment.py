from unittest import TestCase

from elmo_on_md.data_loaders.sentiment_loader import SentimentLoader
from elmo_on_md.evaluation.model_loader import load_model
from elmo_on_md.evaluation.sentiment_analysis import SentimentAnalysis


class TestSentiment(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        elmo = load_model('original') # go back to original
        cls.sentiment_model = SentimentAnalysis(elmo)
        sentiment_data = SentimentLoader().load_data()
        train = sentiment_data['train']
        train_subset = dict()
        train_subset['sentences'] = train['sentences'][:10]
        train_subset['labels'] = train['labels'][:10]
        test_subset = dict()
        test_subset['sentences'] = train['sentences'][10:20]
        test_subset['labels'] = train['labels'][10:20]
        cls.train_subset = train_subset
        cls.test_subset = test_subset
    def test_train_predict(self):
        self.sentiment_model.train(self.train_subset,self.train_subset, n_epochs=2)
        y_pred = self.sentiment_model.predict(self.test_subset)
        self.assertEqual(len(y_pred),10)
