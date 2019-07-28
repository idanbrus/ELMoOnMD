from unittest import TestCase

from sklearn.model_selection import train_test_split

from elmo_on_md.data_loaders.ner_loader import NERLoader
from elmo_on_md.evaluation.model_loader import load_model
from elmo_on_md.evaluation.named_entitiy_recognition import NER


class TestNER(TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        elmo = load_model('original')
        cls.ner_model = NER(elmo)
        ner_data = NERLoader().load_data()[:10]
        cls.train_set, cls.test_set = train_test_split(ner_data, test_size=0.2)

    def test_train_predict(self):
        self.ner_model.train(self.train_set, n_epochs=2)
        y_pred = self.ner_model.predict(self.test_set)
        y_pred
