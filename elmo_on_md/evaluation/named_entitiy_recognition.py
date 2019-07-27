from typing import List

import sklearn_crfsuite
import pandas as pd
from ELMoForManyLangs.elmoformanylangs import Embedder


class NER():
    def __init__(self, elmo: Embedder):
        self.elmo = elmo
        self.crf = sklearn_crfsuite.CRF()

    def train(self, train_set: List[pd.DataFrame]):
        tokens = [train_set[i]['word'] for i in range(len(train_set))]
        X = self.elmo.sents2elmo(tokens)
        y = [train_set[i]['name_entity'] for i in range(len(train_set))]

        self.crf.fit(X,y)
        return self

    def predict(self, test_set: List[pd.DataFrame]):
        tokens = [test_set[i]['word'] for i in range(len(test_set))]
        X = self.elmo.sents2elmo(tokens)
        y_pred = self.crf.predict(X)
        return y_pred