from typing import List

import sklearn_crfsuite
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam

from ELMoForManyLangs.elmoformanylangs import Embedder
from allennlp.modules.conditional_random_field import ConditionalRandomField


class NER():
    def __init__(self, elmo: Embedder, n_tags =1):
        self.elmo = elmo
        self.model = BiLSTM(n_tags=n_tags)
        self.criterion = nn.BCELoss()  # Binary cross entropy
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def train(self, train_set: List[pd.DataFrame],
              n_epochs: int = 10,
              batch_size: int = 64):
        # create input for the model
        batch_generator = self._chunker_list(train_set, batch_size)

        for epoch in range(n_epochs):
            for batch_set in batch_generator:
                self.optimizer.zero_grad()

                X, max_sentence_length = self._create_input(batch_set)
                output = self.model(X)

                y = self._create_labels(batch_set, max_sentence_length)
                loss = self.criterion(output, y)

                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, test_set: List[pd.DataFrame]):
        tokens = [sentence['word'] for sentence in test_set]
        X = self.elmo.sents2elmo(tokens)
        y_pred = self.crf.predict(X)
        return y_pred

    def _create_input(self, train_set):
        tokens = [sentence['word'] for sentence in train_set]
        X = self.elmo.sents2elmo(tokens)
        max_sentence_length = max([sentence.shape[0] for sentence in X])
        input = torch.zeros(len(X), max_sentence_length, X[0].shape[1])
        for i, sentence in enumerate(X):
            input[i, :sentence.shape[0], :] = torch.from_numpy(sentence)
        return input, max_sentence_length

    def _create_labels(self, train_set, max_sentence_length):
        labels = torch.zeros(len(train_set), max_sentence_length)
        for i, sentence in enumerate(train_set):
            labels[i, :len(sentence)] = sentence['label']
        return labels

    def _chunker_list(self, l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim=1024,
                 hidden_dim=256,
                 n_tags=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.hidden2label = nn.Linear(hidden_dim * 2, n_tags)

    def forward(self, *input):
        output, (hn, cn) = self.lstm(input)
        output = nn.Sigmoid(self.hidden2label(output))
        return output
