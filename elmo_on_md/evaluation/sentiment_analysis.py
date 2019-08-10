import torch
import torch.nn as nn
from ELMoForManyLangs.elmoformanylangs import Embedder
from typing import List,Dict
from torch.optim import Adam
import numpy as np


class MyRNN(nn.Module):
    def __init__(self, embedding_dim=1024,
                 hidden_dim=256,
                 n_tags=1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, n_tags)
        self.softmax = nn.Softmax()

    def forward(self, input):
        output, hidden = self.rnn(input)
        output = self.hidden2label(hidden.squeeze()).squeeze()
        output = self.softmax(output)
        return output

    def initHidden(self, input_size):
        return torch.zeros((1, input_size, self.hidden_dim))


class SentimentAnalysis():
    def __init__(self, elmo: Embedder, n_tags=3):
        self.elmo = elmo
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MyRNN(n_tags=n_tags)
        #TODO: set weights
        weights = torch.ones(n_tags)
        weights = torch.cat([weights, torch.ones(1)]).to(self.device)
        self.criterion = nn.CrossEntropyLoss()  # Binary cross entropy
        self.optimizer = Adam(self.model.parameters(), lr=1e-5)
        self.max_sentence_length = 100

    def train(self, train_set: Dict,
              n_epochs: int = 10,
              batch_size: int = 64):
        # create input for the model
        for epoch in range(n_epochs):
            batch_generator = self._chunker_list(train_set, batch_size)
            epoch_loss = 0.0
            for batch_set in batch_generator:
                self.optimizer.zero_grad()

                X = self._create_input(batch_set)
                output = self.model(X)

                y = self._create_labels(batch_set)
                loss = self.criterion(output, y)

                loss.backward()
                self.optimizer.step()
                epoch_loss += output.shape[1] * loss.item()
            print('Loss:', epoch_loss)
        return self

    def predict(self, test_set: Dict):
        X = self._create_input(test_set)
        y_pred = self.model(X)
        return np.argmax(y_pred.detach().numpy(), axis=1)

    def _create_input(self, train_set):
        tokens = train_set['sentences']
        #We set the first axis as the sentence length, so that the RNN can go over it
        X = self.elmo.sents2elmo([sentence[:self.max_sentence_length] for sentence in tokens])
        input = torch.zeros(self.max_sentence_length, len(X), X[0].shape[1])
        for i, sentence in enumerate(X):
            input[:sentence.shape[0], i, :] = torch.from_numpy(sentence)

        return input

    def _create_labels(self, train_set):
        return torch.from_numpy(np.array(train_set['labels']).astype('long')).long()

    def _chunker_list(self, train_set, n):
        s = train_set['sentences']
        l = train_set['labels']
        for i in range(0, len(l), n):
            subset = dict()
            subset['sentences'] = s[i:i + n]
            subset['labels'] = l[i:i + n]
            yield subset