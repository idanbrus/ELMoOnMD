from typing import List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from ELMoForManyLangs.elmoformanylangs import Embedder


class NER():
    def __init__(self, elmo: Embedder, n_tags:int =8, pos_weight:float = 1):
        self.elmo = elmo
        self.n_tags = n_tags
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # set up the model
        self.model = BiLSTM(n_tags=n_tags + 1).to(self.device)
        weights = torch.ones(n_tags) * pos_weight
        weights = torch.cat([weights, torch.ones(1)]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)  # Binary cross entropy
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def train(self, train_set: List[pd.DataFrame],
              tags_columns:List['str'],
              n_epochs: int = 10,
              batch_size: int = 64):
        # create input for the model
        batch_generator = self._chunker_list(train_set, batch_size)

        for epoch in tqdm(range(n_epochs)):
            for batch_set in batch_generator:
                self.optimizer.zero_grad()

                X, max_sentence_length = self._create_input(batch_set)
                output = self.model(X)
                output = output.view(output.shape[0] * output.shape[1], -1) # flatten the results

                y = self._create_labels(batch_set, max_sentence_length, tags_columns)
                loss = self.criterion(output, y)

                loss.backward()
                self.optimizer.step()

        return self

    def predict(self, test_set: List[pd.DataFrame]) -> torch.tensor:
        with torch.no_grad():

            X, max_sentence_length = self._create_input(test_set)
            y_pred = self.model(X)
        y_pred = y_pred.argmax(dim=-1)
        return y_pred.to('cpu')

    def _create_input(self, train_set)-> Tuple[torch.tensor, int]:
        tokens = [sentence['word'] for sentence in train_set]
        X = self.elmo.sents2elmo(tokens)
        max_sentence_length = max([sentence.shape[0] for sentence in X])
        input = torch.zeros(len(X), max_sentence_length, X[0].shape[1])
        for i, sentence in enumerate(X):
            input[i, :sentence.shape[0], :] = torch.from_numpy(sentence)
        return input.to(self.device), max_sentence_length

    def _create_labels(self, train_set, max_sentence_length, tags_columns) -> torch.tensor:
        tag_col_names = tags_columns + ['not_name']
        labels = torch.ones(len(train_set), max_sentence_length, dtype=torch.long) * len(tags_columns)
        for i, sentence in enumerate(train_set):
            labels[i, :len(sentence)] = torch.from_numpy(sentence[tag_col_names].values).argmax(1)
        labels = labels.view(labels.shape[0] * labels.shape[1])
        return labels.to(self.device)

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
        output, (hn, cn) = self.lstm(input[0])
        output = self.hidden2label(output)
        return output
