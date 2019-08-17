from typing import List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from ELMoForManyLangs.elmoformanylangs import Embedder
from elmo_on_md.model.bi_lstm import BiLSTM


class NER():
    def __init__(self, elmo: Embedder, n_tags: int = 8, pos_weight: float = 1):
        self.elmo = elmo
        self.n_tags = n_tags
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # set up the model
        self.model = BiLSTM(n_tags=n_tags + 1, p_dropout=0.8, device=self.device).to(self.device)
        weights = torch.ones(n_tags) * pos_weight
        weights = torch.cat([weights, torch.ones(1)]).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weights)  # Binary cross entropy
        self.optimizer = Adam(self.model.parameters(), lr=1e-3)

    def train(self, train_set: List[pd.DataFrame],
              val_set: List[pd.DataFrame],
              tags_columns: List['str'],
              n_epochs: int = 10,
              batch_size: int = 64):

        X, max_sentence_length = self._create_input(train_set)
        y = self._create_labels(train_set, max_sentence_length, tags_columns)

        X_val, max_sentence_length = self._create_input(val_set)
        y_val = self._create_labels(val_set, max_sentence_length, tags_columns)

        # create input for the model

        for epoch in range(n_epochs):
            loss = 0
            batch_generator = self._chunker_list(X, y, batch_size)
            for batch_X, batch_y in batch_generator:
                self.optimizer.zero_grad()

                output = self.model(batch_X)
                output = output.view(output.shape[0] * output.shape[1], -1)  # flatten the results

                loss = self.criterion(output, batch_y.view(-1))

                loss.backward(retain_graph=True)
                self.optimizer.step()

            # Validate
            with torch.no_grad():
                self.model = self.model.eval()
                output = self.model(X_val.to(self.device))
                # print(output[0]) #TODO delete
                self.model = self.model.train()
                output = output.view(output.shape[0] * output.shape[1], -1)  # flatten the results

                val_loss = self.criterion(output, y_val.view(-1).to(self.device))
                print(f"Epoch: {epoch}\t Train Loss: {loss}\t Validation Loss: {val_loss}")
        return self

    def predict(self, test_set: List[pd.DataFrame]) -> torch.tensor:
        y_pred = self.predict_proba(test_set)
        y_pred = y_pred.argmax(dim=-1)
        return y_pred.to('cpu')

    def predict_proba(self, test_set:List[pd.DataFrame]) -> torch.tensor:
        with torch.no_grad():
            X, max_sentence_length = self._create_input(test_set)
            self.model = self.model.eval()
            y_pred = self.model(X.to(self.device))
            self.model = self.model.train()
        return y_pred

    def _create_input(self, train_set) -> Tuple[torch.tensor, int]:
        tokens = [sentence['word'] for sentence in train_set]
        X = self.elmo.sents2elmo(tokens, output_layer=-2)
        max_sentence_length = max([sentence.shape[1] for sentence in X])
        input = torch.zeros(3, len(X), max_sentence_length, X[0].shape[-1])
        for i, sentence in enumerate(X):
            input[:, i, :sentence.shape[1], :] = torch.from_numpy(sentence)
        return input, max_sentence_length

    def _create_labels(self, train_set, max_sentence_length, tags_columns) -> torch.tensor:
        tag_col_names = tags_columns + ['not_name']
        labels = torch.ones(len(train_set), max_sentence_length, dtype=torch.long) * len(tags_columns)
        for i, sentence in enumerate(train_set):
            labels[i, :len(sentence)] = torch.from_numpy(sentence[tag_col_names].values).argmax(1)
        return labels

    def _chunker_list(self, x, y, n_batches):
        for i in range(0, x.shape[1], n_batches):
            yield x[:, i:i + n_batches].to(self.device), y[i:i + n_batches].to(self.device)
