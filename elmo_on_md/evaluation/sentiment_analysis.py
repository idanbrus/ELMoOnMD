import torch
import torch.nn as nn
from ELMoForManyLangs.elmoformanylangs import Embedder
from typing import List, Dict
from torch.optim import Adam
import numpy as np
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_recall_fscore_support
import os

from elmo_on_md.model.bi_lstm import BiLSTM


# RECENT CHANGE
class MyBiLSTM(nn.Module):
    def __init__(self, embedding_dim: int = 1024, hidden_dim: int = 256, max_sentence_length: int = 64, n_tags=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=hidden_dim * 2, hidden_size=hidden_dim // 2, bidirectional=True)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, n_tags)
        self.softmax = nn.Softmax()

    def forward(self, input, lengths):
        X = nn.utils.rnn.pack_padded_sequence(input, lengths, enforce_sorted=False)
        output, (hn, cn) = self.lstm(input)
        output = self.dropout(output)
        output = nn.utils.rnn.pack_padded_sequence(output, lengths, enforce_sorted=False)
        output, (hn, cn) = self.lstm2(output)
        hidden = torch.cat([hn[0], hn[1]], dim=1)
        hidden = self.dropout(hidden)
        # output = self.fc1(output.reshape(output.shape[1], -1))
        output = self.fc1(hidden.squeeze())
        output = self.dropout(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.softmax(output)
        return output


class SentimentAnalysis():
    def __init__(self, elmos: List[Embedder], lr: float = 1e-4):
        self.elmos = elmos
        self.max_sentence_length = 90
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = MyBiLSTM(max_sentence_length=self.max_sentence_length).to(self.device)

        # Cross Entropy loss gets weights
        self.optimizer = Adam(self.model.parameters(), lr=lr)

    def train(self, train_set: Dict,
              val_set: Dict,
              n_epochs: int = 10,
              batch_size: int = 64,
              tb_dir: str = 'default'):

        labels = np.array(train_set['labels'])
        unique, counts = np.unique(labels, return_counts=True)
        weights = torch.FloatTensor(3).to(self.device)
        for (x, y) in list(zip(unique, counts)):
            weights[x] = 1.0 / y
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        X, lengths = self._create_input(train_set)
        y = self._create_labels(train_set)

        X_val, lengths_val = self._create_input(val_set)
        y_val = self._create_labels(val_set)

        # create the tensorboard
        path = os.path.join('../sentiment_runs/', tb_dir)  # , str(datetime.datetime.now()))
        writer = SummaryWriter(path)
        global_step = 0

        for epoch in range(n_epochs):
            batch_generator = self._chunker_list(X, y, lengths, batch_size)
            for batch_X, batch_y, batch_lengths in batch_generator:
                self.optimizer.zero_grad()

                output = self.model(batch_X, batch_lengths)
                loss = self.criterion(output, batch_y)

                loss.backward()
                self.optimizer.step()

            # Validate
            print(f"Epoch: {epoch}\t Train Loss: {loss}\t")
            continue
            with torch.no_grad():
                val_output = self.model(X_val, lengths_val)
                val_loss = self.criterion(val_output, y_val.to(self.device))
                print(f"Epoch: {epoch}\t Train Loss: {loss}\t Validation Loss: {val_loss}")

                writer.add_scalar('train_loss', loss, global_step=global_step)

                # validation set
                if global_step % 10 == 0:
                    output.to('cpu')
                    precision, recall, f_score, support = precision_recall_fscore_support(y_val.to('cpu'), np.argmax(
                        val_output.to('cpu').detach().numpy(), axis=1))
                    writer.add_scalar('validation/Precision', precision.mean(), global_step=global_step)
                    writer.add_scalar('validation/Recall', recall.mean(), global_step=global_step)
                    writer.add_scalar('validation/F_score', f_score.mean(), global_step=global_step)
                global_step += 1
        return self

    def predict(self, test_set: Dict, batch_size: int = 64):
        X, lengths = self._create_input(test_set)
        batch_generator = self._chunker_list(X, X, lengths, batch_size)
        y_pred = []
        for batch_X, _, batch_lengths in batch_generator:
            y_probas = self.model(batch_X.to(self.device), batch_lengths)
            batch_y_pred = np.argmax(y_probas.to('cpu').detach().numpy(), axis=1)
            y_pred.append(batch_y_pred)
        return np.concatenate(y_pred, axis=0)

    def _create_input(self, train_set):
        tokens = train_set['sentences']
        lengths = [min([len(s), self.max_sentence_length]) for s in train_set['sentences']]
        # We set the first axis as the sentence length, so that the RNN can go over it
        Xs = []
        for elmo in self.elmos:
            X = elmo.sents2elmo([sentence[:self.max_sentence_length] for sentence in tokens])
            Xs.append(X)

        total_embedding = 0
        for X in Xs:
            total_embedding += X[0].shape[1]
        input = torch.zeros(self.max_sentence_length, len(Xs[0]), total_embedding)

        embedding_start = 0
        for X in Xs:
            for i, sentence in enumerate(X):
                input[:sentence.shape[0], i, embedding_start:(embedding_start + sentence.shape[1])] = torch.from_numpy(
                    sentence)
            embedding_start += X[0].shape[1]

        return input, lengths

    def _create_labels(self, train_set):
        return torch.from_numpy(np.array(train_set['labels']).astype('long')).long()

    def _chunker_list(self, X, y, lengths, n):
        for i in range(0, X.shape[1], n):
            yield X[:, i:i + n].to(self.device), y[i:i + n].to(self.device), lengths[i:i + n]
