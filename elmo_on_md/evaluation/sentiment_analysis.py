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


class MyRNN(nn.Module):
    def __init__(self, embedding_dim: int = 1024,
                 hidden_dim: int = 256,
                 n_tags: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(embedding_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.hidden2label = nn.Linear(hidden_dim // 2, n_tags)
        self.softmax = nn.Softmax()


    def forward(self, input,lengths):
        X = nn.utils.rnn.pack_padded_sequence(input,lengths,enforce_sorted=False)
        output, (hidden, something_else) = self.rnn(X)
        hidden = torch.cat([hidden[0],hidden[1]],dim=1)

        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        output = self.fc(hidden.squeeze())  # removed squueeze
        output = F.relu(output)
        output = self.hidden2label(output)
        output = self.softmax(output)
        return output

    def initHidden(self, input_size):  # remove
        return torch.zeros((1, input_size, self.hidden_dim))

#RECENT CHANGE
class MyBiLSTM(nn.Module):
    def __init__(self, embedding_dim: int = 1024, hidden_dim: int = 256, max_sentence_length: int = 64, n_tags=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.dropout = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // 2, n_tags)
        self.softmax = nn.Softmax()

    def forward(self, input,lengths):
        X = nn.utils.rnn.pack_padded_sequence(input,lengths,enforce_sorted=False)
        output, (hn, cn) = self.lstm(input)
        hidden = torch.cat([hn[0], hn[1]], dim=1)
        hidden = self.dropout(hidden)
        #output = self.fc1(output.reshape(output.shape[1], -1))
        output = self.fc1(hidden.squeeze())
        output = self.dropout(output)
        output = self.relu(output)
        output = self.fc2(output)
        output = self.softmax(output)
        return output


class SentimentAnalysis():
    def __init__(self, elmo: Embedder, lr: float = 1e-4):
        self.elmo = elmo
        self.max_sentence_length = 30
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
        X,lengths = self._create_input(train_set)
        y = self._create_labels(train_set)

        X_val,lengths_val = self._create_input(val_set)
        y_val = self._create_labels(val_set)

        # create the tensorboard
        path = os.path.join('../../sentiment_runs/', tb_dir)  # , str(datetime.datetime.now()))
        writer = SummaryWriter(path)
        global_step = 0

        for epoch in range(n_epochs):
            batch_generator = self._chunker_list(X, y,lengths,batch_size)
            epoch_loss = 0.0
            for batch_X,batch_y,batch_lengths in batch_generator:
                self.optimizer.zero_grad()

                output = self.model(batch_X,batch_lengths)
                loss = self.criterion(output, batch_y)

                loss.backward()
                self.optimizer.step()
                # epoch_loss += output.shape[1] * loss.item()
            # Validate
            with torch.no_grad():
                val_output = self.model(X_val.to(self.device),lengths_val)
                val_loss = self.criterion(val_output, y_val.to(self.device))
                print(f"Epoch: {epoch}\t Train Loss: {loss}\t Validation Loss: {val_loss}")

                writer.add_scalar('train_loss', loss, global_step=global_step)

                # validation set
                if global_step % 2 == 0:
                    output.to('cpu')
                    precision, recall, f_score, support = precision_recall_fscore_support(y_val.to('cpu'), np.argmax(
                        val_output.to('cpu').detach().numpy(), axis=1))
                    writer.add_scalar('validation/Precision', precision.mean(), global_step=global_step)
                    writer.add_scalar('validation/Recall', recall.mean(), global_step=global_step)
                    writer.add_scalar('validation/F_score', f_score.mean(), global_step=global_step)
                global_step += 1
        return self

    def predict(self, test_set: Dict):
        X,lengths = self._create_input(test_set)
        y_pred = self.model(X.to(self.device),lengths)
        return np.argmax(y_pred.to('cpu').detach().numpy(), axis=1)


    def _create_input(self, train_set):
        tokens = train_set['sentences']
        lengths = [min([len(s),self.max_sentence_length]) for s in train_set['sentences']]
        # We set the first axis as the sentence length, so that the RNN can go over it
        X = self.elmo.sents2elmo([sentence[:self.max_sentence_length] for sentence in tokens])
        input = torch.zeros(self.max_sentence_length, len(X), X[0].shape[1])
        for i, sentence in enumerate(X):
            input[:sentence.shape[0], i, :] = torch.from_numpy(sentence)

        return input,lengths

    def _create_labels(self, train_set):
        return torch.from_numpy(np.array(train_set['labels']).astype('long')).long()

    def _chunker_list(self, X,y,lengths, n):
        for i in range(0, X.shape[1], n):
            yield X[:, i:i + n].to(self.device), y[i:i + n].to(self.device),lengths[i:i+n]
