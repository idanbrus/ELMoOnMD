import torch
from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim: int = 1024,
                 hidden_dim: int = 256,
                 n_tags: int = 1,
                 device='cpu',
                 p_dropout=0.5):
        super().__init__()
        self.layer_weights = torch.ones((3, 1, 1, 1), requires_grad=True, device=device) / 3
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(p=p_dropout)
        self.hidden2label = nn.Linear(hidden_dim * 2, n_tags)

    def forward(self, *input):
        output = (input[0] * self.layer_weights).sum(dim=0)
        output = output.transpose(0, 1)
        output, (hn, cn) = self.lstm(output)
        output = output.transpose(0, 1)
        output = self.relu1(output)
        output = self.dropout(output)
        output = self.hidden2label(output)
        return output
