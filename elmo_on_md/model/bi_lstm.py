from torch import nn


class BiLSTM(nn.Module):
    def __init__(self, embedding_dim: int = 1024,
                 hidden_dim: int = 256,
                 n_tags: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, bidirectional=True)
        self.relu = nn.ReLU()
        self.hidden2label = nn.Linear(hidden_dim * 2, n_tags)

    def forward(self, *input):
        output, (hn, cn) = self.lstm(input[0])
        output = self.relu(output)
        output = self.hidden2label(output)
        return output
