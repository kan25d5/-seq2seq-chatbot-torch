import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(self.input_dim, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)

    def forward(self, x):
        len_source_sequences = (x.t() > 0).sum(dim=-1).to("cpu")
        x = self.embedding(x)
        x = pack_padded_sequence(x, len_source_sequences, enforce_sorted=False)
        h, states = self.lstm(x)
        h, _ = pad_packed_sequence(h)
        return h, states


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(self.output_dim, self.hidden_dim)
        self.lstm = nn.LSTM(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x, states):
        x = self.embedding(x)
        ht, states = self.lstm(x, states)
        y = self.out(ht)
        return y, states

