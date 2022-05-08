from numpy import source
import torch
import torch.nn as nn
from gensim.models import KeyedVectors


class Attention(nn.Module):
    def __init__(self, output_dim, hidden_dim) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.W_a = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        self.W_c = nn.Parameter(torch.Tensor(hidden_dim + hidden_dim, output_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))

        nn.init.xavier_normal_(self.W_a)
        nn.init.xavier_normal_(self.W_c)

    def forward(self, ht, hs, source=None):
        # g = hs * Wa * ht
        score = torch.einsum("jik,kl->jil", (hs, self.W_a))
        score = torch.einsum("jik,lik->jil", (ht, score))

        # softmax(g)
        score = score - torch.max(score, dim=-1, keepdim=True)[0]
        score = torch.exp(score)

        # 系列長を取得し，パディングされた部分にマスク処理する
        if source is not None:
            mask_source = source.t().eq(0).unsqueeze(0)
            score.data.masked_fill_(mask_source, 0)

        a = score / torch.sum(score, dim=-1, keepdim=True)
        c = torch.einsum("jik,kil->jil", (a, hs))
        h = torch.cat((c, ht), -1)
        att = torch.tanh(torch.einsum("jik,kl->jil", (h, self.W_c)) + self.b)

        return att


class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, wv: KeyedVectors, padding_idx=0,num_layers=4) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embedding = nn.Embedding(
            self.output_dim, self.hidden_dim, padding_idx=padding_idx
        )
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers)
        self.attn = Attention(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

        nn.init.xavier_normal_(self.lstm.weight_ih_l0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.xavier_normal_(self.out.weight)

        self.embedding.weight = nn.Parameter(torch.from_numpy(wv.vectors))
        self.embedding.weight.requires_grad = False

    def forward(self, x, hs, states, source=source):
        x = self.embedding(x)
        ht, states = self.lstm(x, states)
        ht = self.attn(ht, hs, source=source)
        y = self.out(ht)
        return y, states
