import random
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from layers.seq2seq_layers import Encoder
from layers.seq2seq_attention_layers import Decoder
from gensim.models import KeyedVectors


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        wv: KeyedVectors,
        maxlen=20,
        teacher_forcing_rate=0.5,
        ignore_index=0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.maxlen = maxlen
        self.teacher_forcing_rate = teacher_forcing_rate
        self.criterion = nn.CrossEntropyLoss(
            reduction="mean", ignore_index=ignore_index
        )

        self.encoder = Encoder(self.input_dim, self.hidden_dim, wv)
        self.decoder = Decoder(self.hidden_dim, self.output_dim, wv)

    def forward(self, source, target=None, use_teacher_forcing=False):
        batch_size = source.size(1)

        if target is not None:
            len_target_seq = target.size(0)
        else:
            len_target_seq = self.maxlen

        hs, states = self.encoder(source)

        y = torch.ones((1, batch_size), dtype=torch.long, device=self.device)
        output_dim = (len_target_seq, batch_size, self.output_dim)
        output = torch.zeros(output_dim, device=self.device)

        for t in range(len_target_seq):
            out, states = self.decoder(y, hs, states, source=source)
            output[t] = out

            if use_teacher_forcing and target is not None:
                y = target[t].unsqueeze(0)
            else:
                y = out.max(-1)[1]

        return output

    def compute_loss(self, t, preds):
        t = t.contiguous().view(-1)
        preds = preds.contiguous().view(-1, preds.size(-1))
        loss = self.criterion(preds, t)
        return loss

    def training_step(self, batch, batch_idx):
        x, t = batch
        use_teacher_forcing = random.random() < self.teacher_forcing_rate
        preds = self.forward(x, t, use_teacher_forcing)
        loss = self.compute_loss(t, preds)
        return loss

    def validation_step(self, batch, batch_idx):
        x, t = batch
        preds = self.forward(x, t)
        loss = self.compute_loss(t, preds)
        return loss

    def test_step(self, batch, batch_idx):
        x, t = batch
        preds = self.forward(x, t)
        loss = self.compute_loss(t, preds)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), amsgrad=True)
