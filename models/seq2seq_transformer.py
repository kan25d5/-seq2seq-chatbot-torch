from typing import Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch import Tensor
from torchmetrics import Accuracy
from layers.seq2seq_transformer_layers import Transformer


class Seq2Seq(pl.LightningModule):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        encoder_num_layers=6,
        decoder_num_layers=6,
        encoder_dropout=0.1,
        decoder_dropout=0.1,
        emb_size=512,
        maxlen=140,
        padding_idx=0,
        learning_ratio=0.0001,
        eos_idx=2,
    ):
        # 親クラスのイニシャライザ
        super().__init__()

        # モデルの定義
        self.model = Transformer(
            src_vocab_size,
            tgt_vocab_size,
            encoder_num_layers,
            decoder_num_layers,
            encoder_dropout,
            decoder_dropout,
            emb_size,
            maxlen,
            padding_idx,
            eos_idx=eos_idx,
        )

        # フィールド値の定義
        self.learning_ratio = learning_ratio
        self.acc = Accuracy()
        self.criterion = nn.CrossEntropyLoss(ignore_index=padding_idx)

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        return self.model(source, target)

    def compute_loss(self, preds: Tensor, target: Tensor) -> Tensor:
        preds = preds.reshape(-1, preds.shape[-1])
        target = target.reshape(-1)
        loss = self.criterion(preds, target)
        return loss

    def compute_acc(self, preds: Tensor, target: Tensor) -> Tensor:
        preds = preds.reshape(-1, preds.shape[-1])
        target = target.reshape(-1)
        acc = self.acc(preds, target)
        return acc

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        src, tgt = batch
        batch_size = src.size(0)
        tgt_out = tgt[1:, :]
        preds = self.forward(src, tgt)

        loss = self.compute_loss(preds, tgt_out)
        self.log(
            "train_loss",
            value=loss,
            on_step=False,
            batch_size=batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        src, tgt = batch
        batch_size = src.size(0)
        tgt_out = tgt[1:, :]
        preds = self.forward(src, tgt)

        loss = self.compute_loss(preds, tgt_out)
        acc = self.compute_acc(preds, tgt_out)

        self.log(
            "val_loss",
            value=loss,
            on_step=False,
            batch_size=batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_acc",
            value=acc,
            on_step=False,
            batch_size=batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx):
        src, tgt = batch
        batch_size = src.size(0)
        tgt_out = tgt[1:, :]
        preds = self.forward(src, tgt)

        loss = self.compute_loss(preds, tgt_out)
        acc = self.compute_acc(preds, tgt_out)

        self.log(
            "test_loss",
            value=loss,
            on_step=False,
            batch_size=batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_acc",
            value=acc,
            on_step=False,
            batch_size=batch_size,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_ratio, betas=(0.9, 0.98), eps=1e-9)

