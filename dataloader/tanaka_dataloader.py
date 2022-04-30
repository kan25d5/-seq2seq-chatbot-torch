import numpy as np
import torch
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from dataloader.tanaka_dataset import TanakaDataset


class TanakaDataLoader(object):
    def __init__(
        self,
        dataset: TanakaDataset,
        batch_size: int,
        device="cuda",
        batch_first=False,
        random_state=None,
    ) -> None:
        self.dataset = list(zip(dataset.messages, dataset.responses))
        self.batch_size = batch_size
        self.device = device
        self.batch_first = batch_first
        self.random_state = random_state

        self._idx = 0

    def __len__(self):
        N = len(self.dataset)
        b = self.batch_size
        return N // b + bool(N % b)

    def __iter__(self):
        return self

    def __next__(self):
        if self._idx >= len(self.dataset):
            self._reset()
            raise StopIteration()

        x, t = zip(*self.dataset[self._idx : (self._idx + self.batch_size)])
        x = pad_sequences(x, padding="post")
        t = pad_sequences(t, padding="post")

        x = torch.LongTensor(x)
        t = torch.LongTensor(t)

        if not self.batch_first:
            x = x.t()
            t = t.t()

        self._idx += self.batch_size

        return x.to(self.device), t.to(self.device)

    def _reset(self):
        if self.shuffle:
            self.dataset = shuffle(self.dataset, random_state=self.random_state)
        self._idx = 0
