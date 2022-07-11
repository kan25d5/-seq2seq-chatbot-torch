import torch
import torch.nn as nn
from torch import Tensor
from dataloader.twitter_transform import TwitterTransform
from utilities.vocab import TanakaVocabs
from utilities.decode import greedy_decode


class GenerateResponse(object):
    def __init__(
        self, model: nn.Module, vocabs: TanakaVocabs, transform=TwitterTransform()
    ) -> None:
        self.model = model
        self.vocabs = vocabs
        self.transform = transform

    def _encode_x(self, x: str):
        x = self.transform(x)
        x = self.vocabs.vocab_X.encode(x, True, True)
        x = torch.LongTensor([x]).T
        return x

    def _encode_y(self, src: Tensor):
        ys = greedy_decode(self.model, src)
        ys = ys.view(-1).tolist()
        y = "".join(self.vocabs.y_decode(ys))
        y = y.replace("<s>", "").replace("</s>", "")
        return y

    def generate_response(self, x: str):
        src = self._encode_x(x)
        y = self._encode_y(src)
        return y

    def __call__(self, x: str):
        return self.generate_response(x)
