import torch
import torch.nn as nn
from torch import Tensor
from dataloader.twitter_transform import TwitterTransform
from models.seq2seq_transformer import Seq2Seq
from utilities.vocab import TanakaVocabs


class GenerateResponse(object):
    def __init__(self, model: nn.Module = None) -> None:
        # ユーティリティクラスのインスタンス化
        self.vocabs = TanakaVocabs()
        self.transform = TwitterTransform()
        self.vocabs.load_char2id_pkl("utilities/char2id.model")

        # モデルのインスタンス化＆ロード
        input_dim = len(self.vocabs.vocab_X.char2id)
        output_dim = len(self.vocabs.vocab_y.char2id)

        if model is None:
            self.model = Seq2Seq(input_dim, output_dim, maxlen=60 + 8)
            self.model.load_state_dict(torch.load("output/model.pth"))
        else:
            self.model = model

    def __greedy_decode(self, src: Tensor):
        src_shape = (src.shape[0], src.shape[0])
        src_mask = torch.zeros(src_shape, dtype=torch.bool, device=self.model.device)
        memory = self.model.encode(src, src_mask)
        ys = torch.ones(1, 1).type(torch.long).to(self.model.device)

        for i in range(self.model.maxlen - 6):
            tgt_mask = (
                self.model._generate_square_subsequent_mask(ys.size(0))
                .type(torch.bool)
                .to(self.model.device)
            )
            out = self.model.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.model.generater(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()

            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

            if next_word == self.model.eos_idx:
                break

        return ys

    def _encode_x(self, x: str):
        x = self.transform(x)
        x = self.vocabs.vocab_X.encode(x, True, True)
        x = torch.LongTensor([x]).T
        return x

    def _encode_y(self, src: Tensor):
        ys = self.__greedy_decode(src)
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
