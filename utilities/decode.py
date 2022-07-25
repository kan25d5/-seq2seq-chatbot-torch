import torch
from torch import Tensor
from layers.seq2seq_transformer_layers import Transformer


def greedy_decode(model: Transformer, src: Tensor, is_view_minus_one=True) -> Tensor:
    src_shape = (src.shape[0], src.shape[0])
    src_mask = torch.zeros(src_shape, dtype=torch.bool, device=model.device)
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).type(torch.long).to(model.device)

    for i in range(model.maxlen - 6):
        tgt_mask = model._generate_square_subsequent_mask(ys.size(0))
        tgt_mask = tgt_mask.type(torch.bool)
        tgt_mask = tgt_mask.to(model.device)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generater(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)

        if next_word == model.eos_idx:
            break

    if is_view_minus_one:
        ys = ys.view(-1)

    return ys

