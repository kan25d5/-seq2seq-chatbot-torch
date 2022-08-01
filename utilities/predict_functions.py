import torch
from dataloader.twitter_transform import TwitterTransform
from models.seq2seq_transformer import Seq2Seq
from utilities.decode import greedy_decode
from utilities.vocab import TanakaVocabs
from utilities.constant import CHAR2ID, TOP_WORDS

"""
🐛

(chatbot) spcc-a40g19:~/project/seq2seq-chatbot-torch> python run.py predict normal
 msg :おはようございます
Traceback (most recent call last):
  File "/home/s2110184/project/seq2seq-chatbot-torch/utilities/predict_functions.py", line 64, in predict
    tgt = greedy_decode(model.model, src)
  File "/home/s2110184/project/seq2seq-chatbot-torch/utilities/decode.py", line 16, in greedy_decode
    out = model.decode(ys, memory, tgt_mask)
  File "/home/s2110184/project/seq2seq-chatbot-torch/layers/seq2seq_transformer_layers.py", line 85, in decode
    return self.decoder(self.pe(self.tgt_tok_emb(tgt)), memory, tgt_mask)
  File "/home/s2110184/anaconda3/envs/chatbot/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/s2110184/anaconda3/envs/chatbot/lib/python3.9/site-packages/torch/nn/modules/transformer.py", line 252, in forward
    output = mod(output, memory, tgt_mask=tgt_mask,
  File "/home/s2110184/anaconda3/envs/chatbot/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/s2110184/anaconda3/envs/chatbot/lib/python3.9/site-packages/torch/nn/modules/transformer.py", line 458, in forward
    x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
  File "/home/s2110184/anaconda3/envs/chatbot/lib/python3.9/site-packages/torch/nn/modules/transformer.py", line 475, in _mha_block
    x = self.multihead_attn(x, mem, mem,
  File "/home/s2110184/anaconda3/envs/chatbot/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1110, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/s2110184/anaconda3/envs/chatbot/lib/python3.9/site-packages/torch/nn/modules/activation.py", line 1038, in forward
    attn_output, attn_output_weights = F.multi_head_attention_forward(
  File "/home/s2110184/anaconda3/envs/chatbot/lib/python3.9/site-packages/torch/nn/functional.py", line 5301, in multi_head_attention_forward
    k = k.contiguous().view(k.shape[0], bsz * num_heads, head_dim).transpose(0, 1)
RuntimeError: shape '[1, 8, 64]' is invalid for input of size 2560
"""


MODEL_NORMAL = "output/no_optuna/normal/EN6_DN6_ED0.2_DD0.2.pth"
MODEL_NEUTRAL = ""
MODEL_NEG = ""
MODEL_POS = ""


def get_model(args, vocabs):
    src_vocab_size = len(vocabs.vocab_X.char2id)
    tgt_vocab_size = len(vocabs.vocab_y.char2id)

    model = Seq2Seq(
        src_vocab_size,
        tgt_vocab_size,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_dropout=args.encoder_dropout,
        decoder_dropout=args.decoder_dropout,
        learning_ratio=args.learning_ratio,
    )

    return model


def get_load_model(sentiment_type: str, model: Seq2Seq):
    if sentiment_type == "normal":
        model.load_state_dict(torch.load(MODEL_NORMAL), strict=False)
    elif sentiment_type == "neutral":
        model.load_state_dict(torch.load(MODEL_NEUTRAL), strict=False)
    elif sentiment_type == "neg":
        model.load_state_dict(torch.load(MODEL_NEG), strict=False)
    elif sentiment_type == "pos":
        model.load_state_dict(torch.load(MODEL_POS), strict=False)
    else:
        raise ValueError("sentiment_typeの指定はnormal/neutral/neg/posのいずれかのみ")

    return model


def predict(args):
    transform = TwitterTransform()
    vocabs = TanakaVocabs(TOP_WORDS)
    vocabs.load_char2id_pkl(CHAR2ID)

    model = get_model(args, vocabs)
    model = get_load_model(args.sentiment_type, model)

    while True:
        in_msg = input(" msg :")
        if in_msg == "":
            print("対話を終了します．")
            break

        X = vocabs.X_transform([transform(in_msg)])
        src = torch.LongTensor(X)

        tgt = greedy_decode(model.model, src)
        res = vocabs.y_decode(tgt.tolist())

        print(" res : " + res)
        print("=" * 25)

