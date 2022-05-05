from typing import List
from gensim.models import KeyedVectors
from dataloader.tanaka_dataset import TanakaDataset
from utilities.vocab import Vocab
from collections import Counter


class TanakaVocabs(object):
    def __init__(
        self,
        datasets: List[TanakaDataset],
        wv: KeyedVectors,
        top_words=50000,
        X_bos=False,
        X_eos=False,
        y_bos=False,
        y_eos=False,
        translation_direction="enja",
    ) -> None:
        self.top_words = top_words
        self.wv = wv
        self.X_bos = X_bos
        self.y_bos = y_bos
        self.X_eos = X_eos
        self.y_eos = y_eos

        # Vocabクラスの作成
        if translation_direction == "enja":
            self.vocab_X = Vocab(top_words)
            self.vocab_y = JaVocab(wv, top_words)
        elif translation_direction == "jaen":
            self.vocab_X = JaVocab(wv, top_words)
            self.vocab_y = Vocab(top_words)
        else:
            raise ValueError("corpus_typeはenja, jaenのいずれかの値指定")

        # 各データセットに対して語彙を適合
        for dataset in datasets:
            self.vocab_X.fit(dataset.messages)
            self.vocab_y.fit(dataset.responses)

        # データセットをID列に書き換える
        for dataset in datasets:
            dataset.messages = self.vocab_X.transform(dataset.messages, X_bos, X_eos)
            dataset.responses = self.vocab_y.transform(dataset.responses, y_bos, y_eos)

    def X_transform(self, sentences):
        return self.vocab_X.transform(sentences, self.X_bos, self.X_eos)

    def y_transform(self, sentences):
        return self.vocab_y.transform(sentences, self.y_bos, self.y_eos)

    def X_decode(self, idx_seq):
        return self.vocab_X.decode(idx_seq)

    def y_decode(self, idx_seq):
        return self.vocab_y.decode(idx_seq)


class JaVocab(object):
    def __init__(self, wv: KeyedVectors, top_words=50000) -> None:
        self.wv = wv
        self.top_words = top_words

        self._words = []
        self._top_words = []
        self.special_words = ["<pad>", "<s>", "</s>", "<unk>"]
        self.pad_char = self.special_words[0]
        self.bos_char = self.special_words[1]
        self.eos_char = self.special_words[2]
        self.unk_char = self.special_words[3]

    def fit(self, sentences):
        for sentence in sentences:
            for w in sentence.split():
                self._words.append(w)

        # 出現頻度上位top_wordsだけ抽出し，_top_wordsに格納
        counter = Counter(self._words).most_common(
            self.top_words - len(self.special_words)
        )
        self._top_words = [c[0] for c in counter]

        # _top_wordsに1回だけ特殊トークンを追加
        for w in self.special_words:
            if w in self._top_words:
                self._top_words.append(w)

    def transform(self, sentences, bos=False, eos=False):
        output_t = []
        for sentence in sentences:
            output_t.append(self.encode(sentence, bos, eos))
        return output_t

    def encode(self, sentence, bos=False, eos=False):
        output_e = []

        for w in sentence.split():
            if w in self._top_words and w in self.wv.key_to_index:
                output_e.append(self.wv.key_to_index[w])
            else:
                output_e.append(self.wv.key_to_index[self.unk_char])

        if bos:
            output_e = [self.wv.key_to_index[self.bos_char]] + output_e
        if eos:
            output_e = output_e + [self.wv.key_to_index[self.eos_char]]

        return output_e

    def decode(self, idx_seq):
        return [self.wv.index_to_key[idx] for idx in idx_seq]
