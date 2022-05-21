import dill
from typing import List
from collections import Counter
from dataloader.tanaka_dataset import TanakaDataset


class TanakaVocabs(object):
    def __init__(
        self,
        datasets: List[TanakaDataset],
        top_words=30000,
        X_bos=False,
        X_eos=False,
        y_bos=False,
        y_eos=False,
    ) -> None:
        self.vocab_X = Vocab(top_words)
        self.vocab_y = Vocab(top_words)

        # Fit words from all datasets.
        for dataset in datasets:
            self.vocab_X.fit(dataset.messages)
            self.vocab_y.fit(dataset.responses)

        # word-to-index transform from all datasets.
        for dataset in datasets:
            dataset.messages = self.vocab_X.transform(dataset.messages, X_bos, X_eos)
            dataset.responses = self.vocab_y.transform(dataset.responses, y_bos, y_eos)

    def X_transform(self, sentences):
        return self.vocab_X.transform(sentences)

    def y_transform(self, sentences):
        return self.vocab_y.transform(sentences)

    def X_decode(self, idx_seq):
        return self.vocab_X.decode(idx_seq)

    def y_decode(self, idx_seq):
        return self.vocab_y.decode(idx_seq)

    def load_char2id_pkl(self, filepath):
        with open(filepath, "rb") as f:
            self.vocab_X.char2id = dill.load(f)
            self.vocab_y.char2id = dill.load(f)
        self.vocab_X.id2char = {v: k for k, v in self.vocab_X.char2id.items()}
        self.vocab_y.id2char = {v: k for k, v in self.vocab_y.char2id.items()}

    def save_char2id_pkl(self, filepath):
        with open(filepath, "wb") as f:
            dill.dump(self.vocab_X.char2id, f)
            dill.dump(self.vocab_y.char2id, f)


class Vocab(object):
    def __init__(self, top_words=30000) -> None:
        self.top_words = top_words
        self._words = []
        self.char2id = {}
        self.id2char = {}
        self.special_words = ["<pad>", "<s>", "</s>", "<unk>"]
        self.pad_char = self.special_words[0]
        self.bos_char = self.special_words[1]
        self.eos_char = self.special_words[2]
        self.unk_char = self.special_words[3]

    def fit(self, sentences):
        for sentence in sentences:
            for w in sentence.split():
                self._words.append(w)

        counter = Counter(self._words).most_common(self.top_words)
        self._words = [c[0] for c in counter]

        self.char2id = {w: (len(self.special_words) + idx) for idx, w in enumerate(self._words)}
        for idx, w in enumerate(self.special_words):
            self.char2id[w] = idx

        self.id2char = {v: k for k, v in self.char2id.items()}

    def transform(self, sentences, bos=False, eos=False):
        output_t = []
        for sentence in sentences:
            output_t.append(self.encode(sentence, bos, eos))
        return output_t

    def encode(self, sentence, bos=False, eos=False):
        output_e = []
        unk_idx = self.char2id[self.unk_char]
        bos_idx = self.char2id[self.bos_char]
        eos_idx = self.char2id[self.eos_char]

        for w in sentence.split():
            if w in self.char2id:
                output_e.append(self.char2id[w])
            else:

                output_e.append(unk_idx)
        if bos:
            output_e = [bos_idx] + output_e
        if eos:
            output_e = output_e + [eos_idx]

        return output_e

    def decode(self, idx_seq):
        return [self.id2char[c] for c in idx_seq]
