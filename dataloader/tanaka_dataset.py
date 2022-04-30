"""田中コーパス用のDataset"""
import os
from utilities.constant import TANAKA_CORPUS_FOLDER


class TanakaDataset(object):
    def __init__(self, corpus_type="train", translation_direction="enja") -> None:
        if corpus_type not in ["train", "test", "dev"]:
            raise ValueError("corpus_typeはtrain, test, devのいずれかの値指定")
        if translation_direction not in ["enja", "jaen"]:
            raise ValueError("translation_directionはenja, jaenのいずれかの値指定")

        self.corpus_type = corpus_type
        self.translation_direction = translation_direction

        self.messages = []
        self.responses = []
        self._load_corpus()

    def _load_corpus(self):
        if self.translation_direction == "enja":
            self.messages = self._load_corpus_("en")
            self.responses = self._load_corpus_("ja")
        else:
            self.messages = self._load_corpus_("ja")
            self.responses = self._load_corpus_("en")
        assert len(self.messages) == len(self.responses), "発話リストと応答リストのサイズが一致しない．"

    def _load_corpus_(self, jaen):
        corpus_list = []
        filename = self.corpus_type + "." + jaen
        filepath = os.path.join(TANAKA_CORPUS_FOLDER, filename)

        with open(filepath) as f:
            corpus_list = f.readlines()

        return corpus_list

    def get_sentences(self):
        return self.messages + self.responses

    def __len__(self):
        return len(self.messages)

    def __iter__(self):
        return self

    def __getitem__(self, idx: int):
        x = self.messages[idx]
        t = self.responses[idx]

        return x, t
