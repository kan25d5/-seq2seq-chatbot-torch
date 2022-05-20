import dill
from dataloader.twitter_transform import TwitterTransform
from utilities.functions import load_json


class TwitterDataset(object):
    def __init__(self, limit_len=60) -> None:
        self.limit_len = limit_len
        self.transform = TwitterTransform()
        self.messages = []
        self.responses = []

    def load_corpus(self, corpus_filelist: list):
        for corpus_file in corpus_filelist:
            self._load_corpus_file(corpus_file)
        assert len(self.messages) == len(self.responses), "発話系列と応答系列のサイズが一致しない．"

    def load_corpus_pkl(self, filepath):
        with open(filepath, "rb") as f:
            self.messages = dill.load(f)
            self.responses = dill.load(f)

    def save_corpus_pkl(self, filepath):
        with open(filepath, "wb") as f:
            dill.dump(self.messages, f)
            dill.dump(self.responses, f)

    def _load_corpus_file(self, corpus_file: str):
        corpus_dialogues = load_json(corpus_file)
        for dialogue in corpus_dialogues:
            for i in range(len(dialogue) - 1):
                msg = dialogue[i]["text"]
                res = dialogue[i + 1]["text"]

                msg = self.transform(msg)
                res = self.transform(res)

                if len(msg.split()) > self.limit_len:
                    continue
                if len(res.split()) > self.limit_len:
                    continue

                self.messages.append(msg)
                self.responses.append(res)

