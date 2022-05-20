import glob
from dataloader.twitter_transform import TwitterTransform
from utilities.functions import load_json
from utilities.constant import TWITTER_CORPUS_FOLDER

""" 
系列サイズを制限する際の対話ペアの減少数確認 

結果：
999 : 1.0
140 : 0.9999923520699123
130 : 0.9999821548297954
120 : 0.9999439151793567
110 : 0.9996494698709795
100 : 0.9967814960880838
90 : 0.9758210690276676
80 : 0.9228056176595805
70 : 0.8707257630722245
60 : 0.8170513150615786
50 : 0.7508100432617912
40 : 0.6643909825805646
30 : 0.5464815697631437
20 : 0.37538972577072016

"""


class CorpusSize(object):
    def __init__(self, corpus_files: list) -> None:
        self.transform = TwitterTransform()
        self.messages = []
        self.responses = []

        for filepath in corpus_files:
            self._load_corpus_file(filepath)

    def _load_corpus_file(self, filepath: str):
        corpus_file = load_json(filepath)
        for dialogue in corpus_file:
            for i in range(len(dialogue) - 1):
                message = dialogue[i]["text"]
                response = dialogue[i + 1]["text"]

                message = self.transform(message)
                response = self.transform(response)

                self.messages.append(message)
                self.responses.append(response)

    def __len__(self):
        return len(self.messages)


class CorpusLimiedSize(object):
    def __init__(self, corpus_size: CorpusSize, limit_len: int) -> None:
        corpus_size = corpus_size
        self.limit_len = limit_len
        self.count = 0

        messages = corpus_size.messages
        responses = corpus_size.responses

        for msg, res in zip(messages, responses):
            if len(msg.split()) > limit_len:
                continue
            if len(res.split()) > limit_len:
                continue
            self.count += 1

    def __len__(self):
        return self.count


def main():
    files = glob.glob(TWITTER_CORPUS_FOLDER + "*.json")
    corpus_size_ = CorpusSize(files)

    limit_len = [999] + list(range(140, 10, -10))
    corpus_limited_sizes = [CorpusLimiedSize(corpus_size_, lim) for lim in limit_len]
    max_size = len(corpus_limited_sizes[0])

    for corpus_limited_size in corpus_limited_sizes:
        ratio = len(corpus_limited_size) / max_size
        print("{} : {}".format(corpus_limited_size.limit_len, ratio))


if __name__ == "__main__":
    main()
