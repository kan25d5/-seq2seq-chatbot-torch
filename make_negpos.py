import glob
import json
import re
import unicodedata
import MeCab
import emoji
import oseti
import neologdn
import sengiri
from dataloader.twitter_transform import TwitterTransform
from utilities.functions import load_json
from utilities.constant import TWITTER_CORPUS_FOLDER


MAXLEN = 60

RE_PATTERNS = [r"#[^#\s]*", r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", r"@[a-zA-Z\d_]+"]
REMOVE_CHARS = ["�"]

NEGATION = ("ない", "ず", "ぬ")
PARELLEL_PARTICLES = ("か", "と", "に", "も", "や", "とか", "だの", "なり", "やら")


class Analyzer(oseti.Analyzer):
    def __init__(self, mecab_args=""):
        super().__init__(mecab_args)
        self.re_removes = [re.compile(pattern) for pattern in RE_PATTERNS]

    def preprocess(self, text):
        for rc in REMOVE_CHARS:
            text = text.replace(rc, "")
        for re_remove in self.re_removes:
            text = re_remove.sub("", text)

        text = text.strip()
        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
        text = neologdn.normalize(text)
        text = emoji.replace_emoji(text, replace="")

        return text

    def remove_empty_line(self, text: str):
        text_list = [line for line in text.split("\n") if line != ""]
        return "".join(text_list)

    def analyze(self, text):
        """ Calculate sentiment polarity scores per sentence
            Arg:
                text (str)
            Return:
                scores (float)
        """
        scores = []
        lemmas_list = []

        text = self.preprocess(text)
        if text == "":
            return 0, []

        text = self.remove_empty_line(text)
        sentences = sengiri.tokenize(text)
        for sentence in sentences:
            polarities, lemmas = self._calc_sentiment_polarity(sentence)
            if polarities:
                scores.append(sum(p[1] for p in polarities) / len(polarities))
            else:
                scores.append(0)
            lemmas_list.extend(lemmas)

        if len(sentences) <= 0:
            score = 0
        else:
            score = sum(scores) / len(sentences)
        return score, lemmas_list

    def _calc_sentiment_polarity(self, sentence):
        polarities = []
        lemmas = []
        n_parallel = 0
        substr_count = 0
        node = self.tagger.parseToNode(sentence)
        while node:
            if "BOS/EOS" not in node.feature:
                surface = node.surface
                substr_count += len(surface)
                lemma = node.surface
                wago = ""
                if lemma in self.word_dict:
                    polarity = 1 if self.word_dict[lemma] == "p" else -1
                    n_parallel += node.next.surface in PARELLEL_PARTICLES
                else:
                    wago = self._lookup_wago(lemma, lemmas)
                    if wago:
                        polarity = 1 if self.wago_dict[wago].startswith("ポジ") else -1
                    else:
                        polarity = None
                if polarity:
                    polarities.append([wago or lemma, polarity])
                elif (
                    polarities
                    and surface in NEGATION
                    and not self._has_arujanai(sentence[:substr_count])
                ):
                    polarities[-1][1] *= -1
                    if polarities[-1][0].endswith("-NEGATION"):
                        polarities[-1][0] = polarities[-1][0][:-9]
                    else:
                        polarities[-1][0] += "-NEGATION"
                    # parallel negation
                    if n_parallel and len(polarities) > 1:
                        n_parallel = (
                            len(polarities) if len(polarities) > n_parallel else n_parallel + 1
                        )
                        n_parallel = (
                            n_parallel + 1 if len(polarities) == n_parallel else n_parallel
                        )
                        for i in range(2, n_parallel):
                            polarities[-i][1] *= -1
                            if polarities[-i][0].endswith("-NEGATION"):
                                polarities[-i][0] = polarities[-i][0][:-9]
                            else:
                                polarities[-i][0] += "-NEGATION"
                        n_parallel = 0
                lemmas.append(lemma)
            node = node.next
        return polarities, lemmas


def save_json(filename, content):
    with open("output/{}.json".format(filename), "w") as f:
        f.write(json.dumps(content, ensure_ascii=False, indent=4))


def main():
    analyzer = Analyzer()
    transform = TwitterTransform()

    neg = []
    pos = []
    neutral = []
    errs = []
    for filepath in glob.glob(TWITTER_CORPUS_FOLDER + "*.json"):
        for dialogue in load_json(filepath):
            for i in range(len(dialogue) - 1):
                msg = dialogue[i]["text"]
                res = dialogue[i + 1]["text"]

                msg_lemmas = transform(msg)

                try:
                    res_score, res_lemmas = analyzer.analyze(res)
                except Exception as e:
                    errs.append({"err_msg": str(e), "msg": "".join(msg), "res": "".join(res)})
                    continue

                if len(msg_lemmas) > MAXLEN or len(res_lemmas) > MAXLEN:
                    continue

                msg_wakati = " ".join(msg_lemmas)
                res_wakati = " ".join(res_lemmas)

                if res_score > 0:
                    print("pos : ")
                    print("\t" + msg_wakati)
                    print("\t" + res_wakati)
                    pos.append([msg_wakati, res_wakati])
                elif res_score < 0:
                    print("neg : ")
                    print("\t" + msg_wakati)
                    print("\t" + res_wakati)
                    neg.append([msg_wakati, res_wakati])
                else:
                    neutral.append([msg_wakati, res_wakati])

    save_json("neg", neg)
    save_json("pos", pos)
    save_json("neutral", neutral)
    save_json("err", errs)


if __name__ == "__main__":
    main()
