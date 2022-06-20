import json
import math
import oseti
import glob
from dataloader.twitter_transform import TwitterTransform
from utilities.constant import TWITTER_CORPUS_FOLDER
from utilities.functions import load_json


analyzer = oseti.Analyzer()
transform = TwitterTransform()


def get_sentiment_dialogue(limit_len=60):
    sentiment_dialogue = {"positive": [], "negative": [], "normal": []}

    for filepath in glob.glob(TWITTER_CORPUS_FOLDER + "*.json"):
        corpus_dialogues = load_json(filepath)
        print("\tload corpus : " + filepath)
        for dialogue in corpus_dialogues:
            for i in range(len(dialogue) - 1):
                msg = dialogue[i]["text"]
                res = dialogue[i + 1]["text"]

                msg = transform(msg)
                res = transform(res)

                msg_len = len(msg.split())
                res_len = len(res.split())
                try:
                    res_score = analyzer.analyze(res.replace(" ", ""))
                except IndexError:
                    continue

                if type(res_score) == list and len(res_score) > 0:
                    res_score = sum(res_score) / len(res_score)
                elif type(res_score) == list and len(res_score) == 0:
                    res_score = 0
                elif type(res_score) == float:
                    pass
                elif type(res_score) == int:
                    pass
                else:
                    raise ValueError("想定しない型：res_scoreの型は{}".format(type(res_score)))

                if (msg_len < 5 or msg_len >= limit_len) or (
                    res_len < 5 or res_len >= limit_len
                ):
                    continue
                if res_score == 0:
                    sentiment_dialogue["normal"].append((msg, res))
                    continue

                if res_score > 0:
                    sentiment_dialogue["positive"].append((msg, res))
                else:
                    sentiment_dialogue["negative"].append((msg, res))

        print("\tmsg :" + msg)
        print("\tres :" + res)
        print("\tres_score : {}".format(res_score))
        print("\tnegapos size")
        print("\t\t pos  : {}".format(len(sentiment_dialogue["positive"])))
        print("\t\t neg  : {}".format(len(sentiment_dialogue["negative"])))
        print("\t\t norm : {}".format(len(sentiment_dialogue["normal"])))

    return sentiment_dialogue


def main():
    sentiment_dialogue = get_sentiment_dialogue()
    pos_dialogue = sentiment_dialogue["positive"]
    neg_dialogue = sentiment_dialogue["negative"]
    print("pos_dialogue len : {}".format(len(pos_dialogue)))
    print("neg_dialogue len : {}".format(len(neg_dialogue)))

    with open("pos.json", "w") as f:
        f.write(json.dumps(pos_dialogue, ensure_ascii=False, indent=4))
    with open("neg.json", "w") as f:
        f.write(json.dumps(neg_dialogue, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    main()
