import re
import emoji
import neologdn
import unicodedata
from fugashi import Tagger

RE_PATTERNS = [r"#[^#\s]*", r"https?://[\w/:%#\$&\?\(\)~\.=\+\-]+", r"@[a-zA-Z\d_]+"]
REMOVE_CHARS = ["�"]


class TwitterTransform(object):
    def __init__(self) -> None:
        self.tagger = Tagger("-Owakati")
        self.re_removes = [re.compile(pattern) for pattern in RE_PATTERNS]

    def __call__(self, text: str):
        for rc in REMOVE_CHARS:
            text = text.replace(rc, "")
        for re_remove in self.re_removes:
            text = re_remove.sub("", text)

        text = text.strip()
        text = text.lower()
        text = unicodedata.normalize("NFKC", text)
        text = neologdn.normalize(text)
        text = emoji.replace_emoji(text, replace="")

        wakati = self.tagger.parse(text)
        return wakati


if __name__ == "__main__":
    transform = TwitterTransform()
    print(
        transform(
            "@djmatsunaga 悲しい告知をしています！ #cnann https://twitter.com/creepynuts_ann/status/1527126916092030976"
        )
    )
    print(
        transform(
            "🎁 #オリコンプレゼント  ㊗『オールナイトニッポン』55周年 🎉  月曜日担当《Creepy Nuts》の 直筆サイン入りチェキをプレゼント❗  📻 5/30(月)13時〆 1⃣@oricon  フォロー 2⃣この投稿をRT 3⃣↓記入 https://t.co/6nSH787Dkc"
        )
    )
    print(
        transform(
            "ほらもう私はもう嫌ですやっぱ可愛いからうみ覚えててくれたんだよ���なんだよ顔小さすぎだしお前可愛すぎ���(怒ってない) https://t.co/dFxXLGaVUA"
        )
    )

