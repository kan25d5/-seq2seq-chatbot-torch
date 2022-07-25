import argparse

# --------------------------------------
# ArgumentParserによる属性追加
# --------------------------------------
parser = argparse.ArgumentParser(description="Twitter上の対話データを用いた応答予測モデルのトレーニングもしくは応答予測")
parser.add_argument("mode", help="predict, trainingのどちらかを指定")
parser.add_argument(
    "sentiment_type", type=str, help="pos/neg/neutralを指定し，同一性格タイプで学習する",
)
parser.add_argument(
    "-bs", "--batch_size", default=100, type=int, help="trainingの際のバッチサイズを指定(default=100)"
)
parser.add_argument(
    "-es", "--epoch_size", default=30, type=int, help="trainingの際のエポックサイズを指定(default=30)"
)
parser.add_argument(
    "-op", "--optuna", default=False, type=bool, help="optunaによるハイパラ探索(default=False)"
)
parser.add_argument(
    "-nt", "--n_trials", default=50, type=int, help="optunaによるハイパラ探索探索数(default=50)"
)
parser.add_argument(
    "-to", "--timeout", default=1800, type=int, help="optunaによるトライアル時間制限(default=1800sec)"
)
parser.add_argument(
    "-ed",
    "--encoder_dropout",
    default=0.2,
    type=float,
    help="Encoderのdropout割合を指定．--optunaがTrueなら無視．（default=0.2)",
)
parser.add_argument(
    "-dd",
    "--decoder_dropout",
    default=0.2,
    type=float,
    help="Decoderのdropout割合を指定．--optunaがTrueなら無視．（default=0.2)",
)
parser.add_argument(
    "-enl",
    "--encoder_num_layers",
    default=6,
    type=int,
    help="Encoderのレイヤー数を指定．--optunaがTrueなら無視．（default=6)",
)
parser.add_argument(
    "-dnl",
    "--decoder_num_layers",
    default=6,
    type=int,
    help="Decoderのレイヤー数を指定．--optunaがTrueなら無視．（default=6)",
)


# --------------------------------------
# コマンドライン引数による分岐関数
# --------------------------------------
def _switch_predict(args):
    sentiment_type = args.sentiment_type

    if sentiment_type == "neutral":
        pass
    elif sentiment_type == "neg" or sentiment_type == "pos":
        pass
    else:
        raise ValueError("sentiment_typeはneg/pos/neutralのいずれか")


def _switch_training(args):
    sentiment_type = args.sentiment_type

    if sentiment_type == "neutral":
        import training_neutral as tn

        tn.training_run(args)
    elif sentiment_type == "neg" or sentiment_type == "pos":
        pass
    else:
        raise ValueError("sentiment_typeはneg/pos/neutralのいずれか")


# --------------------------------------
# エントリーポイント
# --------------------------------------
def main():
    args = parser.parse_args()

    if args.mode == "predict":
        _switch_predict(args)
    elif args.mode == "training":
        _switch_training(args)
    else:
        raise argparse.ArgumentError(None, message="modeの指定が誤っています．")


if __name__ == "__main__":
    main()
