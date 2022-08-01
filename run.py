import argparse


description = "Twitterコーパスを学習し，応答予測を行うモデル．"
parser = argparse.ArgumentParser(description=description)

help_mode = "training/predict モデルをトレーニングするか，モデルの応答予測を行うか．"
help_sentiment_type = "neg/pos/neutral 利用するモデルの選択．"
help_train_size = "全データ中の訓練データサイズの割合"
help_val_size = "訓練データ以外データにける検証データサイズの割合"
help_enl = "encoder layerのレイヤー数"
help_dnl = "decoder layerのレイヤー数"
help_edo = "encoderのdropout率"
help_ddo = "decoderのdropout率"
help_optuna = "optunaによるハイパラ探索を行うか．オプションを指定するとハイパラ探索を行う．"
help_tw = "語彙数"
help_nt = "optunaによるトライアル数"
help_lt = "学習率"

parser.add_argument("mode", type=str, default="training", help=help_mode)
parser.add_argument("sentiment_type", type=str, default="neg", help=help_sentiment_type)
parser.add_argument("-optuna", "--is_optuna", action="store_true", help=help_optuna)
parser.add_argument("-ts", "--train_size", type=float, default=0.9, help=help_train_size)
parser.add_argument("-vs", "--val_size", type=float, default=0.7, help=help_val_size)
parser.add_argument("-bs", "--batch_size", type=int, default=80, help="バッチサイズ")
parser.add_argument("-es", "--epoch_size", type=int, default=30, help="エポックサイズ")
parser.add_argument("-enl", "--encoder_num_layers", type=int, default=6, help=help_enl)
parser.add_argument("-dnl", "--decoder_num_layers", type=int, default=6, help=help_dnl)
parser.add_argument("-edo", "--encoder_dropout", type=float, default=0.2, help=help_edo)
parser.add_argument("-ddo", "--decoder_dropout", type=float, default=0.2, help=help_ddo)
parser.add_argument("-tw", "--top_words", type=int, default=80000, help=help_tw)
parser.add_argument("-nt", "--n_trials", type=int, default=100, help=help_nt)
parser.add_argument("-lt", "--learning_ratio", type=float, default=0.0001, help=help_lt)


def main():
    args = parser.parse_args()

    boot_mode = args.mode
    if boot_mode == "training":
        from utilities.training_functions import training

        training(args)
    elif boot_mode == "predict":
        from utilities.predict_functions import predict

        predict(args)
    else:
        raise ValueError("第一引数の指定はtraining/predictのいずれか．")


if __name__ == "__main__":
    main()
