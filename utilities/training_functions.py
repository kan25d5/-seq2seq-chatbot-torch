import os
import optuna
from typing import List
from utilities.vocab import TanakaVocabs
from sklearn.model_selection import train_test_split
from dataloader.twitter_dataset import TwitterDataset
from dataloader.twitter_transform import TwitterTransform
from utilities.constant import MAXLEN, TOP_WORDS, CHAR2ID


# 転移学習の際の事前学習モデル
PRE_TRAINED_MODEL = "output/no_optuna/normal/EN6_DN6_ED0.2_DD0.2.pth"
# EarlyStoppingのpatience数
PATIENCE = 3


def get_datasets(sentiment_type: str, pkl_list: List[str]) -> List[TwitterDataset]:
    check_pkl = all(os.path.exists(filepath) for filepath in pkl_list)

    if check_pkl:
        return _load_datasets(pkl_list)
    else:
        return _make_datasets(sentiment_type)


def _load_datasets(pkl_list):
    print("The pkl data of the dataset exists.")
    print("\t" + "execute _load_datasets()")

    all_datasets = [TwitterDataset(MAXLEN, TwitterTransform()) for _ in range(3)]
    all_datasets[0].load_corpus_pkl(pkl_list[0])
    all_datasets[1].load_corpus_pkl(pkl_list[1])
    all_datasets[2].load_corpus_pkl(pkl_list[2])

    return all_datasets


def _make_datasets(sentiment_type):
    from sklearn.model_selection import train_test_split
    from utilities.functions import load_json

    def make_twitter_dataset(x, y):
        from dataloader.twitter_dataset import TwitterDataset
        from utilities.constant import MAXLEN

        dataset = TwitterDataset(limit_len=MAXLEN)
        dataset.messages = x
        dataset.responses = y
        return dataset

    print("The pkl data of the dataset does not exist.")
    print("\t" + "execute _make_datasets()")

    filepath = "output/{}.json".format(sentiment_type)
    corpus = load_json(filepath)

    X = [turn[0] for turn in corpus]
    y = [turn[1] for turn in corpus]

    X_train, X_other, y_train, y_other = train_test_split(X, y, test_size=0.1)
    X_val, X_test, y_val, y_test = train_test_split(X_other, y_other, test_size=0.3)
    all_datasets = [
        make_twitter_dataset(X_train, y_train),
        make_twitter_dataset(X_val, y_val),
        make_twitter_dataset(X_test, y_test),
    ]

    return all_datasets


def get_vocab(all_datasets: List[TwitterDataset], pkl_list: List[str]):
    check_pkl = all(os.path.exists(filepath) for filepath in pkl_list)
    if check_pkl:
        return _get_vocab_exists()
    else:
        return _get_vocab_no_exists(all_datasets, pkl_list)


def _get_vocab_exists():
    vocabs = TanakaVocabs(TOP_WORDS)
    if os.path.exists(CHAR2ID):
        vocabs.load_char2id_pkl(CHAR2ID)
        return vocabs
    else:
        raise ValueError("loaded dataset pkl, but char2id does not exist.")


def _get_vocab_no_exists(all_datasets, pkl_list):
    vocabs = TanakaVocabs(TOP_WORDS)

    if os.path.exists(CHAR2ID):
        vocabs.load_char2id_pkl(CHAR2ID)
    else:
        vocabs.fit(all_datasets)
        vocabs.save_char2id_pkl(CHAR2ID)

    all_datasets = vocabs.transform(all_datasets)
    all_datasets[0].save_corpus_pkl(pkl_list[0])
    all_datasets[1].save_corpus_pkl(pkl_list[1])
    all_datasets[2].save_corpus_pkl(pkl_list[2])

    return vocabs


def get_dataloader(all_datasets: List[TwitterDataset], batch_size: int):
    from dataloader.tanaka_dataloader import TanakaDataLoader

    train_dataloader = TanakaDataLoader(all_datasets[0], batch_size=batch_size)
    val_dataloader = TanakaDataLoader(all_datasets[1], batch_size=batch_size)
    test_dataloader = TanakaDataLoader(all_datasets[2], batch_size=1, random_state=0)

    all_dataloader = [train_dataloader, val_dataloader, test_dataloader]
    return all_dataloader


def training_optuna(trial: optuna.Trial, args, model_data):
    """optunaによるハイパラ探索（目的関数）"""
    # --------------------------------------------------
    # 必要ライブラリのインポート
    # --------------------------------------------------
    import torch
    import pytorch_lightning as pl
    from multiprocessing import freeze_support
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
    from dataloader.tanaka_dataloader import TanakaDataLoader
    from models.seq2seq_transformer import Seq2Seq
    from utilities.callbacks import DisplayPredictDialogue, SaveModelParams

    # --------------------------------------------------
    # 事前準備
    # --------------------------------------------------
    # 必要なデータの取り出し
    all_datasets = model_data["all_datasets"]
    vocabs = model_data["vocabs"]
    all_dataloader = model_data["all_dataloader"]

    # ハイパラ設定
    encoder_num_layers = trial.suggest_int("encoder_num_layers", 3, 8)
    decoder_num_layers = trial.suggest_int("decoder_num_layers", 3, 8)
    encoder_dropout = trial.suggest_float("encoder_dropout", 0.1, 0.6)
    decoder_dropout = trial.suggest_float("decoder_dropout", 0.1, 0.6)
    learning_ratio = trial.suggest_float("learning_ratio", 0.00001, 0.0005)

    # 現在のハイパラ設定情報の画面出力
    print("ハイパラ探索：")
    print("\tencoder_num_layers : {}".format(encoder_num_layers))
    print("\tdecoder_num_layers : {}".format(decoder_num_layers))
    print("\tencoder_dropout : {:.3f}".format(encoder_dropout))
    print("\tdecoder_dropout : {:.3f}".format(decoder_dropout))
    print("\tlearning_ratio : {:.6f}".format(learning_ratio))

    # Source/Targetの語彙サイズ
    src_vocab_size = len(vocabs.vocab_X.char2id)
    tgt_vocab_size = len(vocabs.vocab_y.char2id)

    # 学習モデルの保存先
    folderpath = "output/optuna/{}".format(args.sentiment_type)
    filename = "EN{}_DN{}_ED{:.3f}_DD{:.3f}".format(
        encoder_num_layers, decoder_num_layers, encoder_dropout, decoder_dropout
    )

    # コールバックの定義
    callbacks = [
        DisplayPredictDialogue(
            vocabs,
            TanakaDataLoader(all_datasets[0], 1, random_state=0),
            TanakaDataLoader(all_datasets[1], 1, random_state=0),
            translation_direction="jaja",
            folderpath=folderpath,
            filename=filename,
        ),
        SaveModelParams(folderpath, filename),
        EarlyStopping("val_loss", verbose=True, mode="min", patience=PATIENCE),
        PyTorchLightningPruningCallback(trial, "val_loss"),
    ]

    # loggerの定義
    version = "optuna/{}".format(args.sentiment_type)
    logger = (TensorBoardLogger(os.getcwd(), version=version),)

    # --------------------------------------------------
    # モデルの定義
    # --------------------------------------------------
    # モデルの定義
    model = Seq2Seq(
        src_vocab_size,
        tgt_vocab_size,
        encoder_num_layers=encoder_num_layers,
        decoder_num_layers=decoder_num_layers,
        encoder_dropout=encoder_dropout,
        decoder_dropout=decoder_dropout,
        learning_ratio=learning_ratio,
    )

    # sentiment_typeがneg/posならば転移学習
    # 事前学習モデルのパラメータをロードする．
    if args.sentiment_type == "neg" or args.sentiment_type == "pos":
        model.load_state_dict(torch.load(PRE_TRAINED_MODEL), strict=False)

    # --------------------------------------------------
    # モデルの学習
    # --------------------------------------------------
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=args.epoch_size,
        accelerator="gpu",
        devices=-1,
    )

    # Windows環境なら必要
    # freeze_support()
    trainer.fit(model, train_dataloaders=all_dataloader[0], val_dataloaders=all_dataloader[1])
    trainer.test(model, all_dataloader[2])

    torch.save(model.state_dict(), os.path.join(folderpath, filename + ".pth"))

    return trainer.callback_metrics["val_loss"].item()


def training_no_optuna(args, model_data):
    """ モデルのトレーニング """

    # --------------------------------------------------
    # 必要ライブラリのインポート
    # --------------------------------------------------
    import torch
    import pytorch_lightning as pl
    from multiprocessing import freeze_support
    from pytorch_lightning.callbacks import EarlyStopping
    from pytorch_lightning.loggers import TensorBoardLogger
    from dataloader.tanaka_dataloader import TanakaDataLoader
    from models.seq2seq_transformer import Seq2Seq
    from utilities.callbacks import DisplayPredictDialogue, SaveModelParams

    # --------------------------------------------------
    # 事前準備
    # --------------------------------------------------

    # 必要なデータの取り出し
    all_datasets = model_data["all_datasets"]
    vocabs = model_data["vocabs"]
    all_dataloader = model_data["all_dataloader"]

    # Source/Targetの語彙数
    src_vocab_size = len(vocabs.vocab_X.char2id)
    tgt_vocab_size = len(vocabs.vocab_y.char2id)

    # 学習モデルの保存先
    folderpath = "output/no_optuna/{}".format(args.sentiment_type)
    filename = "EN{}_DN{}_ED{:.3f}_DD{:.3f}".format(
        args.encoder_num_layers,
        args.decoder_num_layers,
        args.encoder_dropout,
        args.decoder_dropout,
    )

    # コールバックの定義
    callbacks = [
        DisplayPredictDialogue(
            vocabs,
            TanakaDataLoader(all_datasets[0], 1, random_state=0),
            TanakaDataLoader(all_datasets[1], 1, random_state=0),
            translation_direction="jaja",
            folderpath=folderpath,
            filename=filename,
        ),
        SaveModelParams(folderpath, filename),
        EarlyStopping("val_loss", verbose=True, mode="min", patience=3),
    ]

    # loggerの定義
    version = "no_optuna/{}".format(args.sentiment_type)
    logger = TensorBoardLogger(os.getcwd(), version=version)

    # --------------------------------------------------
    # モデルの定義
    # --------------------------------------------------
    model = Seq2Seq(
        src_vocab_size,
        tgt_vocab_size,
        encoder_num_layers=args.encoder_num_layers,
        decoder_num_layers=args.decoder_num_layers,
        encoder_dropout=args.encoder_dropout,
        decoder_dropout=args.decoder_dropout,
        learning_ratio=args.learning_ratio,
    )

    # neg/posならば転移学習．
    # 事前学習モデルをロードする
    if args.sentiment_type == "neg" or args.sentiment_type == "pos":
        model.load_state_dict(torch.load(PRE_TRAINED_MODEL), strict=False)

    # --------------------------------------------------
    # モデルの学習
    # --------------------------------------------------
    trainer = pl.Trainer(
        callbacks=callbacks,
        logger=logger,
        max_epochs=args.epoch_size,
        accelerator="gpu",
        devices=-1,
    )

    # Windows環境なら必要
    # freeze_support()
    trainer.fit(model, train_dataloaders=all_dataloader[0], val_dataloaders=all_dataloader[1])
    trainer.test(model, all_dataloader[2])

    torch.save(model.state_dict(), os.path.join(folderpath, filename + ".pth"))


def training(args):
    # --------------------------------------------------
    # コマンドライン引数
    # --------------------------------------------------
    sentimet_type = args.sentiment_type
    batch_size = args.batch_size
    epoch_size = args.epoch_size
    is_optuna = args.is_optuna
    n_trials = args.n_trials

    print("コマンドライン引数")
    print("\tbatch_size : {}".format(batch_size))
    print("\tepoch_size : {}".format(epoch_size))
    print("\tis_optuna : {}".format(is_optuna))
    print("\tn_trials : {}".format(n_trials))

    # --------------------------------------------------
    # 定数
    # --------------------------------------------------
    dataset_train_pkl = "dataloader/twitter_dataset_{}_train.model".format(sentimet_type)
    dataset_val_pkl = "dataloader/twitter_dataset_{}_val.model".format(sentimet_type)
    dataset_test_pkl = "dataloader/twitter_dataset_{}_test.model".format(sentimet_type)
    pkl_list = [
        dataset_train_pkl,
        dataset_val_pkl,
        dataset_test_pkl,
    ]

    # --------------------------------------------------
    # DataLoaderとVocabsを作成
    # --------------------------------------------------
    all_datasets = get_datasets(sentimet_type, pkl_list)
    vocabs = get_vocab(all_datasets, pkl_list)
    all_dataloader = get_dataloader(all_datasets, batch_size)
    model_data = {
        "all_datasets": all_datasets,
        "vocabs": vocabs,
        "all_dataloader": all_dataloader,
    }

    # --------------------------------------------------
    # トレーニング
    # --------------------------------------------------
    if is_optuna:
        study = optuna.create_study(direction="minimize")
        study.optimize(lambda trial: training_optuna(trial, args, model_data), n_trials=n_trials)
    else:
        training_no_optuna(args, model_data)
