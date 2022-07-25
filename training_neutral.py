import os
import pytorch_lightning as pl
from typing import List
from multiprocessing import freeze_support

import torch
from dataloader.tanaka_dataloader import TanakaDataLoader
from dataloader.twitter_dataset import TwitterDataset
from models.seq2seq_transformer import Seq2Seq
from utilities.callbacks import DisplayPredictDialogue, SaveModelParams
from utilities.training_functions import get_dataloader, get_datasets, get_vocab
from utilities.vocab import TanakaVocabs
from pytorch_lightning.loggers import TensorBoardLogger


# コマンドライン引数
batch_size: int = 100
epoch_size: int = 40
encoder_num_layers: int = 6
decoder_num_layers: int = 6
encoder_dropout: float = 0.2
decoder_dropout: float = 0.2

# グローバル変数
sentiment_type: str = "neutral"
vocabs: TanakaVocabs
all_datasets: List[TwitterDataset]
all_dataloader: List[TanakaDataLoader]

# 定数
dataset_train_pkl = "dataloader/twitter_dataset_neutral_train.model"
dataset_val_pkl = "dataloader/twitter_dataset_neutral_val.model"
dataset_test_pkl = "dataloader/twitter_dataset_neutral_test.model"
pkl_list = [
    dataset_train_pkl,
    dataset_val_pkl,
    dataset_test_pkl,
]


def _setting_field(args):
    global batch_size
    global epoch_size
    global encoder_num_layers
    global encoder_dropout
    global decoder_num_layers
    global decoder_dropout

    batch_size = args.batch_size
    epoch_size = args.epoch_size
    encoder_num_layers = args.encoder_num_layers
    encoder_dropout = args.encoder_dropout
    decoder_num_layers = args.decoder_num_layers
    decoder_dropout = args.decoder_dropout


def training_run(args):
    global vocabs
    global all_datasets
    global all_dataloader

    # --------------------------------------------------
    # コマンドライン引数をグローバル変数にセット
    # --------------------------------------------------
    _setting_field(args)

    # --------------------------------------------------
    # DataLoaderとVocabsを作成
    # --------------------------------------------------
    all_datasets = get_datasets(sentiment_type, pkl_list)
    vocabs = get_vocab(all_datasets, pkl_list)
    all_dataloader = get_dataloader(all_datasets, batch_size)

    # --------------------------------------------------
    # モデルの定義
    # --------------------------------------------------
    input_dim = len(vocabs.vocab_X.char2id)
    output_dim = len(vocabs.vocab_y.char2id)

    model = Seq2Seq(
        input_dim,
        output_dim,
        maxlen=60 + 8,
        encoder_num_layers=encoder_num_layers,
        encoder_dropout=encoder_dropout,
        decoder_num_layers=decoder_num_layers,
        decoder_dropout=decoder_dropout,
    )

    # --------------------------------------------------
    # コールバックの定義
    # --------------------------------------------------
    from pytorch_lightning.callbacks import EarlyStopping

    folderpath = "output/no_optuna/neutral"
    filename = "EN{}_DN{}_ED{}_DD{}".format(
        encoder_num_layers, decoder_num_layers, encoder_dropout, decoder_dropout
    )

    callbacks = [
        DisplayPredictDialogue(
            vocabs,
            TanakaDataLoader(all_datasets[0], batch_size=1, random_state=0),
            TanakaDataLoader(all_datasets[2], batch_size=1, random_state=0),
            translation_direction="jaja",
            folderpath=folderpath,
            filename=filename,
        ),
        SaveModelParams(folderpath=folderpath, filename=filename),
        EarlyStopping("val_loss", verbose=True, mode="min", patience=5),
    ]

    # --------------------------------------------------
    # モデルのトレーニング
    # --------------------------------------------------
    freeze_support()
    trainer = pl.Trainer(
        callbacks=callbacks,
        max_epochs=epoch_size,
        accelerator="gpu",
        devices=-1,
        logger=TensorBoardLogger(os.getcwd(), version="no_optuna/neutral"),
    )

    trainer.fit(model, train_dataloaders=all_dataloader[0], val_dataloaders=all_dataloader[1])
    trainer.test(model, all_dataloader[2])

    torch.save(model.state_dict(), os.path.join(folderpath, filename + ".pth"))
