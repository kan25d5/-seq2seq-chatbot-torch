# --------------------------------------
# 定数
# ---------------------------------------
# DATA_SIZE : 使用するコーパスファイルの割合
# TRAIN_SIZE : コーパスファイルのうち，トレーニングで利用する割合
# VAL_SIZE : トレーニングで利用しないコーパスファイルのうち，検証で利用する割合．
#            残り，つまり1-VAL_SIZEは，テストデータで利用する．
# TOP_WORDS : 出現頻度上位TOP_WORDSの語彙を使って学習．他はUNKトークンとして配置．
# BATCH_SIZE : バッチサイズ
# EPOCH_SIZE : 最大エポックサイズ
import os
import random
from typing import List, Tuple
from dataloader.twitter_dataset import TwitterDataset


DATA_SIZE = 1
TRAIN_SIZE = 0.9
VAL_SIZE = 0.7
TOP_WORDS = 80000

BATCH_SIZE = 100
EPOCH_SIZE = 100
MAXLEN = 60

POS_CORPUS_FILE = "output/pos.json"
NEG_CORPUS_FILE = "output/neg.json"

USE_CORPUS = POS_CORPUS_FILE


def split_train_val_test(filepath, train_size=0.7, val_size=0.7):
    from utilities.functions import load_json

    dialogues = load_json(filepath)
    all_size = len(dialogues)

    train_dialogue = dialogues[0 : int(all_size * train_size)]
    train_other = dialogues[int(all_size * train_size) :]
    val_dialogue = train_other[0 : int(len(train_other) * val_size)]
    test_dialogue = train_other[int(len(train_other) * val_size) :]

    print("train_dialogue : {} turns".format(len(train_dialogue)))
    print("val_dialogue : {} turns".format(len(val_dialogue)))
    print("test_dialogue : {} turns".format(len(test_dialogue)))

    return train_dialogue, val_dialogue, test_dialogue


def get_datasets(all_dialogues: List[list]) -> List[TwitterDataset]:
    def get_dataset_using_dialogue(dialogue: List[Tuple]):

        dataset = TwitterDataset()
        dataset.messages = [d[0] for d in dialogue]
        dataset.responses = [d[1] for d in dialogue]
        return dataset

    train_dialogue = all_dialogues[0]
    val_dialogue = all_dialogues[1]
    test_dialogue = all_dialogues[2]

    train_dataset = get_dataset_using_dialogue(train_dialogue)
    val_dataset = get_dataset_using_dialogue(val_dialogue)
    test_dataset = get_dataset_using_dialogue(test_dialogue)

    return train_dataset, val_dataset, test_dataset


def get_vocab_and_transform(all_datasets: List[TwitterDataset]):
    from utilities.vocab import TanakaVocabs
    from utilities.constant import CHAR2ID

    vocabs = TanakaVocabs(TOP_WORDS)
    vocabs.load_char2id_pkl(CHAR2ID)

    for dataset in all_datasets:
        dataset.messages = vocabs.X_transform(dataset.messages)
        dataset.responses = vocabs.y_transform(dataset.responses)

    return all_datasets, vocabs


def main():
    # -------------------------------------------------
    # CORPUS_FILEからtrain, val, test文のターンを分割する
    # train : List(message, response)
    # -------------------------------------------------
    train_dialogue, val_dialogue, test_dialogue = split_train_val_test(USE_CORPUS)

    # -------------------------------------------------
    # 分割したターンリストからTwitterDatasetを作成する
    # -------------------------------------------------
    all_dialogues = [train_dialogue, val_dialogue, test_dialogue]
    train_dataset, val_dataset, test_dataset = get_datasets(all_dialogues)

    # -------------------------------------------------
    # Vocabを使って単語をベクトル化する(one-hot)
    # -------------------------------------------------
    all_datasets = [train_dataset, val_dataset, test_dataset]
    all_datasets, vocabs = get_vocab_and_transform(all_datasets)

    # -------------------------------------------------
    # DataLoaderの作成
    # -------------------------------------------------
    from dataloader.tanaka_dataloader import TanakaDataLoader

    train_dataloader = TanakaDataLoader(all_datasets[0], batch_size=BATCH_SIZE)
    val_dataloader = TanakaDataLoader(all_datasets[1], batch_size=BATCH_SIZE)
    test_dataloader = TanakaDataLoader(all_datasets[2], batch_size=1, random_state=0)

    # -------------------------------------------------
    # モデルの作成
    # -------------------------------------------------
    from models.seq2seq_transformer import Seq2Seq

    src_vocab_size = len(vocabs.vocab_X.char2id)
    tgt_vocab_size = len(vocabs.vocab_y.char2id)
    model = Seq2Seq(src_vocab_size, tgt_vocab_size, maxlen=MAXLEN)

    # -------------------------------------------------
    # トレーニング
    # -------------------------------------------------
    import torch
    import pytorch_lightning as pl

    # from pytorch_lightning.strategies.ddp import DDPStrategy
    from pytorch_lightning.loggers import TensorBoardLogger
    from multiprocessing import freeze_support
    from utilities.callbacks import DisplayPredictDialogue
    from utilities.constant import SAVE_MODELS_PTH

    model.load_state_dict(torch.load("output/E38_model.pth"))

    freeze_support()
    trainer = pl.Trainer(
        callbacks=[
            DisplayPredictDialogue(
                vocabs,
                TanakaDataLoader(train_dataset, batch_size=1, random_state=0),
                TanakaDataLoader(test_dataset, batch_size=1, random_state=0),
            )
        ],
        # strategy=DDPStrategy(find_unused_parameters=False)
        max_epochs=EPOCH_SIZE,
        accelerator="gpu",
        devices=2,
        logger=TensorBoardLogger(
            os.getcwd(),
            version="B{}_E{}_S{}_pos".format(BATCH_SIZE, EPOCH_SIZE, DATA_SIZE),
        ),
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader
    )
    trainer.test(model, test_dataloader)

    torch.save(model.state_dict(), "output/pos_model.pth")


if __name__ == "__main__":
    main()
