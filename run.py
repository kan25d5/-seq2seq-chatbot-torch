# --------------------------------------
# 定数
# DATA_SIZE : 使用するコーパスファイルの割合
# TRAIN_SIZE : コーパスファイルのうち，トレーニングで利用する割合
# VAL_SIZE : トレーニングで利用しないコーパスファイルのうち，検証で利用する割合．
#            残り，つまり1-VAL_SIZEは，テストデータで利用する．
# TOP_WORDS : 出現頻度上位TOP_WORDSの語彙を使って学習．他はUNKトークンとして配置．
# BATCH_SIZE : バッチサイズ
# EPOCH_SIZE : 最大エポックサイズ
# --------------------------------------
DATA_SIZE = 1
TRAIN_SIZE = 0.8
VAL_SIZE = 0.7
TOP_WORDS = 80000

BATCH_SIZE = 100
EPOCH_SIZE = 100


def main():
    import os

    # Reference:
    # https://stackoverflow.com/questions/30791550/limit-number-of-threads-in-numpy/31622299#31622299
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"

    # --------------------------------------
    # Datasetの作成
    # --------------------------------------
    from dataloader.twitter_dataset import TwitterDataset
    from utilities.functions import train_val_test

    train_files, val_files, test_files = train_val_test(
        all_size=DATA_SIZE, train_size=TRAIN_SIZE, val_size=VAL_SIZE
    )

    train_dataset = TwitterDataset()
    val_dataset = TwitterDataset()
    test_dataset = TwitterDataset()

    dataset_train_pkl = "dataloader/twitter_dataset_S{}_train.model".format(DATA_SIZE)
    dataset_val_pkl = "dataloader/twitter_dataset_S{}_val.model".format(DATA_SIZE)
    dataset_test_pkl = "dataloader/twitter_dataset_S{}_test.model".format(DATA_SIZE)

    if not os.path.exists(dataset_train_pkl):
        train_dataset.load_corpus(train_files)
    else:
        train_dataset.load_corpus_pkl(dataset_train_pkl)

    if not os.path.exists(dataset_val_pkl):
        val_dataset.load_corpus(val_files)
    else:
        val_dataset.load_corpus_pkl(dataset_val_pkl)

    if not os.path.exists(dataset_test_pkl):
        test_dataset.load_corpus(test_files)
    else:
        test_dataset.load_corpus_pkl(dataset_test_pkl)

    # --------------------------------------
    # Vocabの作成
    # --------------------------------------
    from utilities.vocab import TanakaVocabs

    vocabs = TanakaVocabs(
        [train_dataset, val_dataset, test_dataset],
        top_words=50000,
        X_bos=True,
        X_eos=True,
        y_bos=True,
        y_eos=True,
    )

    train_dataset.save_corpus_pkl(dataset_train_pkl)
    val_dataset.save_corpus_pkl(dataset_val_pkl)
    test_dataset.save_corpus_pkl(dataset_test_pkl)

    # --------------------------------------
    # DataLoaderの作成
    # --------------------------------------
    from dataloader.tanaka_dataloader import TanakaDataLoader

    train_dataloader = TanakaDataLoader(train_dataset, batch_size=BATCH_SIZE)
    val_dataloader = TanakaDataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_dataloader = TanakaDataLoader(test_dataset, batch_size=1, random_state=0)

    # --------------------------------------
    # Modelの作成
    # --------------------------------------
    from models.seq2seq_transformer import Seq2Seq

    input_dim = len(vocabs.vocab_X.char2id)
    output_dim = len(vocabs.vocab_y.char2id)

    model = Seq2Seq(input_dim, output_dim, maxlen=60 + 8)

    # --------------------------------------
    # Modelの適合
    # --------------------------------------
    import pytorch_lightning as pl

    # from pytorch_lightning.strategies.ddp import DDPStrategy
    from multiprocessing import freeze_support
    from utilities.callbacks import DisplayPredictDialogue

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
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
