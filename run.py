BATCH_SIZE = 100
EPOCH_SIZE = 100


def main():
    # --------------------------------------
    # Datasetの作成
    # --------------------------------------
    from dataloader.tanaka_dataset import TanakaDataset

    train_dataset = TanakaDataset(corpus_type="train")
    dev_dataset = TanakaDataset(corpus_type="dev")
    test_dataset = TanakaDataset(corpus_type="test")
    datasets = [train_dataset, dev_dataset, test_dataset]

    # --------------------------------------
    # Vocabの作成
    # --------------------------------------
    from gensim.models import KeyedVectors
    from utilities.vocab_w2v import TanakaVocabs

    wv_filepath = "w2v/top_50000.model"
    wv = KeyedVectors.load_word2vec_format(wv_filepath, binary=True)
    vocabs = TanakaVocabs(datasets, wv, X_eos=True, y_bos=True, y_eos=True)

    # --------------------------------------
    # DataLoaderの作成
    # --------------------------------------
    from dataloader.tanaka_dataloader import TanakaDataLoader

    train_dataloader = TanakaDataLoader(train_dataset, batch_size=BATCH_SIZE)
    dev_dataloader = TanakaDataLoader(dev_dataset, batch_size=BATCH_SIZE)
    test_dataloader = TanakaDataLoader(test_dataset, batch_size=1, random_state=0)

    for x, t in train_dataloader:
        print(x)
        print(t)
        break
    train_dataloader._reset()

    # --------------------------------------
    # Modelの作成
    # --------------------------------------
    from models.seq2seq import Seq2Seq

    input_dim = len(vocabs.vocab_X.char2id)
    hidden_dim = 256
    output_dim = len(vocabs.vocab_y.wv.key_to_index)

    model = Seq2Seq(input_dim, hidden_dim, output_dim)

    # --------------------------------------
    # Modelの適合
    # --------------------------------------
    import pytorch_lightning as pl
    from pytorch_lightning.strategies.ddp import DDPStrategy
    from multiprocessing import freeze_support
    from utilities.callbacks import DisplayPredictDialogue

    freeze_support()
    trainer = pl.Trainer(
        callbacks=[DisplayPredictDialogue(vocabs, test_dataloader)],
        max_epochs=EPOCH_SIZE,
        accelerator="gpu",
        devices=2,
        plugins=DDPStrategy(find_unused_parameters=False),
    )
    trainer.fit(
        model, train_dataloaders=train_dataloader, val_dataloaders=dev_dataloader
    )
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
