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
    from utilities.vocab import TanakaVocabs

    vocabs = TanakaVocabs(
        datasets, top_words=50000, X_bos=True, X_eos=True, y_bos=True, y_eos=True
    )

    # --------------------------------------
    # DataLoaderの作成
    # --------------------------------------
    from dataloader.tanaka_dataloader import TanakaDataLoader

    train_dataloader = TanakaDataLoader(train_dataset, batch_size=BATCH_SIZE)
    dev_dataloader = TanakaDataLoader(dev_dataset, batch_size=BATCH_SIZE)
    test_dataloader = TanakaDataLoader(test_dataset, batch_size=1, random_state=0)

    # --------------------------------------
    # Modelの作成
    # --------------------------------------
    from models.seq2seq_transformer import Seq2Seq

    input_dim = len(vocabs.vocab_X.char2id)
    output_dim = len(vocabs.vocab_y.char2id)

    model = Seq2Seq(input_dim, output_dim)

    # --------------------------------------
    # Modelの適合
    # --------------------------------------
    import pytorch_lightning as pl
    from pytorch_lightning.strategies.ddp import DDPStrategy
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
        max_epochs=EPOCH_SIZE,
        accelerator="gpu",
        devices=2,
        plugins=DDPStrategy(find_unused_parameters=False),
    )
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=dev_dataloader)
    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    main()
