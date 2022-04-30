import pytorch_lightning as pl
from dataloader.tanaka_dataloader import TanakaDataLoader
from dataloader.tanaka_dataset import TanakaDataset
from models.seq2seq import Seq2Seq
from utilities.callbacks import DisplayPredictDialogue
from utilities.vocab import TanakaVocabs
from pytorch_lightning.strategies.ddp import DDPStrategy
from multiprocessing import freeze_support

BATCH_SIZE = 100
EPOCH_SIZE = 100


def main():
    train_dataset = TanakaDataset(corpus_type="train")
    dev_dataset = TanakaDataset(corpus_type="dev")
    test_dataset = TanakaDataset(corpus_type="test")

    vocab = TanakaVocabs(
        [train_dataset, dev_dataset, test_dataset], X_eos=True, y_bos=True, y_eos=True
    )

    x, t = train_dataset[0]
    print(x)
    print(t)

    print(vocab.X_decode(x))
    print(vocab.y_decode(t))

    train_dataloader = TanakaDataLoader(train_dataset, BATCH_SIZE)
    dev_dataloader = TanakaDataLoader(dev_dataset, BATCH_SIZE)
    test_dataloader = TanakaDataLoader(test_dataset, 1, random_state=0)

    for x, t in train_dataloader:
        print(x)
        print(t)
        break
    train_dataloader._reset()

    input_dim = len(vocab.vocab_X.char2id)
    hidden_dim = 256
    output_dim = len(vocab.vocab_y.char2id)
    model = Seq2Seq(input_dim, hidden_dim, output_dim)

    freeze_support()
    trainer = pl.Trainer(
        callbacks=[DisplayPredictDialogue(vocab, test_dataloader)],
        max_epochs=EPOCH_SIZE,
        accelerator="gpu",
        devices=2,
        plugins=DDPStrategy(find_unused_parameters=False),
    )

    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=dev_dataloader)


if __name__ == "__main__":
    main()
