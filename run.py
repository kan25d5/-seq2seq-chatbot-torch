from dataloader.tanaka_dataset import TanakaDataset
from utilities.vocab import TanakaVocabs


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


if __name__ == "__main__":
    main()
