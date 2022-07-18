import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torch import Tensor
from dataloader.tanaka_dataloader import TanakaDataLoader
from utilities.vocab import TanakaVocabs
from models.seq2seq_transformer import Seq2Seq
from utilities.decode import greedy_decode


class DisplayPredictDialogue(Callback):
    def __init__(
        self,
        vocabs: TanakaVocabs,
        train_dataloader: TanakaDataLoader,
        test_dataloader: TanakaDataLoader,
        translation_direction="enja",
        filename="output"
    ) -> None:
        if translation_direction not in ["enja", "jaen"]:
            raise ValueError("translation_directionはenja, jaenのいずれかの値指定")

        super().__init__()
        self.vocabs = vocabs
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.translation_direction = translation_direction
        self.filename = filename

        self._current_epoch = -1

    def _predict_dataloader(self, model: Tensor, current_epoch, dataloader, dataloader_type, f):
        f.write("dataloader_type : {}".format(dataloader_type))
        f.write("current epoch : {}\n".format(current_epoch))

        for idx, (src, tgt) in enumerate(dataloader):
            source = src.view(-1).tolist()
            preds = greedy_decode(model, src).tolist()
            target = tgt.view(-1).tolist()

            source = " ".join(self.vocabs.vocab_X.decode(source))
            target = "".join(self.vocabs.vocab_y.decode(target))
            preds = "".join(self.vocabs.vocab_y.decode(preds))

            print("source : {}".format(source))
            print("target : {}".format(target))
            print("preds : {}".format(preds))

            f.write("source : {}\n".format(source))
            f.write("target : {}\n".format(target))
            f.write("preds : {}\n".format(preds))
            f.write("--------------------------\n")

            if idx > 9:
                break

        f.write("\n")
        f.write("=================================\n")
        f.write("\n")

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: Seq2Seq):
        if self._current_epoch == trainer.current_epoch:
            return
        else:
            self._current_epoch = trainer.current_epoch

        current_epoch = trainer.current_epoch
        with open("output/{}.txt".format(self.filename), "a") as f:
            self._predict_dataloader(pl_module, current_epoch, self.test_dataloader, "test", f)
            self._predict_dataloader(pl_module, current_epoch, self.train_dataloader, "train", f)

