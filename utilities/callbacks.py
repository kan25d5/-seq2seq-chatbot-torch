import os
import pytorch_lightning as pl
from torch import Tensor
from pytorch_lightning.callbacks import Callback, EarlyStopping
import torch
from dataloader.tanaka_dataloader import TanakaDataLoader
from utilities.decode import greedy_decode
from utilities.vocab import TanakaVocabs


class DisplayPredictDialogue(Callback):
    """ Validationで予測した応答をコンソールに出力する """

    def __init__(
        self,
        vocabs: TanakaVocabs,
        train_dataloader: TanakaDataLoader,
        test_dataloader: TanakaDataLoader,
        translation_direction="enja",
        folderpath="",
        filename="",
    ) -> None:
        super().__init__()
        self.vocabs = vocabs
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.translation_direction = translation_direction
        self.folderpath = folderpath
        self.filename = filename

        self._current_epoch = -1

    def _predict_dataloader(self, model, current_epoch, dataloader, dataloader_type, f):
        f.write("dataloader_type : {}".format(dataloader_type))
        f.write("current epoch : {}\n".format(current_epoch))

        for idx, (src, tgt) in enumerate(dataloader):
            source = src.view(-1).tolist()
            preds = greedy_decode(model.model, src).tolist()
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

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module):
        if self._current_epoch == trainer.current_epoch:
            return
        else:
            self._current_epoch = trainer.current_epoch

        current_epoch = trainer.current_epoch
        filepath = os.path.join(self.folderpath, self.filename)
        filepath = filepath + "E{}.txt".format(current_epoch)

        with open(filepath, "a") as f:
            self._predict_dataloader(pl_module, current_epoch, self.test_dataloader, "test", f)
        with open(filepath, "a") as f:
            self._predict_dataloader(pl_module, current_epoch, self.train_dataloader, "train", f)


class SaveModelParams(Callback):
    """ 特定のエポック数ごとにモデルのパラメータを保存する """

    def __init__(self, folderpath="", filename="") -> None:
        super().__init__()

        self.folderpath = folderpath
        self.filename = filename
        self._current_epoch = -1

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self._current_epoch == trainer.current_epoch:
            return
        else:
            self._current_epoch = trainer.current_epoch

        current_epoch = trainer.current_epoch
        folderpath = os.path.join(self.folderpath, self.filename)
        folderpath = folderpath + "E{}.pth".format(current_epoch)

        torch.save(pl_module.state_dict(), folderpath)

