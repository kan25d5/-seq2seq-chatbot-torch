import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from dataloader.tanaka_dataloader import TanakaDataLoader
from utilities.vocab import TanakaVocabs


class DisplayPredictDialogue(Callback):
    def __init__(
        self,
        vocabs: TanakaVocabs,
        test_dataloader: TanakaDataLoader,
        translation_direction="enja",
    ) -> None:
        if translation_direction not in ["enja", "jaen"]:
            raise ValueError("corpus_typeはenja, jaenのいずれかの値指定")

        super().__init__()
        self.vocabs = vocabs
        self.test_dataloader = test_dataloader
        self.translation_direction = translation_direction

        self._current_epoch = -1

    def _decode_stop_eos(self, sentence, is_join_space=False):
        output = []
        for w in sentence:
            if w == "</s>":
                break
            elif w == "<s>":
                continue
            output.append(w)

        if is_join_space:
            return " ".join(output)
        else:
            return "".join(output)

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        if self._current_epoch == trainer.current_epoch:
            return
        else:
            self._current_epoch = trainer.current_epoch

        for idx, (x, t) in enumerate(self.test_dataloader):
            preds = pl_module.forward(x)

            source = x.contiguous().view(-1).tolist()
            target = t.contiguous().view(-1).tolist()
            predict = preds.max(dim=-1)[1].contiguous().view(-1).tolist()

            if self.translation_direction == "enja":
                target = self._decode_stop_eos(self.vocabs.y_decode(target))
                predict = self._decode_stop_eos(self.vocabs.y_decode(predict))
                source = self._decode_stop_eos(self.vocabs.X_decode(source), is_join_space=True)
            else:

                source = self._decode_stop_eos(self.vocabs.X_decode(source))
                target = self._decode_stop_eos(self.vocabs.y_decode(target), is_join_space=True)
                predict = self._decode_stop_eos(self.vocabs.y_decode(predict), is_join_space=True)

            print()
            print("source => " + source)
            print("target => " + target)
            print("predict => " + predict)

            if idx >= 9:
                self.test_dataloader._reset()
                break
