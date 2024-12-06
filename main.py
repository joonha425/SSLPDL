from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pretrain import Pretrain
from finetune import Finetune


def pretrain():
    callback = ModelCheckpoint(
        dirpath = 'pre_path/', 
        monitor = 'train/loss', 
        mode = 'min', 
        save_top_k = 1,
        save_last = True, 
    )

    trainer = pl.Trainer(
        gpus=1,
        callbacks=callback,
        max_epochs=1500,
        val_check_interval=1.0,
    )
    model = Pretrain()
    trainer.fit(model)


def fineTune(pretrained_path):
    mycallback = ModelCheckpoint(
        dirpath = 'path/',
        monitor = 'train/loss',
        mode = 'min',
        save_top_k = 1,
        save_last = True
    )
    trainer = pl.Trainer(
        gpus=1,
        callbacks=[mycallback],
        max_epochs=200,
        val_check_interval=1.0,
    )
    model = Finetune(pretrained_path=pretrained_path)
    trainer.fit(model)


if __name__ == "__main__":
    pretrain()
