import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import ReconDataset
import numpy as np
import matplotlib.pyplot as plt
from encoder import Encoder
import einops
from run import Config


class Pretrain(pl.LightningModule):

    def __init__(self):
        super(Pretrain, self).__init__()
        configs = Config()
        args = configs.parser.parse_args()
        self.loss = torch.nn.CrossEntropyLoss()
        self.save_hyperparameters()
        self.model = Encoder(args)
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)


    def forward(self, x):
        x = einops.rearrange(x, "b c h w -> b 1 c h w")
        loss, x = self.model(x)
        return loss 


    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()
        if self.trainer.is_last_batch:
            sch.step(self.trainer.current_epoch)

        loss = self(batch)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log_dict(
            {
                "train/loss": loss,
            },
            prog_bar=True,
        )


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), 
                lr=1.6e-3, betas=(0.9, 0.95), weight_decay=0.05)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=120)
        return [optimizer], [scheduler]


    def train_dataloader(self):
        year = range(2020, 2024)
        dataset = []
        for y in year:
            dataset.append(ReconDataset(year=y))
        datasets = torch.utils.data.ConcatDataset(dataset)
        train_loader = DataLoader(datasets, batch_size=64, num_workers=8, shuffle=True)
        return train_loader
