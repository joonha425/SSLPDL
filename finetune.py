import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataset import ProbDataset
from decoder import UPerNet
import einops
from pretrain import Pretrain


class Finetune(pl.LightningModule):

    def __init__(
        self,
        out_channels: int = 3,
        channels: int = 64,
        pretrained_path: str = '',
        beta: float = 0.75,
        weight: list = [1., 5., 10.],
        **kwargs,
    ):
        super(Finetune, self).__init__()
        self.beta = beta
        self.save_hyperparameters()
        self.pretrained_model = Pretrain.load_from_checkpoint(pretrained_path)
        self.generator = UPerNet(
            num_class=out_channels, fc_dim=channels * 8, use_softmax=True, fpn_dim=256) 
        self.automatic_optimization = False
        torch.autograd.set_detect_anomaly(True)
        self.one_hot_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight))
        self.prob_den_loss = torch.nn.CrossEntropyLoss(weight=torch.tensor(weight))


    def forward(self, x, mask_ratio=.25):
        x = einops.rearrange(x, "b c h w -> b 1 c h w")
        x1, _ = self.pretrained_model.model.forward_features_seq_out(x, mask_ratio=mask_ratio)
        x = self.generator(x1, segSize=x[0].shape[1:])
        return x


    def training_step(self, batch, batch_idx):
        images, qpe1, qpe2 = batch
        opt = self.optimizers()

        predictions = self(images)
        p1loss = self.one_hot_loss(predictions, qpe1.long())
        p2loss = self.prob_den_loss(predictions, qpe2)
        loss = p1loss * (self.beta) + p2loss * (1-self.beta)
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()

        self.log_dict(
            {
                "train/p1_loss": p1loss,
                "train/p2_loss": p2loss,
                "train/loss": loss,
            },
            prog_bar=True,
        )


    def configure_optimizers(self):
        optimizer = torch.optim.Adam([
            {'params': self.pretrained_model.parameters(), 'lr': 5e-5},
            {'params': self.generator.parameters()},], lr=1e-4)

        return [optimizer], []


    def train_dataloader(self):
        year = range(2020, 2024)
        dataset = []
        for y in year:
            dataset.append(ProbDataset(year=y))
        datasets = torch.utils.data.ConcatDataset(dataset)
        train_loader = DataLoader(datasets, batch_size=64, num_workers=8, shuffle=True)
        return train_loader

