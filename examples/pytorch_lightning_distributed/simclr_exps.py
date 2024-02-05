import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.data import LightlyDataset
import click


@click.command()
@click.option('--path_to_data', default='/data/train', type=str)
@click.option('--input_size', default=128, type=int)
@click.option('--batch_size', default=3, type=int)
@click.option('--num_workers', default=0, type=int)
@click.option('--devices', default=1, type=int)
@click.option('--max_epochs', default=10, type=int)
@click.option('--strategy', default='ddp', type=str)
@click.option('--accelerator', default='cpu', type=str)
def main(path_to_data, input_size, batch_size, num_workers, devices, max_epochs, strategy, accelerator):
    class SimCLR(pl.LightningModule):
        def __init__(self):
            super().__init__()
            regnet = torchvision.models.regnet_x_400mf()
            hidden_dim = regnet.fc.in_features
            self.backbone = nn.Sequential(*list(regnet.children())[:-1])
            self.projection_head = SimCLRProjectionHead(hidden_dim, 2048, 2048)

            # enable gather_distributed to gather features from all gpus
            # before calculating the loss
            self.criterion = NTXentLoss(gather_distributed=True)

        def forward(self, x):
            x = self.backbone(x).flatten(start_dim=1)
            z = self.projection_head(x)
            return z

        def training_step(self, batch, batch_index):
            (x0, x1) = batch[0]
            z0 = self.forward(x0)
            z1 = self.forward(x1)
            loss = self.criterion(z0, z1)
            self.log("train_loss_ssl", loss)
            return loss

        def configure_optimizers(self):
            optim = torch.optim.SGD(self.parameters(), lr=0.06)
            return optim


    model = SimCLR()
    path_to_data = '/data/train' # '/Users/eliasharjes/Documents/uni/master_thesis/ssl/data/train' 
    input_size = 128
    batch_size = 3 # 256
    num_workers = 0
    devices = 1
    max_epochs = 10
    strategy = "ddp"
    accelerator="cpu"

    transform = SimCLRTransform(input_size=input_size, vf_prob=0.5, rr_prob=0.5, min_scale=0.08, normalize={'mean':IMAGENET_NORMALIZE['mean'], 'std':IMAGENET_NORMALIZE['std']}) # min_scale default is 0.08

    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((input_size, input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE['std'],
            ),
        ]
    )

    dataset_train_simclr = LightlyDataset(input_dir=path_to_data, transform=transform)

    dataset_test = LightlyDataset(input_dir=path_to_data, transform=test_transform)

    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Train with DDP and use Synchronized Batch Norm for a more accurate batch norm
    # calculation. Distributed sampling is also enabled with replace_sampler_ddp=True.
    # trainer = pl.Trainer(
    #     max_epochs=10,
    #     devices="auto",
    #     accelerator=accelerator,
    #     # strategy="ddp",
    #     sync_batchnorm=True,
    #     use_distributed_sampler=True,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
    # )
    trainer = pl.Trainer(max_epochs=max_epochs, strategy=strategy, accelerator=accelerator, devices=devices)
    trainer.fit(model=model, train_dataloaders=dataloader_train_simclr)

if __name__ == "__main__":
    main()
