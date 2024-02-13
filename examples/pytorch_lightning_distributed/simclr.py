import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.data import LightlyDataset
from argparse import ArgumentParser
from lightning.pytorch.loggers import MLFlowLogger
import mlflow
import logging

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
        self.logger.experiment.log_metric(run_id=self.logger.run_id, key="train_loss_ssl", value=loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim

def main(args):
    model = SimCLR()

    transform = SimCLRTransform(input_size=args.input_size, vf_prob=0.5, rr_prob=0.5, min_scale=0.08, normalize={'mean':IMAGENET_NORMALIZE['mean'], 'std':IMAGENET_NORMALIZE['std']}) # min_scale default is 0.08
    test_transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.input_size, args.input_size)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=IMAGENET_NORMALIZE["mean"],
                std=IMAGENET_NORMALIZE['std'],
            ),      
        ]
    )

    dataset_train_simclr = LightlyDataset(input_dir=args.path_to_data_train, transform=transform)

    dataset_test = LightlyDataset(input_dir=args.path_to_data_test, transform=test_transform)

    dataloader_train_simclr = torch.utils.data.DataLoader(
        dataset_train_simclr,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    dataloader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
    )

    # accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    # Train with DDP and use Synchronized Batch Norm for a more accurate batch norm
    # calculation. Distributed sampling is also enabled with replace_sampler_ddp=True.

    mlf_logger = MLFlowLogger(experiment_name="ssl/experiments", tracking_uri="https://mlflow-ml.visiolab.io/")
    mlflow.log_metric("test", 1, run_id=mlf_logger.run_id)
    trainer = pl.Trainer(
        logger=mlf_logger,
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        strategy=args.strategy,
        sync_batchnorm=args.sync_batchnorm,
        use_distributed_sampler=args.use_distributed_sampler,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
        num_nodes=args.num_nodes,
    )

    trainer.fit(model=model, train_dataloaders=dataloader_train_simclr)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path-to-data-train', default='/data/train', type=str)
    parser.add_argument('--path-to-data-test', default='/data/test', type=str) 
    parser.add_argument('--input-size', default=128, type=int)
    parser.add_argument('--batch-size', default=3, type=int)
    parser.add_argument('--num-workers', default=0, type=int)
    parser.add_argument('--devices', default=1, type=int)
    parser.add_argument('--max-epochs', default=5, type=int)
    parser.add_argument('--strategy', default='ddp', type=str)
    parser.add_argument('--accelerator', default='gpu', type=str)
    parser.add_argument('--num-nodes', default=1, type=int)
    parser.add_argument('--sync-batchnorm', default=False, type=bool)
    parser.add_argument('--use-distributed-sampler', default=False, type=bool)
    args = parser.parse_args()
    main(args)