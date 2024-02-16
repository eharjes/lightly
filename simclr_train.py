import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
# from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.sim_clr_transform import SimCLRTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.data import LightlyDataset
from argparse import ArgumentParser
from pytorch_lightning.loggers import MLFlowLogger
import mlflow
import logging
import os
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision.utils import make_grid

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
        self.log("train_loss_ssl", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # self.logger.experiment.log_metric(run_id=self.logger.run_id, key="train_loss_ssl", value=loss)
        return loss

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
    
def generate_embeddings(model, dataloader):
    """Generates representations for all images in the dataloader with
    the given model
    """

    embeddings = []
    filenames = []
    with torch.no_grad():
        for img, _, fnames in dataloader:
            img = img.to(model.device)
            emb = model.backbone(img).flatten(start_dim=1)
            embeddings.append(emb)
            filenames.extend(fnames)

    embeddings = torch.cat(embeddings, 0)
    embeddings = normalize(embeddings)
    return embeddings, filenames

def get_image_as_tensor(filename: str):
    """Load an image file and convert it to a PyTorch tensor."""
    image = Image.open(filename).convert('RGB')  # Convert to RGB to ensure 3 color channels
    transform = torchvision.transforms.ToTensor()  # Convert image to a tensor
    return transform(image)

def plot_knn_examples(embeddings, filenames, n_neighbors=2, num_examples=10, save_dir='/Users/eliasharjes/Documents/uni/master_thesis/ssl/plot_test'):
    path_to_data = '/Users/eliasharjes/Documents/uni/master_thesis/ssl/data/test'
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)
    num_examples = 2
    samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

    images_tensors = []
    for idx in samples_idx:
        # Include the sample itself as the first image
        sample_and_neighbors_indices = indices[idx]
        for neighbor_idx in sample_and_neighbors_indices:
            fname = os.path.join(path_to_data, filenames[neighbor_idx])
            image_tensor = get_image_as_tensor(fname)
            images_tensors.append(image_tensor)

    # Create a grid of images
    images_grid = make_grid(images_tensors, nrow=n_neighbors + 1)
    plt.figure(figsize=(20, 10))
    plt.imshow(images_grid.permute(1, 2, 0))
    plt.axis('off')

    # Save the complete grid
    os.makedirs('/Users/eliasharjes/Documents/uni/master_thesis/ssl/plot_test', exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'knn_examples_grid2.png'), bbox_inches='tight', dpi=100)
    plt.close()

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
    with mlflow.start_run() as run:
        mlf_logger = MLFlowLogger(run_id=run.info.run_id, experiment_name="ssl/experiments", tracking_uri="https://mlflow-ml.visiolab.io/", log_model=False)
        mlflow.log_metric("test", 1)
        mlflow.log_params(transform.params)
        mlflow.log_params(vars(args))

    checkpoint_callback_top_k = ModelCheckpoint(
        dirpath='checkpoints',
        every_n_epochs=1,  
        save_top_k=3,
        monitor="train_loss_ssl",
        mode="min",
        save_last=True,   
    )
        
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback_top_k],
        logger=mlf_logger,
        max_epochs=args.max_epochs,
        devices=args.devices,
        accelerator=args.accelerator,
        strategy=args.strategy,
        sync_batchnorm=args.sync_batchnorm,
        use_distributed_sampler=args.use_distributed_sampler,  # or replace_sampler_ddp=True for PyTorch Lightning <2.0
        num_nodes=args.num_nodes,
        log_every_n_steps=5,
    )

    trainer.fit(model=model, train_dataloaders=dataloader_train_simclr)

    # Plotting
    plot_embeddings = True
    if plot_embeddings:
        model.eval()
        embeddings, filenames = generate_embeddings(model, dataloader_test)
        plot_knn_examples(embeddings, filenames, num_examples=30)

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