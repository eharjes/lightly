# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import pytorch_lightning as pl
import torch
import torchvision
from torch import nn

from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.data import LightlyDataset
# from sklearn.neighbors import NearestNeighbors
# from sklearn.preprocessing import normalize
# from PIL import Image
# import numpy as np
# import matplotlib.pyplot as plt
# import os
# from torchvision.utils import make_grid


class SimCLR(pl.LightningModule):
    def __init__(self):
        super().__init__()
        regnet = torchvision.models.regnet_x_400mf()
        self.backbone = nn.Sequential(*list(regnet.children())[:-1])
        hidden_dim = regnet.fc.in_features
        self.projection_head = SimCLRProjectionHead(hidden_dim, 2048, 2048)
        self.criterion = NTXentLoss()

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

    ## Version like in notebook
    # def configure_optimizers(self):
    #     optim = torch.optim.SGD(
    #         self.parameters(), lr=6e-2, momentum=0.9, weight_decay=5e-4
    #     )
    #     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, 5)
    #     return [optim], [scheduler]
    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr=0.06)
        return optim
    
model = SimCLR()
path_to_data = '/Users/eliasharjes/Documents/uni/master_thesis/ssl/data/train' # '/data/train' 
input_size = 128
batch_size = 2 # 256
num_workers = 0

transform = SimCLRTransform(input_size=input_size, vf_prob=0.5, rr_prob=0.5, min_scale=0.08, normalize={'mean':IMAGENET_NORMALIZE['mean'], 'std':IMAGENET_NORMALIZE['std']}) # min_scale default is 0.08

# We create a torchvision transformation for embedding the dataset after
# training
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

trainer = pl.Trainer(max_epochs=25, devices=2, accelerator=accelerator, log_every_n_steps=1)
trainer.fit(model=model, train_dataloaders=dataloader_train_simclr)

## Plotting
# plot_embeddings = True
# if plot_embeddings:
#     model.eval()
#     embeddings, filenames = generate_embeddings(model, dataloader_test)
#     plot_knn_examples(embeddings, filenames, num_examples=30)
    
# def generate_embeddings(model, dataloader):
#     """Generates representations for all images in the dataloader with
#     the given model
#     """

#     embeddings = []
#     filenames = []
#     with torch.no_grad():
#         for img, _, fnames in dataloader:
#             img = img.to(model.device)
#             emb = model.backbone(img).flatten(start_dim=1)
#             embeddings.append(emb)
#             filenames.extend(fnames)

#     embeddings = torch.cat(embeddings, 0)
#     embeddings = normalize(embeddings)
#     return embeddings, filenames

# def get_image_as_tensor(filename: str):
#     """Load an image file and convert it to a PyTorch tensor."""
#     image = Image.open(filename).convert('RGB')  # Convert to RGB to ensure 3 color channels
#     transform = torchvision.transforms.ToTensor()  # Convert image to a tensor
#     return transform(image)

# def plot_knn_examples(embeddings, filenames, n_neighbors=5, num_examples=10, save_dir='/home/elias/lightly/plots'):
#     nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embeddings)
#     _, indices = nbrs.kneighbors(embeddings)
#     samples_idx = np.random.choice(len(indices), size=num_examples, replace=False)

#     images_tensors = []
#     for idx in samples_idx:
#         # Include the sample itself as the first image
#         sample_and_neighbors_indices = indices[idx]
#         for neighbor_idx in sample_and_neighbors_indices:
#             fname = os.path.join(path_to_data, filenames[neighbor_idx])
#             image_tensor = get_image_as_tensor(fname)
#             images_tensors.append(image_tensor)

#     # Create a grid of images
#     images_grid = make_grid(images_tensors, nrow=n_neighbors + 1)
#     plt.figure(figsize=(20, 10))
#     plt.imshow(images_grid.permute(1, 2, 0))
#     plt.axis('off')

#     # Save the complete grid
#     os.makedirs('/home/elias/lightly/plots', exist_ok=True)
#     plt.savefig(os.path.join(save_dir, 'knn_examples_grid2.png'), bbox_inches='tight', dpi=100)
#     plt.close()


##  Standard Setting
# model = SimCLR()

# transform = SimCLRTransform(input_size=32)
# dataset = torchvision.datasets.CIFAR10(
#     "datasets/cifar10", download=True, transform=transform
# )
# # or create a dataset from a folder containing images or videos:
# # dataset = LightlyDataset("path/to/folder", transform=transform)

# dataloader = torch.utils.data.DataLoader(
#     dataset,
#     batch_size=3,
#     shuffle=True,
#     drop_last=True,
#     num_workers=0,
# )

# accelerator = "gpu" if torch.cuda.is_available() else "cpu"

# trainer = pl.Trainer(max_epochs=2, devices=1, accelerator=accelerator)
# trainer.fit(model=model, train_dataloaders=dataloader)


