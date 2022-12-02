from typing import Optional, Sequence
from argparse import ArgumentParser

import torchvision
import torch.nn as nn
import torch
import resource
import os
import warnings
warnings.filterwarnings("ignore")
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

torch.cuda.empty_cache()
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.multiprocessing.set_sharing_strategy('file_system')

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.utilities.seed import seed_everything

from monai.losses.dice import DiceLoss
from monai.networks.nets import EfficientNetBN

from datamodule import DICOMDataModule

class DICOMLightningModule(LightningModule):
    def __init__(self, hparams, **kwargs):
        super().__init__()
        self.logsdir = hparams.logsdir
        self.lr = hparams.lr
        self.shape = hparams.shape
        self.filter = hparams.filter

        self.weight_decay = hparams.weight_decay
        self.batch_size = hparams.batch_size
        self.devices = hparams.devices
        self.model = nn.Sequential(
            EfficientNetBN(
                "efficientnet-b8", 
                spatial_dims=2, 
                in_channels=1, 
                num_classes=1, 
                pretrained=True, 
                adv_prop=True,
            ),
            nn.Sigmoid(),
        )

        self.loss = DiceLoss()
        self.save_hyperparameters()

    def forward(self, image):
        return self.model(image)

    def _common_step(self, batch, batch_idx, optimizer_idx, stage: Optional[str] = 'evaluation'):
        image = batch["image"]
        label = batch["label"]
        _device = image.device
        isinverted = batch["isinverted"]
        laterality = batch["laterality"]

        # Invert the image if true
        image[isinverted == 1.0] = 1.0 - image[isinverted == 1.0]

        preds = self.forward(image)
        loss = self.loss(label, preds)

        if batch_idx == 0:
            grid = torchvision.utils.make_grid(image, normalize=False, scale_each=False, nrow=8, padding=0)
            tensorboard = self.logger.experiment  # type: ignore
            tensorboard.add_image(f'{stage}_samples', grid.clamp(0., 1.), self.current_epoch*self.batch_size + batch_idx)

        info = {f'loss': loss}
        return info

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='validation')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, optimizer_idx=0, stage='test')

    def _common_epoch_end(self, outputs, stage: Optional[str] = 'common'):
        loss = torch.stack([x[f'loss'] for x in outputs]).mean()
        self.log(f'{stage}_loss_epoch', loss, on_step=False, prog_bar=True, logger=True, sync_dist=True)

    def train_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='train')

    def validation_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='validation')

    def test_epoch_end(self, outputs):
        return self._common_epoch_end(outputs, stage='test')

    def configure_optimizers(self):
        optimizer = torch.optim.RAdam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 200], gamma=0.1)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--conda_env", type=str, default="NeRV")
    parser.add_argument("--notification_email", type=str, default="quantm88@gmail.com")

    # Model arguments
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    parser.add_argument("--shape", type=int, default=512, help="spatial size of the tensor")
    parser.add_argument("--epochs", type=int, default=301, help="number of epochs")
    parser.add_argument("--train_samples", type=int, default=1000, help="training samples")
    parser.add_argument("--val_samples", type=int, default=400, help="validation samples")
    parser.add_argument("--test_samples", type=int, default=400, help="test samples")

    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--ckpt", type=str, default=None, help="path to checkpoint")
    parser.add_argument("--logsdir", type=str, default='logsfrecaling', help="logging directory")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--filter", type=str, default='sobel', help="None, sobel, laplacian, canny")

    parser = Trainer.add_argparse_args(parser)

    # Collect the hyper parameters
    hparams = parser.parse_args()  # type: ignore

    # Seed the application
    seed_everything(42)

    # Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=hparams.logsdir,
        filename='{epoch:02d}-{validation_loss_epoch:.2f}',
        save_top_k=-1,
        save_last=True,
        every_n_epochs=5,
    )
    lr_callback = LearningRateMonitor(logging_interval='step')

    # Logger
    tensorboard_logger = TensorBoardLogger(save_dir=hparams.logsdir, log_graph=True)

    # Init model with callbacks
    trainer = Trainer.from_argparse_args(
        hparams,
        max_epochs=hparams.epochs,
        logger=[tensorboard_logger],
        callbacks=[
            lr_callback,
            checkpoint_callback,
        ],
        # accumulate_grad_batches=5,
        # strategy="ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        strategy="fsdp",  # "fsdp", #"ddp_sharded", #"horovod", #"deepspeed", #"ddp_sharded",
        precision=16,  # if hparams.use_amp else 32,
        # stochastic_weight_avg=True,
        # deterministic=False,
        # profiler="simple",
    )
    
    # Create data module
    train_csvfile = os.path.join(hparams.datadir, "train.csv")
    test_csvfile = None #os.path.join(hparams.datadir, "test.csv")
    train_datadir = os.path.join(hparams.datadir, "train_images")
    test_datadir = os.path.join(hparams.datadir, "test_images")

    datamodule = DICOMDataModule(
        train_datadir=train_datadir,
        train_csvfile=train_csvfile,
        val_datadir=train_datadir,
        val_csvfile=train_csvfile,
        test_datadir=train_datadir,
        test_csvfile=train_csvfile,
        batch_size=hparams.batch_size,
        shape=hparams.shape,
    )
    datamodule.setup(seed=42)

    ####### Test camera mu and bandwidth ########
    # test_random_uniform_cameras(hparams, datamodule)
    #############################################

    model = DICOMLightningModule(
        hparams=hparams
    )
    model = model.load_from_checkpoint(hparams.ckpt, strict=False) if hparams.ckpt is not None else model

    trainer.fit(
        model,
        datamodule,
    )

    # test

    # serve