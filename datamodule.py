import os
import glob
import pydicom
import pandas as pd
from typing import Callable, Optional, Sequence

from argparse import ArgumentParser
from monai.data import Dataset, DataLoader, list_data_collate
from monai.data.utils import first
from monai.utils import set_determinism
from monai.transforms import (
    apply_transform,
    Randomizable, 
    HistogramNormalized,
    EnsureChannelFirstd,
    Compose,
    OneOf,
    LoadImaged,
    Spacingd,
    Orientationd,
    DivisiblePadd,
    RandFlipd,
    RandZoomd,
    RandScaleCropd,
    CropForegroundd,
    Resized, Rotate90d, HistogramNormalized,
    ScaleIntensityd,
    ScaleIntensityRanged,
    ToTensord,
)

from pytorch_lightning import LightningDataModule

class DICOMDataset(Dataset, Randomizable):
    def __init__(
        self, 
        keys: Sequence,
        datadir: str = None,   # type: ignore
        csvfile: str = None,   # type: ignore
        transform: Optional[Callable] = None,
        length: Optional[Callable] = None,
        batch_size: int = 32,
    ) -> None: 
        self.keys = keys,
        self.datadir = datadir
        self.csvfile = csvfile
        self.length = length
        self.batch_size = batch_size
        self.transform = transform

        def glob_files(folders: str = None, extension: str = '*.nii.gz'):  # type: ignore
            assert folders is not None
            paths = [glob.glob(os.path.join(folder, extension), recursive=True) for folder in folders]
            files = sorted([item for sublist in paths for item in sublist])
            print(len(files))
            print(files[:1])
            return files

        self.files = glob_files(folders=self.datadir, extension='**/*.dcm')

        # Read the dataframe
        self.df = pd.read_csv(self.csvfile)
        print("Data file: ", self.csvfile, len(self.df))

    def __len__(self) -> int:
        return min(self.length, len(self.files)) if self.length is not None else len(self.files)  # type: ignore

    def _transform(self, index: int):
        data = {}
        filepath = self.files[index]
        # Remove ext then split
        # /path/file.ext to /path/file to path, file
        patient_id, image_id = os.path.splitext(filepath)[0].split("/")[-2:]
        
        if self.df is not None:
            label = self.df.loc[(self.df['patient_id'] == patient_id) & (self.df['image_id'] == image_id)]
        else:
            label = None

        data["image"] = filepath
        data["patient_id"] = patient_id
        data["image_id"] = image_id
        data["label"] = label 

        # Attach other attributes
        # Extracting data from the mri file
        _dcm = pydicom.read_file(filepath)

        # depending on this value, X-ray may look inverted - fix that:
        data["isinverted"] = 1.0 if _dcm.PhotometricInterpretation == "MONOCHROME1" else 0.0

        # Extract the laterality
        # R right
        # L left
        # B both (e.g., cleavage)
        data["laterality"] = _dcm.ImageLaterality 

        # Apply other transforms for image augmentations
        if self.transform is not None:
            data = apply_transform(self.transform, data)
        return data


class DICOMDataModule(LightningDataModule):
    def __init__(self,
        train_csvfile: str = "path/to/csvfile",
        train_datadir: str = "path/to/datadir",
        val_csvfile: str = "path/to/csvfile",
        val_datadir: str = "path/to/datadir",
        test_csvfile: str = "path/to/csvfile",
        test_datadir: str = "path/to/datadir",
        shape: int = 256,
        batch_size: int = 32
    ) -> None:
        super().__init__()

        self.batch_size = batch_size
        self.shape = shape
        
        # self.setup()
        self.train_csvfile = train_csvfile
        self.train_datadir = train_datadir
        self.val_csvfile = val_csvfile
        self.val_datadir = val_datadir
        self.test_csvfile = test_csvfile
        self.test_datadir = test_datadir

        # self.setup()
        
    def setup(self, seed: int = 2222, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        self.train_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"],),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0), mode=["bilinear"], align_corners=True),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0,),
            CropForegroundd(keys=["image"], source_key="image2d", select_fn=(lambda x: x > 0), margin=0),
            HistogramNormalized(keys=["image"], min=0.0, max=1.0,),  # type: ignore
            RandZoomd(keys=["image"], prob=1.0, min_zoom=0.8, max_zoom=1.1, padding_mode='constant', mode=["trilinear"], align_corners=True),
            Resized(keys=["image"], spatial_size=self.shape, size_mode="longest", mode=["trilinear"], align_corners=True),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=2),
            DivisiblePadd(keys=["image"], k=self.shape, mode="constant", constant_values=0),
            ToTensord(keys=["image", "label", "isinverted", "laterality"],),
        ])

        self.train_datasets = DICOMDataset(
            keys=["image", "label", "isinverted", "laterality"],
            datadir=self.train_datadir,
            csvfile=self.train_csvfile,
            transform=self.train_transforms,  
            batch_size=self.batch_size,  
        )

        self.train_loader = DataLoader(
            self.train_datasets,
            batch_size=self.batch_size,
            num_workers=16,
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.train_loader

    def val_dataloader(self):
        self.val_transforms = Compose([
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"],),
            Spacingd(keys=["image"], pixdim=(1.0, 1.0), mode=["bilinear"], align_corners=True),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0,),
            CropForegroundd(keys=["image"], source_key="image2d", select_fn=(lambda x: x > 0), margin=0),
            HistogramNormalized(keys=["image"], min=0.0, max=1.0,),  # type: ignore
            RandZoomd(keys=["image"], prob=1.0, min_zoom=0.8, max_zoom=1.1, padding_mode='constant', mode=["trilinear"], align_corners=True),
            Resized(keys=["image"], spatial_size=self.shape, size_mode="longest", mode=["trilinear"], align_corners=True),
            DivisiblePadd(keys=["image"], k=self.shape, mode="constant", constant_values=0),
            ToTensord(keys=["image", "label", "isinverted", "laterality"],),
        ])

        self.val_datasets = DICOMDataset(
            keys=["image", "label", "isinverted", "laterality"],
            datadir=self.val_datadir,
            csvfile=self.val_csvfile,
            transform=self.val_transforms,
            batch_size=self.batch_size,
        )

        self.val_loader = DataLoader(
            self.val_datasets,
            batch_size=self.batch_size,
            num_workers=8,
            collate_fn=list_data_collate,
            shuffle=True,
        )
        return self.val_loader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=2222)
    parser.add_argument("--shape", type=int, default=256,help="isotropic shape")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")

    hparams = parser.parse_args()
    
    # Create data module
    train_csvfile = os.path.join(hparams.datadir, "train.csv")
    test_csvfile = os.path.join(hparams.datadir, "test.csv")
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
    datamodule.setup(seed=hparams.seed)

    debug_data = first(datamodule.train_dataloader())
    
    print(
        debug_data["image"].shape, 
        debug_data["label"], 
        debug_data["isinverted"], 
        debug_data["laterality"], 
    )