import os
import glob
import pydicom
import numpy as np
import pandas as pd
from typing import Callable, Optional, Sequence, List

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
    Identityd,
    LoadImaged,
    Spacingd,
    Orientationd,
    DivisiblePadd,
    RandFlipd,
    RandZoomd,
    SqueezeDimd,
    RandScaleCropd, 
    RandRotated, 
    CropForegroundd,
    Resized, Rotate90d, 
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
        upsample: bool = True,
        has_png: bool = False,
    ) -> None: 
        self.keys = keys,
        self.datadir = datadir
        self.csvfile = csvfile
        self.length = length
        self.batch_size = batch_size
        self.transform = transform
        self.upsample = upsample
        self.has_png = has_png

        def glob_files(folders: str = None, extension: str = '*.nii.gz'):  # type: ignore
            assert folders is not None
            # paths = [glob.glob(os.path.join(folder, extension), recursive=True) for folder in folders]
            paths = glob.glob(os.path.join(folders, extension), recursive=True) 
            files = sorted([item for sublist in paths for item in sublist])
            return files

        if self.csvfile is not None:
            # Read the dataframe
            print("Data file: ", self.csvfile)
            self.df = pd.read_csv(self.csvfile).fillna(0) # Fill not a number
            print("Before upsampling: ")
            print(self.df.describe())
            if self.upsample:
                ### Separate the majority and minority classes
                df_minority = self.df[self.df['cancer']==1]
                df_majority = self.df[self.df['cancer']==0]
                ### Now, downsamples majority labels equal to the number of samples in the minority class
                ### Sum both to 100_000
                df_minority_up = df_minority.sample( (100_000-len(df_majority)), random_state=0, replace=True)
                ### concat the majority and minority dataframes
                self.df = pd.concat([df_majority, df_minority_up])
                ## Shuffle the dataset to prevent the model from getting biased by similar samples
                self.df = self.df.sample(frac=1, random_state=0)
                print("After upsampling: ")
                print(self.df.describe())

            self.files = [os.path.join(self.datadir, 
                                       self.df.iloc[id].at["patient_id"].astype(str), 
                                       self.df.iloc[id].at["image_id"].astype(str)+".dcm") \
                                    for id in range(len(self.df)) ]
            print(len(self.files))
            print(self.files[:1])
        else:
            self.files = glob_files(folders=self.datadir, extension='**/*.dcm')  
            print(len(self.files))
            print(self.files[:1])

    def __len__(self) -> int:
        return min(self.length, len(self.files)) if self.length is not None else len(self.files)  # type: ignore

    def _transform(self, index: int):
        data = {}
        filepath = self.files[index]
        # Remove ext then split
        # /path/file.ext to /path/file to path, file
        patient_id, image_id = os.path.splitext(filepath)[0].split("/")[-2:]
        
        if self.df is not None:
            label = self.df.loc[(self.df['patient_id'] == int(patient_id)) & (self.df['image_id'] == int(image_id)), 'cancer'].values[0]
        data["image"] = filepath.replace("dcm", "png") if self.has_png else filepath # Adjust fast reading here
        data["label"] = 1 if label==1 else 0
        # data["label"] = 1. if label==1 else 0.
        data["patient_id"] = patient_id if patient_id is not None else None
        data["image_id"] = image_id if image_id is not None else None
        
        if self.has_png:
            pass
        else:
            # Attach other attributes
            # Extracting data from the mri file
            _dcm = pydicom.read_file(filepath)

            # depending on this value, X-ray may look inverted - fix that:
            # data["isinverted"] = 1.0 if _dcm.PhotometricInterpretation == "MONOCHROME1" else 0.0
            if _dcm.PhotometricInterpretation is not None and _dcm.PhotometricInterpretation == "MONOCHROME1":
                data["isinverted"] = 1
            else: 
                data["isinverted"] = 0
            # Extract the laterality
            # L left
            # R right
            # B both (e.g., cleavage)
            if _dcm.ImageLaterality == "L":
                data["laterality"] = 0
            elif _dcm.ImageLaterality == "R":
                data["laterality"] = 1
            # elif _dcm.ImageLaterality == "B":
            #     data["laterality"] = 2
            # print(data)

        # Apply other transforms for image augmentations
        if self.transform is not None:
            data = apply_transform(self.transform, data)

        return data

class PhotometricInterpretationMONOCHROME1:
    def __call__(self, data):
        if data["isinverted"] == 0:
            data["image"] = 1 - data["image"]
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
        batch_size: int = 32, 
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
        
    def setup(self, seed: int = 42, stage: Optional[str] = None):
        # make assignments here (val/train/test split)
        # called on every process in DDP
        set_determinism(seed=seed)

    def train_dataloader(self):
        has_png = self.train_csvfile is not None # 
        self.train_transforms = Compose([
            LoadImaged(
                keys=["image"], 
                reader="pilreader" if has_png else "pydicomreader",
                ensure_channel_first=True, 
                # image_only=True, 
                # overwriting=True, 
                # meta_keys=["PhotometricInterpretation"], 
                # allow_missing_keys=True, 
                # simple_keys=True
            ),
            # EnsureChannelFirstd(keys=["image"],),
            # Spacingd(keys=["image"], pixdim=(1.0, 1.0), mode=["bilinear"], align_corners=True),
            # SqueezeDimd(keys=["image"], dim=3),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0,),
            Identityd(keys=["image"]) if has_png else PhotometricInterpretationMONOCHROME1(), 
            CropForegroundd(keys=["image"], source_key="image", select_fn=(lambda x: x < 1), margin=0),
            # HistogramNormalized(keys=["image"], min=0.0, max=1.0,),  # type: ignore
            # RandRotated(keys=["image"], prob=1.0, range_x=0.3),
            # RandZoomd(keys=["image"], prob=1.0, min_zoom=1.0, max_zoom=1.2, padding_mode='constant', constant_values=1, mode=["bilinear"], align_corners=True),
            Resized(keys=["image"], spatial_size=self.shape, size_mode="longest", mode=["bilinear"], align_corners=True),
            RandFlipd(keys=["image"], prob=0.5, spatial_axis=0),
            DivisiblePadd(keys=["image"], k=self.shape, mode="constant", constant_values=1),
            ToTensord(keys=["image", "label"]) if has_png else ToTensord(keys=["image", "label", "isinverted", "laterality"],),
        ])

        self.train_datasets = DICOMDataset(
            keys=["image", "label"], #, "isinverted", "laterality"],
            datadir=self.train_datadir,
            csvfile=self.train_csvfile,
            transform=self.train_transforms,  
            batch_size=self.batch_size,  
            length=100000, 
            has_png=has_png
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
        has_png = self.val_csvfile is not None # 
        self.val_transforms = Compose([
            LoadImaged(
                keys=["image"], 
                reader="pilreader" if has_png else "pydicomreader",
                ensure_channel_first=True, 
                # image_only=True, 
                # overwriting=True, 
                # meta_keys=["PhotometricInterpretation"], 
                # allow_missing_keys=True, 
                # simple_keys=True
            ),
            # EnsureChannelFirstd(keys=["image"],),
            # Spacingd(keys=["image"], pixdim=(1.0, 1.0), mode=["bilinear"], align_corners=True),
            # SqueezeDimd(keys=["image"], dim=3),
            ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0,),
            Identityd(keys=["image"]) if has_png else PhotometricInterpretationMONOCHROME1(), 
            CropForegroundd(keys=["image"], source_key="image", select_fn=(lambda x: x < 1), margin=0),
            # HistogramNormalized(keys=["image"], min=0.0, max=1.0,),  # type: ignore
            # RandZoomd(keys=["image"], prob=1.0, min_zoom=0.8, max_zoom=1.1, padding_mode='edge', mode=["bilinear"], align_corners=True),
            Resized(keys=["image"], spatial_size=self.shape, size_mode="longest", mode=["bilinear"], align_corners=True),
            DivisiblePadd(keys=["image"], k=self.shape, mode="constant", constant_values=1),
            ToTensord(keys=["image", "label"]) if has_png else ToTensord(keys=["image", "label", "isinverted", "laterality"],),
        ])

        self.val_datasets = DICOMDataset(
            keys=["image", "label"], #, "isinverted", "laterality"],
            datadir=self.val_datadir,
            csvfile=self.val_csvfile,
            transform=self.val_transforms,
            batch_size=self.batch_size,
            length=100000, 
            has_png=has_png
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shape", type=int, default=512, help="isotropic shape")
    parser.add_argument("--datadir", type=str, default='data', help="data directory")
    parser.add_argument("--batch_size", type=int, default=8, help="batch size")

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