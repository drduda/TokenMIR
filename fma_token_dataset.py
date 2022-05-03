from typing import Optional
from torch.utils.data import Dataset, DataLoader
import torch
import os
import random
from tokenmir_datamodule import TokenMIRDataModule as DataModule


class FMATokenDataset(Dataset):
    def __init__(self, ds_path, length):
        super().__init__()
        self.token_tracks_ds, self.tracks_length, self.Y = torch.load(ds_path)
        self.size = len(self.token_tracks_ds)
        self.output_units = self.Y.max() + 1
        self.length = length

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        # Get snippet
        try:
            start = random.randint(0, self.tracks_length[item]-self.length)

            token_track = self.token_tracks_ds[item, start:start+self.length].int()
        except ValueError:
            # Some tracks are not stored, so only zeros are inside tocken tracks_ds
            token_track = self.token_tracks_ds[item, : self.length].int()
        Y = self.Y.iloc[item]
        return token_track, Y

from typing import Optional
from torch.utils.data import Dataset, DataLoader
import torch
import os
import random
import pytorch_lightning as pl


class FMATokenDataset(Dataset):
    def __init__(self, ds_path, length):
        super().__init__()
        self.token_tracks_ds, self.tracks_length, self.Y = torch.load(ds_path)
        self.size = len(self.token_tracks_ds)
        self.output_units = self.Y.max() + 1
        self.length = length

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        # Get snippet
        try:
            start = random.randint(0, self.tracks_length[item]-self.length)

            token_track = self.token_tracks_ds[item, start:start+self.length].int()
        except ValueError:
            # Some tracks are not stored, so only zeros are inside tocken tracks_ds
            token_track = self.token_tracks_ds[item, : self.length].int()
        Y = self.Y.iloc[item]
        return token_track, Y

class FMACodebookDataset(FMATokenDataset):
    def __init__(self, ds_path, length):
        super().__init__(ds_path, length)

        self.codebooks = torch.load("./codebooks.pt")

    def __getitem__(self, item):
        tocken_track, Y = super().__getitem__(item)
        codebook_track = self.codebooks[tocken_track.long()]
        return codebook_track, Y


class MIRDataModule(DataModule):
    def __init__(self, data_dir, batch_size, token_length, ds_size="medium"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.token_length = token_length
        self.ds_size = ds_size

    def get_target_distribution_weights(self):
        train_path = os.path.join(self.data_dir, "tokens_ds_size_"+self.ds_size+"_split_training.pt")
        train_ds = FMATokenDataset(train_path, self.token_length)

        target_distribution = train_ds.Y.value_counts().sort_index()
        # Normalize
        target_distribution = torch.Tensor(target_distribution.values / len(train_ds))

        assert torch.sum(target_distribution).item() == 1

        # Inverse distribtion
        num_classes = train_ds.output_units
        inverse_target_distribution = (1/target_distribution)/num_classes
        return inverse_target_distribution


    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size)

class FMATokenDataModule(MIRDataModule):
    def setup(self, stage: Optional[str] = None):
        train_path = os.path.join(self.data_dir, "tokens_ds_size_"+self.ds_size+"_split_training.pt")
        self.train_ds = FMATokenDataset(train_path, self.token_length)

        val_path = os.path.join(self.data_dir, "tokens_ds_size_"+self.ds_size+"_split_validation.pt")
        self.val_ds = FMATokenDataset(val_path, self.token_length)

        test_path = os.path.join(self.data_dir, "tokens_ds_size_"+self.ds_size+"_split_test.pt")
        self.test_ds = FMATokenDataset(test_path, self.token_length)

class CodebookDataModule(MIRDataModule):
    def setup(self, stage: Optional[str] = None):
        train_path = os.path.join(self.data_dir, "tokens_ds_size_"+self.ds_size+"_split_training.pt")
        self.train_ds = FMACodebookDataset(train_path, self.token_length)

        val_path = os.path.join(self.data_dir, "tokens_ds_size_"+self.ds_size+"_split_validation.pt")
        self.val_ds = FMACodebookDataset(val_path, self.token_length)

        test_path = os.path.join(self.data_dir, "tokens_ds_size_"+self.ds_size+"_split_test.pt")
        self.test_ds = FMACodebookDataset(test_path, self.token_length)