from torch.utils.data import Dataset
import torch
import pandas as pd
import random

class FMATokenDataset(Dataset):
    def __init__(self, ds_path, length):
        self.token_tracks_ds, self.tracks_length, self.Y = torch.load(ds_path)
        self.size = len(self.tracks_length)
        self.output_units = self.Y.max()
        self.length = length

    def __len__(self):
        return self.size

    def __getitem__(self, item):
        # Get snippet
        start = random.randint(0, self.tracks_length[item]-self.length)

        token_track = self.token_tracks_ds[item, start:start+self.length].int()
        Y = self.Y.iloc[item]
        return token_track, Y


