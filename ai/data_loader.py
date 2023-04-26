import json
import tqdm

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader


class PacketDataset(Dataset):
    def __init__(self, filepath):
        # with open(filepath, "r") as f:
            # self.data = [line for line in tqdm.tqdm(f, desc='loading data.. ')]
        self.data = pd.read_json(filepath)

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

