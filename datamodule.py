import lightning as pl
import torch
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from datasets import Dataset


class IntegratedMemoryNetworkDataModule(pl.LightningDataModule):
    def __init__(self, dataset_name: str, subset: str, split: str):
        super().__init__()

        self.dataset_name = dataset_name
        self.subset = subset
        self.split = split


