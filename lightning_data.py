import lightning as pl
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import DataLoader


class HFStreamedTextDatamodule(pl.LightningDataModule):
    def __init__(
            self,
            path: str,
            subset: str,
            text_column: str,
            tokenizer,
            batch_size: int = 32,
            max_seq_len: int = 512
    ):
        super().__init__()
        self.path = path
        self.subset = subset
        self.text_column = text_column

        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.dataset = None
        self.tokenizer = tokenizer

    def prepare_data(self):
        self.dataset = load_dataset(path=self.path, name=self.subset, streaming=True)

    def setup(self, stage=None):
        self.dataset = self.dataset.map(lambda x:
                                        self.tokenizer(
                                            x[self.text_column],
                                            truncation=True,
                                            padding='max_length',
                                            max_length=self.max_seq_len
                                        ),
                                        batched=True,
                                        )

    def train_dataloader(self):
        try:
            return DataLoader(self.dataset['train'], batch_size=self.batch_size)
        except KeyError:
            return None

    def val_dataloader(self):
        try:
            return DataLoader(self.dataset['validation'], batch_size=self.batch_size)
        except KeyError:
            return None

    def test_dataloader(self):
        try:
            return DataLoader(self.dataset['test'], batch_size=self.batch_size)
        except KeyError:
            return None
