from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from lightning import LightningDataModule


class HFData(LightningDataModule):
    def __init__(self, config):
        super(HFData, self).__init__()
        self.dataset = None
        self.config = config
        self.tokenizer = config.tokenizer
        self.max_length = config.max_length
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.streaming = config.streaming

    def setup(self, stage=None):
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True, max_length=self.max_length)

        self.dataset = load_dataset(self.config.dataset_name, self.config.dataset_subname, streaming=self.streaming)
        self.dataset = self.dataset.map(tokenize_function, batched=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.batch_size, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset["test"], batch_size=self.batch_size, num_workers=self.num_workers)
