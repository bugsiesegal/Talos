import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

import lightning as pl

import datasets
import transformers


class HuggingfaceTextDatamodule(pl.LightningDataModule):
    """
    Data module for Huggingface datasets.

    Args:
        dataset_name (str): Name of the dataset.
        subset (str): Subset of the dataset.
        batch_size (int): Size of the batches.
        num_workers (int): Number of workers.
        pin_memory (bool): Whether to pin memory.
        drop_last (bool): Whether to drop the last batch.
        shuffle (bool): Whether to shuffle the data.
        seed (int): Random seed.
        max_length (int): Maximum length of the sequences.
        streaming (bool): Whether to stream the data.
        tokenizer_name (str): Name of the tokenizer.
    """

    def __init__(
            self,
            dataset_name: str,
            subset: str,
            batch_size: int,
            context_length: int,
            num_workers: int,
            pin_memory: bool,
            drop_last: bool,
            shuffle: bool,
            seed: int,
            max_length: int,
            streaming: bool,
            tokenizer_name: str,
            **kwargs
    ):
        super().__init__()

        self.dataset_name = dataset_name
        self.subset = subset
        self.batch_size = batch_size
        self.context_length = context_length
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.max_length = max_length
        self.streaming = streaming
        self.tokenizer_name = tokenizer_name
        self.kwargs = kwargs

        self.dataset = None

        # Load tokenizer
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_name)

        self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        """
        Setup the data.

        Args:
            stage (str): Stage of setup.
        """

        # Load dataset
        self.dataset = datasets.load_dataset(self.dataset_name, self.subset, streaming=self.streaming)

        # Tokenize dataset
        self.dataset = self.dataset.map(
            self.tokenize,
            batch_size=self.kwargs["map_batch_size"] if "map_batch_size" in self.kwargs else 10000,
            batched=True,
        )

        # Unfold dataset
        self.dataset = self.dataset.map(
            self.unfold,
            batch_size=self.kwargs["map_batch_size"] if "map_batch_size" in self.kwargs else 10000,
            batched=True,
        )

    def tokenize(self, x):
        """
        Tokenize a batch.

        Args:
            x (dict): Batch.

        Returns:
            dict: Tokenized batch.
        """

        # Tokenize batch
        x = self.tokenizer(
            x["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return x

    def unfold(self, x):
        """
        Unfold a batch.

        Args:
            x (dict): Batch.

        Returns:
            dict: Unfolded batch.
        """

        # Unfold batch
        x["input_ids"] = torch.stack(x["input_ids"]).unfold(1, self.context_length+1, 1)
        x["attention_mask"] = torch.stack(x["attention_mask"]).unfold(1, self.context_length+1, 1)

        return x

    def train_dataloader(self):
        """
        Training data loader.

        Returns:
            DataLoader: Training data loader.
        """

        return DataLoader(
            self.dataset["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        """
        Validation data loader.

        Returns:
            DataLoader: Validation data loader.
        """

        return DataLoader(
            self.dataset["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
        )

    def test_dataloader(self):
        """
        Test data loader.

        Returns:
            DataLoader: Test data loader.
        """

        return DataLoader(
            self.dataset["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
            shuffle=self.shuffle,
        )

