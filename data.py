import torch
import torchvision.transforms
from datasets import load_dataset
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class C4DataModule(LightningDataModule):
    def __init__(self, subset, batch_size, tokenizer_name, max_length, context_length=128, stride=1, streaming=True):
        super().__init__()

        # Check that values are valid
        if context_length > max_length:
            raise ValueError("Context length must be smaller than max length.")

        # Define variables
        self.dataset = None
        self.tokenizer = None
        self.subset = subset
        self.batch_size = batch_size
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.streaming = streaming
        self.context_length = int(context_length)
        self.stride = int(stride)

    def unfold(self, sample):
        unfolded_sample = {"unfolded": torch.tensor(sample).unfold(1, self.context_length + 1, self.stride)}

        return unfolded_sample

    def split(self, sample):
        tensor = torch.stack(sample)
        inputs = tensor[:, :, :-1]
        labels = tensor[:, :, -1]

        return {"inputs": inputs, "labels": labels}

    def setup(self, stage: str) -> None:
        # Load tokenizer and dataset
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.dataset = load_dataset("c4", self.subset, streaming=self.streaming)

        # Set eos_token
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.eos_token else "eos"

        # Tokenize dataset
        self.dataset = self.dataset.map(
            lambda x: self.tokenizer(x["text"], truncation=True, padding="max_length", max_length=self.max_length),
            batched=True,
        )

        # Unfold dataset
        self.dataset = self.dataset.map(self.unfold, batched=True, input_columns="input_ids")

        # Split dataset
        self.dataset = self.dataset.map(self.split, batched=True, input_columns="unfolded")

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dataset["validation"], batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        raise NotImplementedError
