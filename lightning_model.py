from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from input_module import TextInputModule
from output_module import TextOutputModule
from model import IntegratedMemoryModel
from config import Config
import lightning as pl
from lightning.pytorch.loggers import WandbLogger


class LightningIntegratedMemoryModelText(pl.LightningModule):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Initialize the core memory-integrated model
        self.model = IntegratedMemoryModel(config)

        # Initialize input and output modules for text processing
        text_input_module = TextInputModule(config)
        text_output_module = TextOutputModule(config)

        # Add input and output modules to the main model
        self.model.add_input_module('text', text_input_module)
        self.model.add_output_module('text', text_output_module)

        self.num_generated_tokens = 1

        self.loss = torch.nn.CrossEntropyLoss()

        self.lr = config.lr

        self.save_hyperparameters()

        self.wandb_logger = WandbLogger(project="integrated-memory-model", log_model="all")

    def training_step(self, batch, batch_idx):
        self.model.reset_hidden_state()
        # Training step for the model
        input_ids = torch.vstack(batch['input_ids'])
        unfolded_input_ids = input_ids.unfold(0, self.config.context_length + 1, 1)

        total_loss = 0
        total_perplexity = 0

        for batch_ids in unfolded_input_ids:
            # Generate next token and calculate loss
            generated_ids = self.model({"text": batch_ids[:, :-1]})
            loss = self.loss(generated_ids['text'], batch_ids[:, -1])
            self.log('train_loss', loss)
            perplexity = torch.exp(loss)
            self.log('train_perplexity', perplexity)
            total_loss += loss
            total_perplexity += perplexity

        return total_loss

    def validation_step(self, batch, batch_idx):
        self.model.reset_hidden_state()
        # Validation step for the model
        input_ids = torch.vstack(batch['input_ids'])
        unfolded_input_ids = input_ids.unfold(0, self.config.context_length + 1, 1)

        total_loss = 0
        total_perplexity = 0

        for batch_ids in unfolded_input_ids:
            # Generate next token and calculate loss
            generated_ids = self.model({"text": batch_ids[:, :-1]})
            loss = self.loss(generated_ids['text'], batch_ids[:, -1])
            self.log('validation_loss', loss)
            perplexity = torch.exp(loss)
            self.log('validation_perplexity', perplexity)
            total_loss += loss
            total_perplexity += perplexity

        return total_loss

    def test_step(self, batch, batch_idx):
        self.model.reset_hidden_state()
        # Test step for the model
        input_ids = torch.vstack(batch['input_ids'])
        unfolded_input_ids = input_ids.unfold(0, self.config.context_length + 1, 1)

        total_loss = 0
        total_perplexity = 0

        for batch_ids in unfolded_input_ids:
            # Generate next token and calculate loss
            generated_ids = self.model({"text": batch_ids[:, :-1]})
            loss = self.loss(generated_ids['text'], batch_ids[:, -1])
            self.log('test_loss', loss)
            perplexity = torch.exp(loss)
            self.log('test_perplexity', perplexity)
            total_loss += loss
            total_perplexity += perplexity

        return total_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Configure optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
