import torch
import torch.nn as nn
import lightning as pl

from logging_functions import calculate_perplexity
from model import IntegratedMemoryNeuralNetwork


class PLIntegratedMemoryNeuralNetwork(pl.LightningModule):
    def __init__(self, vocab_size, brain_size, embedding_dim, num_heads, num_layers, dim_feedforward, dropout,
                 learning_rate, context_length):
        super().__init__()
        self.model = IntegratedMemoryNeuralNetwork(vocab_size, brain_size, embedding_dim, num_heads, num_layers,
                                                   dim_feedforward, dropout)
        self.learning_rate = learning_rate
        self.context_length = context_length
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        total_loss = 0
        num_steps = 0
        for i in range(1, len(batch['input_ids'])):
            start_idx = max(0, i - self.context_length)
            inputs = torch.vstack(batch['input_ids'][start_idx:i]).T
            targets = batch['input_ids'][i]
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss
            num_steps += 1

        # Calculate metrics
        avg_loss = total_loss / num_steps
        perplexity = calculate_perplexity(avg_loss.cpu())


        # Log metrics
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        return avg_loss

    def validation_step(self, batch, batch_idx):
        total_val_loss = 0
        num_steps = 0
        for i in range(1, len(batch['input_ids'])):
            start_idx = max(0, i - self.context_length)
            inputs = torch.vstack(batch['input_ids'][start_idx:i]).T
            targets = batch['input_ids'][i]
            outputs = self(inputs)
            val_loss = self.criterion(outputs, targets)
            if not torch.isnan(val_loss):
                total_val_loss += val_loss
                num_steps += 1

        # Calculate metrics
        avg_val_loss = total_val_loss / num_steps if num_steps > 0 else torch.tensor(0.0)
        perplexity = calculate_perplexity(avg_val_loss.cpu())

        # Log metrics
        self.log('val_loss', avg_val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return avg_val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self) -> None:
        self.model.reset_state()

    def on_validation_end(self) -> None:
        self.model.reset_state()
