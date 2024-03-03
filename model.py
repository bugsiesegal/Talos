import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Config

import lightning as pl

class IntegratedMemoryModel(nn.Module):
    def __init__(self, config: Config):
        """
        Model Structure
        ----------------
        - Embedding Layer : nn.Embedding
        - Positional Embedding : nn.Embedding
        - Transformer Preprocessor : nn.TransformerEncoder
        - Memory Updater : nn.Linear
        - Transformer Core : nn.TransformerEncoder
        - Memory Splitter : nn.Linear
        - Transformer Postprocessor : nn.TransformerEncoder
        - Output Layer : nn.Linear
        """

        super(IntegratedMemoryModel, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.positional_embedding = nn.Embedding(config.max_length, config.embedding_dim)
        self.transformer_preprocessor = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        ), num_layers=config.preprocessor_layers)
        self.transformer_core = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        ), num_layers=config.core_layers)
        self.transformer_postprocessor = nn.TransformerEncoder(nn.TransformerEncoderLayer(
            d_model=config.embedding_dim,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
        ), num_layers=config.postprocessor_layers)
        self.output_layer = nn.Linear(config.embedding_dim, config.vocab_size)

        self.memory_updater = nn.Linear(config.embedding_dim * 2, config.embedding_dim)
        self.memory_splitter = nn.Linear(config.embedding_dim, config.embedding_dim * 2)

        self.memory = None

        self.thinking_steps = config.thinking_steps

    def forward(self, x):
        if self.memory is None or self.memory.size(0) != x.size(0):
            self.memory = torch.zeros(x.size(0), x.size(1), self.config.embedding_dim, device=x.device)

        x = self.embedding(x) + self.positional_embedding(torch.arange(x.size(1), device=x.device))
        x = self.transformer_preprocessor(x)
        x = self.memory_updater(torch.cat([x, self.memory], dim=-1))
        for i in range(self.thinking_steps):
            x = self.transformer_core(x)
        self.memory, x = torch.chunk(self.memory_splitter(x), 2, dim=-1)
        x = self.transformer_postprocessor(x)
        return self.output_layer(x)

    def reset_memory(self):
        self.memory = None


class LightningIntegratedMemoryModel(pl.LightningModule):
    def __init__(self, config: Config):
        super(LightningIntegratedMemoryModel, self).__init__()
        self.model = IntegratedMemoryModel(config)
        self.config = config
        self.learning_rate = config.learning_rate

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, batch_idx, prefix):
        self.model.reset_memory()
        loss = 0
        unfolded_batch = torch.stack(batch['input_ids']).T.unfold(1, self.config.context_window + 1, 1)
        for i in range(unfolded_batch.size(1)):
            x = unfolded_batch[:, i, :-1]
            y = unfolded_batch[:, i, -1]
            y_hat = self.model(x)[..., -1, :]
            loss += F.cross_entropy(y_hat.view(-1, self.config.vocab_size), y.view(-1))
        loss /= unfolded_batch.size(1)
        self.log(f'{prefix}_loss', loss)
        self.log(f'{prefix}_ppl', torch.exp(loss))
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, 'val')
        return loss

    def test_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, 'test')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=self.config.lr_scheduler_factor,
                    patience=self.config.lr_scheduler_patience,
                    min_lr=self.config.min_lr,
                    verbose=True
                ),
                'interval': 'step',
                'monitor': 'train_loss',
            },
        }







