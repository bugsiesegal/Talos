from functools import partial

import torch
import wandb
from torch.optim import lr_scheduler
from transformers import AutoTokenizer

from input_module import TextInputModule
from lightning_model import LightningIntegratedMemoryModelText
from config import Config
from lightning_data import HFStreamedTextDatamodule
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, BatchSizeFinder, LearningRateFinder, ModelSummary


def run():
    run = wandb.init(project="integrated-memory-model", entity="bugsiesegal")
    config = Config()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    config.vocab_size = tokenizer.vocab_size
    tokenizer.pad_token = tokenizer.eos_token

    # Hyperparameters
    config.max_seq_len = wandb.config["max_seq_len"]
    config.batch_size = 16
    config.learning_rate = 1e-4
    config.epochs = 10
    config.context_length = wandb.config["context_length"]
    config.thinking_steps = wandb.config["thinking_steps"]
    config.think_twice = wandb.config["think_twice"]
    config.stream = True

    # Learning Rate Scheduler
    config.learning_rate_scheduler = partial(lr_scheduler.CosineAnnealingLR, T_max=10)

    # Model Parameters
    config.embed_dim = wandb.config["embed_dim"]
    config.num_heads = wandb.config["num_heads"]
    config.num_layers = wandb.config["num_layers"]
    config.hidden_dim = wandb.config["hidden_dim"]
    config.dropout = 0.0
    config.activation = 'gelu'
    config.bias = False

    # Initialize the model
    model = LightningIntegratedMemoryModelText(config)

    # Initialize the data module
    dm = HFStreamedTextDatamodule(
        path="wikitext",
        subset="wikitext-103-raw-v1",
        text_column="text",
        tokenizer=tokenizer,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len
    )

    wandb_logger = WandbLogger(project="integrated-memory-model", log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="validation_perplexity", mode="min")

    # Initialize the trainer
    trainer = pl.Trainer(
        precision="16-mixed",
        max_epochs=config.epochs,
        logger=wandb_logger,
        callbacks=[
            checkpoint_callback,
            BatchSizeFinder(),
            ModelSummary(max_depth=4)
        ],
        log_every_n_steps=10,
        max_time={"hours": wandb.config["max_time_hours"]}
    )

    # Train the model
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    wandb.init()

    wandb.config.max_seq_len = 128
    wandb.config.context_length = 4
    wandb.config.thinking_steps = 8
    wandb.config.think_twice = False
    wandb.config.embed_dim = 128
    wandb.config.num_heads = 64
    wandb.config.num_layers = 16
    wandb.config.hidden_dim = 512
    wandb.config.max_time_hours = 15

    run()
