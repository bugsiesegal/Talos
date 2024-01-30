from transformers import AutoTokenizer

from lightning_model import LightningIntegratedMemoryModelText
from config import Config
from lightning_data import HFStreamedTextDatamodule
import lightning as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, BatchSizeFinder, LearningRateFinder, ModelSummary

config = Config()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
config.vocab_size = tokenizer.vocab_size
tokenizer.pad_token = tokenizer.eos_token

# Hyperparameters
config.max_seq_len = 128
config.batch_size = 16
config.learning_rate = 1e-4
config.epochs = 10
config.context_length = 4
config.thinking_steps = 10
config.think_twice = False
config.stream = True

# Model Parameters
config.embed_dim = 256
config.num_heads = 16
config.num_layers = 8
config.hidden_dim = 512
config.dropout = 0.1
config.activation = 'gelu'
config.bias = False

model = LightningIntegratedMemoryModelText(config)

# Initialize the data module
dm = HFStreamedTextDatamodule(
    path="c4",
    subset="en",
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
        # BatchSizeFinder(),
        ModelSummary(max_depth=3)
    ],
    log_every_n_steps=10,
    max_time={"hours": 20}
)

# Train the model
trainer.fit(model, datamodule=dm)
