import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint, BatchSizeFinder, LearningRateFinder
from lightning.pytorch.loggers import WandbLogger
from transformers import AutoTokenizer

from config import Config
from lightning_data import HFStreamedTextDatamodule
from lightning_model import LightningIntegratedMemoryModelText

config = Config()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
config.vocab_size = tokenizer.vocab_size
tokenizer.pad_token = tokenizer.eos_token

# Hyperparameters
config.max_seq_len = 512
config.batch_size = 1
config.learning_rate = 1e-4
config.epochs = 10
config.context_length = 8
config.thinking_steps = 5
config.think_twice = False
config.stream = True

# Model Parameters
config.embed_dim = 128
config.num_heads = 8
config.num_layers = 4
config.hidden_dim = 512
config.dropout = 0.1
config.activation = 'gelu'
config.bias = False


model = LightningIntegratedMemoryModelText(config)

# Initialize the data module
dm = HFStreamedTextDatamodule(
    path="wikitext",
    subset="wikitext-2-v1",
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
        # BatchSizeFinder()
    ],
    log_every_n_steps=20,
    max_time={"minutes": 3},
    val_check_interval=1000,
)

# Train the model
trainer.fit(model, datamodule=dm)


