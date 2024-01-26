from transformers import AutoTokenizer

from lightning_model import LightningIntegratedMemoryModelText
from config import Config
from lightning_data import HFStreamedTextDatamodule
import lightning as pl
from lightning.pytorch.loggers import WandbLogger

config = Config()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
config.vocab_size = tokenizer.vocab_size
tokenizer.pad_token = tokenizer.eos_token

# Hyperparameters
config.max_seq_len = 128
config.batch_size = 2
config.learning_rate = 1e-4
config.epochs = 10
config.thinking_steps = 5
config.context_length = 8
config.think_twice = False
config.stream = True

# Model Parameters
config.embed_dim = 128
config.num_heads = 4
config.num_layers = 1
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

# Initialize the trainer
trainer = pl.Trainer(
    precision="16-mixed",
    max_epochs=config.epochs,
    logger=WandbLogger(project="integrated-memory-model", log_model=False),
)

# Train the model
trainer.fit(model, datamodule=dm)


