from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelPruning, LearningRateFinder, BatchSizeFinder
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger

from config import Config
from data import HFData
from model import LightningIntegratedMemoryModel


def train(config: Config):
    # Load data and model
    datamodule = HFData(config)
    model = LightningIntegratedMemoryModel(config)
    # Callbacks
    # pruning = ModelPruning(pruning_fn="l1_unstructured")
    lr_finder = LearningRateFinder()
    batch_finder = BatchSizeFinder()
    # Loggers
    wandb_logger = WandbLogger(project="Talos", log_model=True)
    tb_logger = TensorBoardLogger("lightning_logs", name="Talos")
    # Trainer
    trainer = Trainer(
        precision=config.precision,
        max_time=config.max_time,
        benchmark=True,
        callbacks=[batch_finder, lr_finder],
        logger=[wandb_logger, tb_logger],
        gradient_clip_val=config.gradient_clip_val,
        log_every_n_steps=config.log_every_n_steps,
    )
    # Train
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    config = Config()
    config.batch_size = 32
    config.learning_rate = 1e-4
    config.min_lr = 1e-8
    config.dropout = 0.0
    config.core_layers = 8
    config.max_length = 17
    config.streaming = True
    config.dataset_name = "c4"
    config.dataset_subname = "en"
    train(config)
