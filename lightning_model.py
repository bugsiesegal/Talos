import lightning as pl
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from model import IntegratedMemoryModel


class LightningIntegratedMemoryModel(IntegratedMemoryModel, pl.LightningModule):
    """
    Lightning version of the integrated memory model.
    """

    def __init__(self, embedding_size, brain_state_size, num_layers, thinking_iterations, learning_rate=1e-3, **kwargs):
        super().__init__(embedding_size, brain_state_size, num_layers, thinking_iterations, **kwargs)

        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (torch.Tensor): Batch of shape (batch_size, seq_len).
            batch_idx (int): Batch index.
        """
        # Clear brain state
        self.init_brain_state(batch_size=len(batch["input_ids"]))

        # Initialize loss
        loss = 0

        input_ids = batch["input_ids"].transpose(0, 1)

        # Loop over context windows
        for context_window in input_ids:
            # Get input and target
            input = context_window[:, :-1]
            target = context_window[:, -1]

            # Forward pass
            output = self({"textual_input": input})

            # Calculate loss
            loss += F.cross_entropy(output["textual_output"], target)

        # Average loss
        loss /= len(batch["input_ids"])

        self.log("loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (torch.Tensor): Batch of shape (batch_size, seq_len).
            batch_idx (int): Batch index.
        """
        # Clear brain state
        self.init_brain_state(batch_size=len(batch["input_ids"]))

        # Initialize loss
        loss = 0

        input_ids = batch["input_ids"].transpose(0, 1)

        # Loop over context windows
        for context_window in input_ids:
            # Get input and target
            input = context_window[:, :-1]
            target = context_window[:, -1]

            # Forward pass
            output = self({"textual_input": input})

            # Calculate loss
            loss += F.cross_entropy(output["textual_output"], target)

        # Average loss
        loss /= len(batch["input_ids"])

        self.log("val_loss", loss)

        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure optimizers.

        Returns:
            OptimizerLRScheduler: Optimizer and learning rate scheduler.
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.1, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

