from typing import Any

import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from input_module import TextInputModule
from output_module import TextOutputModule
from model import IntegratedMemoryModel
from config import Config
import lightning as pl


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

    def generate(self, input_ids, max_length, eos_token_id=50256, stop_at_eos=True):
        # Function to generate text token by token
        generated_sequence_length = 0

        outputs = []

        for _ in range(max_length):
            generated_sequence_length += 1
            output = self.model({"text": input_ids})
            outputs.append(output['text'])

            # Append the generated token to the input sequence
            input_ids = torch.cat((input_ids, output['text'].argmax(dim=-1).unsqueeze(1)), dim=1)

            # Stop if EOS token is generated
            if output['text'].argmax(dim=-1)[-1] == eos_token_id and stop_at_eos:
                break

            # Maintain input sequence length
            input_ids = input_ids[:, -self.config.context_length:]

        return torch.stack(outputs, axis=1)

    def training_step(self, batch, batch_idx):
        self.model.reset_hidden_state()
        # Training step for the model
        input_ids = torch.vstack(batch['input_ids'])
        unfolded_input_ids = input_ids.unfold(0, self.config.context_length, 1)

        total_loss = 0

        for batch_ids in unfolded_input_ids:
            loss = 1
            # Generate next token and calculate loss
            generated_ids = self.generate(batch_ids, self.num_generated_tokens, stop_at_eos=False)
            for i in range(generated_ids.shape[1]):
                loss *= self.loss(generated_ids[:, i], batch_ids[:, -self.num_generated_tokens + i])
            total_loss += loss

        self.log('train_loss', total_loss / unfolded_input_ids.shape[0])

        return total_loss / unfolded_input_ids.shape[0]

    def validation_step(self, batch, batch_idx):
        self.model.reset_hidden_state()
        # Validation step for the model
        input_ids = torch.vstack(batch['input_ids'])
        unfolded_input_ids = input_ids.unfold(0, self.config.context_length, 1)

        total_loss = 0

        for batch_ids in unfolded_input_ids:
            loss = 1
            # Generate next token and calculate loss
            generated_ids = self.generate(batch_ids, self.num_generated_tokens, stop_at_eos=False)
            for i in range(generated_ids.shape[1]):
                loss *= self.loss(generated_ids[:, i], batch_ids[:, -self.num_generated_tokens + i])
            total_loss += loss

        self.log('val_loss', total_loss / unfolded_input_ids.shape[0])

        return total_loss / unfolded_input_ids.shape[0]

    def test_step(self, batch, batch_idx):
        self.model.reset_hidden_state()
        # Test step for the model
        input_ids = torch.vstack(batch['input_ids'])
        unfolded_input_ids = input_ids.unfold(0, self.config.context_length, 1)

        total_loss = 0

        for batch_ids in unfolded_input_ids:
            loss = 1
            # Generate next token and calculate loss
            generated_ids = self.generate(batch_ids, self.num_generated_tokens, stop_at_eos=False)
            for i in range(generated_ids.shape[1]):
                loss *= self.loss(generated_ids[:, i], batch_ids[:, -self.num_generated_tokens + i])
            total_loss += loss

        self.log('test_loss', total_loss / unfolded_input_ids.shape[0])

        return total_loss / unfolded_input_ids.shape[0]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Configure optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
