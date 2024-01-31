from typing import Any

import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

from input_module import TextInputModule
from output_module import TextOutputModule
from model import IntegratedMemoryModel
from model_utils import select_output_token
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
        self.context_length = config.context_length

        self.loss = torch.nn.CrossEntropyLoss()

        self.learning_rate_scheduler = config.learning_rate_scheduler

        self.lr = config.lr

        self.save_hyperparameters()

    def generate(self, tokens: torch.Tensor, max_length: int = 100, strategy: str = "argmax", top_k: int = None,
                 temperature: float = 1.0, eos_token_id: int = None, output_logits: bool = False
                 ):
        """
        Generate tokens from the model
        :param tokens: Tensor of shape (batch_size, context_length)
        :param max_length: Maximum length of the generated sequence
        :param strategy: Selection strategy ('argmax', 'sample', 'top_k', 'temperature').
        :param top_k: Parameter for 'top_k' strategy, selects from top k logits.
        :param temperature: Temperature parameter for 'temperature' strategy.
        :param eos_token_id: Token id for the end of sequence token.
        :param output_logits: Whether to output logits or not
        :return: Tensor of generated tokens of shape (batch_size, max_length)
        """

        # Outputs
        outputs = []
        # Generated tokens
        generated_tokens = []

        initial_token_length = tokens.shape[-1]

        for i in range(max_length + initial_token_length):
            # Get the context window
            context_window = tokens[:, i:i + self.context_length]
            # Generate the next token
            generated_ids = self.model({"text": context_window})['text']
            # Append the generated token to generated_tokens
            generated_tokens.append(generated_ids)
            # Select the next token
            selected_token = select_output_token(generated_ids, strategy, top_k, temperature)
            if i == tokens.shape[-1]:
                tokens = torch.cat([tokens, selected_token], dim=1)
                outputs.append(selected_token)

            # If eos_token_id is not None and selected_token is equal to eos_token_id, then break
            if eos_token_id is not None and selected_token == eos_token_id:
                break

        if output_logits:
            return torch.cat(outputs, dim=1), torch.cat(generated_tokens[-len(outputs):], dim=1)
        else:
            return torch.cat(outputs, dim=1)

    def training_step(self, batch, batch_idx):
        # Reset the hidden state of the model at the beginning of each training step
        self.model.reset_hidden_state()

        # Stack the input IDs from the batch and create a tensor
        input_ids = torch.vstack(batch['input_ids'])

        # Unfold the input IDs tensor into sequences for training
        unfolded_input_ids = input_ids.unfold(
            0,
            self.config.context_length + self.num_generated_tokens,
            self.num_generated_tokens
        )

        # Initialize variables to accumulate loss and perplexity
        total_loss = 0
        total_perplexity = 0

        # Iterate over each sequence in the unfolded input IDs
        for batch_ids in unfolded_input_ids:
            # Split the batch IDs into input and target sequences
            batch_ids_inputs = batch_ids[:, :-self.num_generated_tokens]
            batch_ids_target = batch_ids[:, -self.num_generated_tokens:].T

            # Initialize the loss and perplexity for the current sequence
            sequence_loss = 1
            sequence_perplexity = 1

            # Iterate over each target ID in the target sequence
            for target_ids in batch_ids_target:
                # Generate the next token and calculate the loss
                generated_ids = self.model({"text": batch_ids_inputs})
                batch_ids_inputs = torch.cat(
                    [batch_ids_inputs, generated_ids['text'].argmax(axis=-1).unsqueeze(1)],
                    dim=1
                )[:, 1:]

                # Update the sequence loss and perplexity
                sequence_loss *= self.loss(generated_ids['text'], target_ids)
                sequence_perplexity *= torch.exp(sequence_loss)

            # Log the loss and perplexity for the current sequence
            self.log('train_loss', sequence_loss)
            self.log('train_perplexity', sequence_perplexity)

            # Accumulate the total loss and perplexity
            total_loss += sequence_loss
            total_perplexity += sequence_perplexity

        # Return the total loss for the batch
        return total_loss

    def validation_step(self, batch, batch_idx):
        # Reset the hidden state of the model at the beginning of each training step
        self.model.reset_hidden_state()

        # Stack the input IDs from the batch and create a tensor
        input_ids = torch.vstack(batch['input_ids'])

        # Unfold the input IDs tensor into sequences for training
        unfolded_input_ids = input_ids.unfold(
            0,
            self.config.context_length + self.num_generated_tokens,
            self.num_generated_tokens
        )

        # Initialize variables to accumulate loss and perplexity
        total_loss = 0
        total_perplexity = 0

        # Iterate over each sequence in the unfolded input IDs
        for batch_ids in unfolded_input_ids:
            # Split the batch IDs into input and target sequences
            batch_ids_inputs = batch_ids[:, :-self.num_generated_tokens]
            batch_ids_target = batch_ids[:, -self.num_generated_tokens:].T

            # Initialize the loss and perplexity for the current sequence
            sequence_loss = 1
            sequence_perplexity = 1

            # Iterate over each target ID in the target sequence
            for target_ids in batch_ids_target:
                # Generate the next token and calculate the loss
                generated_ids = self.model({"text": batch_ids_inputs})
                batch_ids_inputs = torch.cat(
                    [batch_ids_inputs, generated_ids['text'].argmax(axis=-1).unsqueeze(1)],
                    dim=1
                )[:, 1:]

                # Update the sequence loss and perplexity
                sequence_loss *= self.loss(generated_ids['text'], target_ids)
                sequence_perplexity *= torch.exp(sequence_loss)

            # Log the loss and perplexity for the current sequence
            self.log('validation_loss', sequence_loss)
            self.log('validation_perplexity', sequence_perplexity)

            # Accumulate the total loss and perplexity
            total_loss += sequence_loss
            total_perplexity += sequence_perplexity

        # Return the total loss for the batch
        return total_loss

    def test_step(self, batch, batch_idx):
        # Reset the hidden state of the model at the beginning of each training step
        self.model.reset_hidden_state()

        # Stack the input IDs from the batch and create a tensor
        input_ids = torch.vstack(batch['input_ids'])

        # Unfold the input IDs tensor into sequences for training
        unfolded_input_ids = input_ids.unfold(
            0,
            self.config.context_length + self.num_generated_tokens,
            self.num_generated_tokens
        )

        # Initialize variables to accumulate loss and perplexity
        total_loss = 0
        total_perplexity = 0

        # Iterate over each sequence in the unfolded input IDs
        for batch_ids in unfolded_input_ids:
            # Split the batch IDs into input and target sequences
            batch_ids_inputs = batch_ids[:, :-self.num_generated_tokens]
            batch_ids_target = batch_ids[:, -self.num_generated_tokens:].T

            # Initialize the loss and perplexity for the current sequence
            sequence_loss = 1
            sequence_perplexity = 1

            # Iterate over each target ID in the target sequence
            for target_ids in batch_ids_target:
                # Generate the next token and calculate the loss
                generated_ids = self.model({"text": batch_ids_inputs})
                batch_ids_inputs = torch.cat(
                    [batch_ids_inputs, generated_ids['text'].argmax(axis=-1).unsqueeze(1)],
                    dim=1
                )[:, 1:]

                # Update the sequence loss and perplexity
                sequence_loss *= self.loss(generated_ids['text'], target_ids)
                sequence_perplexity *= torch.exp(sequence_loss)

            # Log the loss and perplexity for the current sequence
            self.log('test_loss', sequence_loss)
            self.log('test_perplexity', sequence_perplexity)

            # Accumulate the total loss and perplexity
            total_loss += sequence_loss
            total_perplexity += sequence_perplexity

        # Return the total loss for the batch
        return total_loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        # Configure optimizer and learning rate scheduler
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = self.learning_rate_scheduler(optimizer)

        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
