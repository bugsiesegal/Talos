from abc import ABC, abstractmethod

import lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputModule(ABC, nn.Module):
    def __init__(self, memory_state_size):
        super(InputModule, self).__init__()
        self.memory_state_size = memory_state_size

    @abstractmethod
    def forward(self, input_data):
        pass


class OutputModule(ABC, nn.Module):
    def __init__(self, memory_state_size):
        super(OutputModule, self).__init__()
        self.memory_state_size = memory_state_size

    @abstractmethod
    def forward(self, memory_state):
        pass


class PipeInModule(nn.Module):
    def __init__(self, brain_state_size, memory_state_size):
        super(PipeInModule, self).__init__()

        # Parameters
        self.brain_state_size = brain_state_size
        self.memory_state_size = memory_state_size

        # Layers
        self.fc1 = nn.Linear(memory_state_size, brain_state_size)

    def forward(self, memory_state, brain_state):
        return brain_state + torch.mean(self.fc1(memory_state), axis=1)


class PipeOutModule(nn.Module):
    def __init__(self, brain_state_size, memory_state_size):
        super(PipeOutModule, self).__init__()

        # Parameters
        self.brain_state_size = brain_state_size
        self.memory_state_size = memory_state_size

        # Layers
        self.fc1 = nn.Linear(brain_state_size, memory_state_size)

    def forward(self, brain_state):
        return self.fc1(brain_state)


class CognitiveModule(nn.Module):
    def __init__(self, brain_state_size, num_layers=3, num_heads=8, dim_feedforward=2048, dropout=0.1):
        super(CognitiveModule, self).__init__()

        # Parameters
        self.brain_state_size = brain_state_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout

        # Transformer Layer
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=brain_state_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='relu',
            bias=False
        )

        # Transformer
        self.transformer = nn.TransformerEncoder(
            encoder_layer=self.transformer_layer,
            num_layers=num_layers,
        )

    def forward(self, brain_state):
        return self.transformer(brain_state)


class TextInputModule(InputModule):
    def __init__(
            self,
            memory_state_size,
            vocab_size,
            num_heads=8,
            dim_feedforward=2048,
            num_layers=16,
            dropout=0.1,
            bias=True,
    ):
        super(TextInputModule, self).__init__(memory_state_size)

        # Embedding
        self.embedding = nn.Embedding(vocab_size, memory_state_size)

        # Transformer Layer
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=memory_state_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=bias,
        )

        # Transformer
        self.transformer = nn.TransformerDecoder(
            decoder_layer=self.transformer_layer,
            num_layers=num_layers,
        )

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        # Embedding
        embedded_input = self.embedding(input_data)

        # Generate mask
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(embedded_input.size(0)).to(
            embedded_input.device)

        # Transformer
        return self.transformer(embedded_input, torch.zeros_like(embedded_input), tgt_mask=tgt_mask)


class TextOutputModule(OutputModule):
    def __init__(
            self,
            memory_state_size,
            vocab_size,
            num_heads=8,
            dim_feedforward=2048,
            num_layers=16,
            dropout=0.1,
            bias=True,
    ):
        super(TextOutputModule, self).__init__(memory_state_size)

        # Transformer Layer
        self.transformer_layer = nn.TransformerDecoderLayer(
            d_model=memory_state_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            bias=bias,
        )

        # Transformer
        self.transformer = nn.TransformerDecoder(
            decoder_layer=self.transformer_layer,
            num_layers=num_layers,
        )

        # Linear
        self.linear = nn.Linear(memory_state_size, vocab_size)

    def forward(self, memory_state: torch.Tensor) -> torch.Tensor:
        # Generate mask
        tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(memory_state.size(0)).to(memory_state.device)

        # Transformer
        transformer_output = self.transformer(memory_state, torch.zeros_like(memory_state), tgt_mask=tgt_mask)

        # Linear
        return self.linear(transformer_output)


class IntegratedMemoryModel(nn.Module):
    def __init__(
            self,
            brain_state_size,
            memory_state_size,
            thinking_iterations=3,
            cognitive_num_heads=8,
            cognitive_dim_feedforward=2048,
            cognitive_num_layers=16,
            cognitive_dropout=0.1,
    ):
        super(IntegratedMemoryModel, self).__init__()

        # Parameters
        self.brain_state_size = brain_state_size
        self.memory_state_size = memory_state_size
        self.thinking_iterations = thinking_iterations

        # Define brain state
        self.brain_state = nn.Parameter(torch.zeros(1, brain_state_size))
        self.brain_state.requires_grad = False

        # Input Modules
        self.input_modules = nn.ModuleDict()

        # Output Modules
        self.output_modules = nn.ModuleDict()

        # Core Modules
        self.pipe_in_module = PipeInModule(brain_state_size, memory_state_size)
        self.pipe_out_module = PipeOutModule(brain_state_size, memory_state_size)
        self.cognitive_module = CognitiveModule(brain_state_size, cognitive_num_layers, cognitive_num_heads,
                                                cognitive_dim_feedforward, cognitive_dropout)

    def reset_brain_state(self, device="cuda") -> None:
        self.brain_state.data = torch.zeros(1, self.brain_state_size, device=device)

    def forward(self, input_data: dict) -> dict:
        # Memory Dict
        memory_data = {}

        # Input loop
        for key, input_module in self.input_modules.items():
            memory_data[key] = input_module(input_data[key])

        # Pipe in
        for memory in memory_data.values():
            self.brain_state.data = self.pipe_in_module(memory, self.brain_state)

        # Cognitive
        for _ in range(self.thinking_iterations):
            self.brain_state.data = self.cognitive_module(self.brain_state)

        # Output Dict
        output_dict = {}

        # Output loop
        for key, output_module in self.output_modules.items():
            output_dict[key] = output_module(self.pipe_out_module(self.brain_state))

        # Return
        return output_dict

    def add_input_module(self, name: str, module: InputModule) -> None:
        if name in self.input_modules:
            raise ValueError(f"Input module with name {name} already exists.")

        self.input_modules[name] = module

    def add_output_module(self, name: str, module: OutputModule) -> None:
        if name in self.output_modules:
            raise ValueError(f"Output module with name {name} already exists.")

        self.output_modules[name] = module

    def get_brain_state(self) -> torch.Tensor:
        return self.brain_state


class IntegratedMemoryModelLightning(pl.LightningModule):
    def __init__(
            self,
            model: IntegratedMemoryModel,
            learning_rate: float = 1e-3,
    ):
        super(IntegratedMemoryModelLightning, self).__init__()

        # Model
        self.model = model

        # Learning rate
        self.learning_rate = learning_rate

    def forward(self, input_data: dict) -> dict:
        return self.model(input_data)

    def training_step(self, batch, batch_idx):
        self.model.reset_brain_state()
        total_loss = 0.0

        for minibatch in batch:
            output = self.model(minibatch["text_input"])
            loss = F.cross_entropy(output['text_output'], minibatch["labels"])
            total_loss += loss

        avg_loss = total_loss / len(batch)
        perplexity = torch.exp(avg_loss)
        self.log('Loss', avg_loss, on_step=True, on_epoch=False)
        self.log('Perplexity', perplexity, on_step=True, on_epoch=False)

        return avg_loss

    def validation_step(self, batch, batch_idx):
        self.model.reset_brain_state()
        total_loss = 0.0

        for minibatch in batch:
            output = self.model({"text_input":minibatch["inputs"].squeeze(0)})
            loss = F.cross_entropy(output['text_output'], minibatch["labels"].squeeze(0))
            total_loss += loss

        avg_loss = total_loss / len(batch)
        perplexity = torch.exp(avg_loss)
        self.log('Val_Loss', avg_loss, on_step=True, on_epoch=False)
        self.log('Val_Perplexity', perplexity, on_step=True, on_epoch=False)

        return avg_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        # Get data
        input_data, target_data = batch

        # Run model
        output = self(input_data)

        return output

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_brain_state(self) -> torch.Tensor:
        return self.model.get_brain_state()

    def reset_brain_state(self, device="cuda") -> None:
        self.model.reset_brain_state(device)
