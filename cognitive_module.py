import torch
import torch.nn as nn
import torch.nn.functional as F


def initialize_transformer(config):
    """Initialize the transformer based on the configuration.

    Args:
        config: Configuration object containing transformer specifications.

    Returns:
        An instance of TransformerEncoder or TransformerDecoder.
    """
    # Dictionary mapping transformer types to their respective initializers
    transformer_types = {
        'encoder': lambda: nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.embed_dim,
                config.num_heads,
                config.hidden_dim,
                config.dropout,
                config.activation,
                config.bias,
                batch_first=True
            ),
            config.num_layers,
            nn.LayerNorm(config.embed_dim)
        ),
        'decoder': lambda: nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                config.embed_dim,
                config.num_heads,
                config.hidden_dim,
                config.dropout,
                config.activation,
                config.bias,
                batch_first=True
            ),
            config.num_layers,
            nn.LayerNorm(config.embed_dim)
        )
    }
    return transformer_types.get(config.transformer_type, lambda: None)()


class TransformerModule(nn.Module):
    def __init__(self, config):
        """Initialize the TransformerModule.

        Args:
            config: A configuration object containing parameters for the module.
        """
        super(TransformerModule, self).__init__()
        self.config = config
        self.device = config.device
        # Initialize the transformer based on the type specified in config
        self.transformer = initialize_transformer(config)

        # If think_twice is enabled in config, create a linear layer for decision-making
        # if config.think_twice:
        #     self.linear = nn.Linear(config.embed_dim, 1, bias=config.bias)

        # # Weight initialization described in nanoGPT by karpathy
        # self.apply(self._init_weights)
        # for pn, p in self.named_parameters():
        #     if pn.endswith('c_proj.weight'):
        #         torch.nn.init.zeros_(p)

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, memory=None):
        # Create a copy of the input tensor to store updates
        x_updated = x.clone()

        for _ in range(self.config.thinking_steps):
            # Run the transformer only on the selected batches
            x_transformed = self.run_transformer(x_updated, memory)
            x_updated = x_transformed.to(x_updated.dtype)

        return x_updated

    def run_transformer(self, x, memory):
        """Run the transformer (encoder or decoder) on the input.

        Args:
            x: Input tensor.
            memory: Memory tensor for decoder.

        Returns:
            The output tensor from the transformer.
        """
        # Process input through the encoder transformer
        if isinstance(self.transformer, nn.TransformerEncoder):
            return self.transformer(x)
        # Process input through the decoder transformer
        elif isinstance(self.transformer, nn.TransformerDecoder):
            # Check if memory is provided for the decoder, and handle accordingly
            if memory is None:
                raise ValueError("Memory must be provided for the TransformerDecoder.")
            return self.transformer(x, memory)
        else:
            raise ValueError(f"Transformer type of {type(self.transformer)} not supported.")

    def think_twice_decision(self, x):
        # Process each batch and return a mask indicating which batches to continue processing
        decision = self.linear(torch.mean(x, dim=1))
        return decision.squeeze() > 0
