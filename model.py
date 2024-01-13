import torch
import torch.nn as nn
import torch.nn.functional as F


class CognitiveModule(nn.Module):
    def __init__(self, brain_state_size, num_layers, **kwargs):
        """
        Cognitive module. This module is the core of the model and is responsible for the cognitive processes. It takes
        in a brain state and outputs a new brain state. The brain state is a tensor of shape (batch_size, brain_state_size).
        The output brain state has the same shape as the input brain state. The number of layers is the number of
        transformer layers in the transformer decoder. The kwargs are passed to the transformer decoder layer.

        Args:
            brain_state_size (int): Size of the brain state.
            num_layers (int): Number of transformer layers.
            **kwargs: Keyword arguments passed to the transformer decoder layer.

        Attributes:
            brain_state_size (int): Size of the brain state.
            transformer_layer (nn.TransformerDecoderLayer): Transformer decoder layer.
            transformer (nn.TransformerDecoder): Transformer decoder.
        """
        super().__init__()

        self.brain_state_size = brain_state_size

        self.transformer_layer = nn.TransformerDecoderLayer(d_model=brain_state_size, **kwargs)

        self.transformer = nn.TransformerDecoder(self.transformer_layer, num_layers=num_layers)

        self.layer_norm = nn.LayerNorm(brain_state_size)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Brain state of shape (batch_size, brain_state_size).
        """
        x = self.transformer(x, torch.zeros_like(x))
        x = self.layer_norm(x)
        return x


class EmbeddingIntegrationModule(nn.Module):
    def __init__(self, embedding_size, brain_state_size):
        """
        Embedding integration module. This module takes in an embedding and a brain state and outputs a new brain state.
        The embedding is a tensor of shape (batch_size, embedding_size). The brain state is a tensor of shape
        (batch_size, brain_state_size). The output brain state has the same shape as the input brain state.

        Args:
            embedding_size (int): Size of the embedding.
            brain_state_size (int): Size of the brain state.

        Attributes:
            embedding_size (int): Size of the embedding.
            brain_state_size (int): Size of the brain state.
            linear (nn.Linear): Linear layer.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.brain_state_size = brain_state_size

        self.linear = nn.Linear(embedding_size, brain_state_size)

    def forward(self, x, y):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Embedding of shape (batch_size, embedding_size).
            y (torch.Tensor): Brain state of shape (batch_size, brain_state_size).

        Returns:
            torch.Tensor: Brain state of shape (batch_size, brain_state_size).
        """
        x = torch.mean(x, dim=1)
        x = self.linear(x)
        x = x + y
        return x


class EmbeddingExtractionModule(nn.Module):
    def __init__(self, embedding_size, brain_state_size):
        """
        Embedding extraction module. This module takes in a brain state and outputs an embedding. The brain state is a
        tensor of shape (batch_size, brain_state_size). The output embedding has the shape (batch_size, embedding_size).

        Args:
            embedding_size (int): Size of the embedding.
            brain_state_size (int): Size of the brain state.

        Attributes:
            embedding_size (int): Size of the embedding.
            brain_state_size (int): Size of the brain state.
            linear (nn.Linear): Linear layer.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.brain_state_size = brain_state_size

        self.linear = nn.Linear(brain_state_size, embedding_size)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Brain state of shape (batch_size, brain_state_size).

        Returns:
            torch.Tensor: Embedding of shape (batch_size, embedding_size).
        """
        x = self.linear(x)
        return x


class IntegratedMemoryModel(nn.Module):
    def __init__(self, embedding_size, brain_state_size, num_layers, thinking_iterations, **kwargs):
        """
        Integrated memory model. This model is a combination of the embedding extraction module, the embedding
        integration module and the cognitive module. It takes in an input and outputs an output. The input is a
        dictionary of tensors. The tensors are the inputs for the input modules. The output is a dictionary of
        tensors. The tensors are the outputs of the output modules. The input modules are stored in the input_modules
        dictionary. The output modules are stored in the output_modules dictionary. The brain state is a tensor of
        shape (batch_size, brain_state_size).

        Args:
            embedding_size (int): Size of the embedding.
            brain_state_size (int): Size of the brain state.
            num_layers (int): Number of transformer layers.
            thinking_iterations (int): Number of thinking iterations.
            **kwargs: Keyword arguments passed to the transformer decoder layer.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.brain_state_size = brain_state_size
        self.thinking_iterations = thinking_iterations

        self.embedding_extraction_module = EmbeddingExtractionModule(embedding_size, brain_state_size)
        self.embedding_integration_module = EmbeddingIntegrationModule(embedding_size, brain_state_size)
        self.cognitive_module = CognitiveModule(brain_state_size, num_layers, **kwargs)

        self.input_modules = nn.ModuleDict()
        self.output_modules = nn.ModuleDict()

        self.brain_state = None

    def add_input_module(self, name, module):
        """
        Adds an input module.

        Args:
            name (str): Name of the input module.
            module (nn.Module): Input module.
        """

        self.input_modules[name] = module

    def add_output_module(self, name, module):
        """
        Adds an output module.

        Args:
            name (str): Name of the output module.
            module (nn.Module): Output module.
        """

        self.output_modules[name] = module

    def init_brain_state(self, batch_size: int) -> None:
        """
        Initializes the brain state.

        Args:
            batch_size (int): Batch size.
        """

        self.brain_state = torch.zeros(batch_size, self.brain_state_size, device=self.device)

    def reset_brain_state(self) -> None:
        """
        Resets the brain state.
        """

        self.brain_state = None

    def to(self, *args, **kwargs) -> "IntegratedMemoryModel":
        """
        Moves the model to the specified device and changes the data type if dtype is specified.
        This method is compatible with PyTorch's nn.Module to() method.

        Args:
            *args: Variable length argument list for device and/or dtype.
            **kwargs: Arbitrary keyword arguments for device and/or dtype.

        Returns:
            IntegratedMemoryModel: The model itself.
        """

        super().to(*args, **kwargs)

        device, dtype = None, None

        # Handling args and kwargs to extract device and dtype
        if len(args) > 0:
            if isinstance(args[0], torch.device):
                device = args[0]
            elif isinstance(args[0], torch.dtype):
                dtype = args[0]

        device = kwargs.get("device", device)
        dtype = kwargs.get("dtype", dtype)

        # Apply to() to all sub-modules with appropriate device and dtype
        for module in self.input_modules.values():
            if device and dtype:
                module.to(device, dtype)
            elif device:
                module.to(device)
            elif dtype:
                module.to(dtype=dtype)

        for module in self.output_modules.values():
            if device and dtype:
                module.to(device, dtype)
            elif device:
                module.to(device)
            elif dtype:
                module.to(dtype=dtype)

        self.embedding_extraction_module.to(*args, **kwargs)
        self.embedding_integration_module.to(*args, **kwargs)
        self.cognitive_module.to(*args, **kwargs)

        if self.brain_state is not None:
            if device and dtype:
                self.brain_state = self.brain_state.to(device, dtype)
            elif device:
                self.brain_state = self.brain_state.to(device)
            elif dtype:
                self.brain_state = self.brain_state.to(dtype=dtype)

        return self

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x (dict): Input dictionary.

        Returns:
            dict: Output dictionary.
        """

        # Loop over input modules and pass the inputs through them to get the embeddings
        # Then integrate the embeddings into the brain state
        for name, module in self.input_modules.items():
            embedding = module(x[name])
            self.brain_state = self.embedding_integration_module(embedding, self.brain_state)

        # Pass the brain state through the cognitive module thinking_iterations times
        for _ in range(self.thinking_iterations):
            self.brain_state = self.cognitive_module(self.brain_state)

        # Loop over output modules and pass the brain state through them to get the outputs
        outputs = {}
        for name, module in self.output_modules.items():
            outputs[name] = module(self.embedding_extraction_module(self.brain_state))

        return outputs
