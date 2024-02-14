from dataclasses import dataclass
from torch.optim import lr_scheduler
from functools import partial

@dataclass
class Config:
    # Tokenizer
    vocab_size: int = 32000
    max_seq_len: int = 512
    # Embedding
    embed_dim: int = 768
    # Transformer
    num_heads: int = 12
    num_layers: int = 2
    hidden_dim: int = 1024
    dropout: float = 0.0
    activation: str = 'gelu'
    bias: bool = False
    transformer_type: str = 'encoder'
    context_length: int = 8
    # Modules
    device: str = 'cuda'
    # Cognitive Module
    thinking_steps: int = 5
    think_twice: bool = False
    # Training
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 1e-4
    # Output
    output_dir: str = 'output'
    # Dataset
    huggingface_dataset: str = 'openai/webtext'
    stream: bool = False
    # Learning rate
    lr: float = 1e-4
    learning_rate_scheduler: partial = partial(lr_scheduler.StepLR, step_size=1, gamma=0.9)
