from dataclasses import dataclass


@dataclass
class Config:
    # Tokenizer
    vocab_size: int = 32000
    max_seq_len: int = 512
    # Embedding
    embed_dim: int = 768
    # Transformer
    num_heads: int = 12
    num_layers: int = 12
    hidden_dim: int = 3072
    dropout: float = 0.1
    activation: str = 'gelu'
    bias: bool = True
    transformer_type: str = 'encoder'
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
