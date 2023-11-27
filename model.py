import lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F


class TextEncoderModule(nn.Module):
    """
    Text encoder model

    This model takes in a sequence of tokens and outputs a vector representation of the sequence.

    Parameters
    ----------
    embedding_dim : int
        The dimension of the embedding layer
    vocab_size : int
        The size of the vocabulary
    num_heads : int
        The number of heads in the multi-head attention layer
    num_layers : int
        The number of layers in the transformer
    dim_feedforward : int
        The dimension of the feedforward layer in the transformer
    dropout : float
        The dropout probability
    brain_size : int
        The dimension of the output vector
    """

    def __init__(self, embedding_dim: int, vocab_size: int,
                 num_heads: int = 8, num_layers: int = 12,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1, brain_size: int = 512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, dropout=dropout,
                                                            dim_feedforward=dim_feedforward,
                                                            batch_first=True)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)

        self.fc = nn.Linear(embedding_dim, brain_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x

    def to(self, device):
        self.embedding = self.embedding.to(device)
        self.transformer_layer = self.transformer_layer.to(device)
        self.transformer = self.transformer.to(device)
        self.fc = self.fc.to(device)
        return super().to(device)


class InPipingModule(nn.Module):
    """
    In-piping model

    This model takes in a vector representation and a brain state vector and outputs a new brain state vector.

    Parameters
    ----------
    brain_size : int
        The dimension of the input and output vectors
    pooling : str
        The pooling method to use. Either 'max' or 'avg'
    """

    def __init__(self, brain_size: int = 512, pooling: str = 'max'):
        super().__init__()
        self.fc = nn.Linear(brain_size, brain_size)

        if pooling == 'max':
            self.pooling = nn.AdaptiveMaxPool1d(1)
        elif pooling == 'avg':
            self.pooling = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError(f'Pooling {pooling} not supported')

    def forward(self, x, brain_state):
        x = self.fc(x)
        x = self.pooling(x.permute(0, 2, 1)).squeeze(2)
        x = x + brain_state
        return x

    def to(self, device):
        self.fc = self.fc.to(device)
        self.pooling = self.pooling.to(device)
        return super().to(device)


class CognitiveModule(nn.Module):
    """
    Cognitive model

    This model takes in a brain state vector and outputs a new brain state vector.

    Parameters
    ----------
    brain_size : int
        The dimension of the input and output vectors
    """
    def __init__(self, brain_size: int = 512, num_heads: int = 8,
                 dim_feedforward: int = 2048, encoder_layers: int = 6, decoder_layers: int = 6,
                 dropout: float = 0.1):
        super().__init__()

        self.transformer = nn.Transformer(d_model=brain_size, nhead=num_heads, dropout=dropout,
                                          num_encoder_layers=encoder_layers, num_decoder_layers=decoder_layers,
                                          dim_feedforward=dim_feedforward, batch_first=True)

        self.fc = nn.Linear(brain_size, brain_size)

    def forward(self, x):
        x = self.transformer(x, x)
        x = self.fc(x)
        return x

    def to(self, device):
        self.transformer = self.transformer.to(device)
        self.fc = self.fc.to(device)
        return super().to(device)


class OutPipingModule(nn.Module):
    """
    Out-piping model

    This model takes in a brain state vector and outputs a vector representation.

    Parameters
    ----------
    brain_size : int
        The dimension of the input and output vectors
    """
    def __init__(self, brain_size: int = 512):
        super().__init__()
        self.fc = nn.Linear(brain_size, brain_size)

    def forward(self, x):
        x = self.fc(x)
        return x

    def to(self, device):
        self.fc = self.fc.to(device)
        return super().to(device)


class TextDecoderModule(nn.Module):
    """
    Text decoder model

    This model takes in a vector representation and outputs a sequence of tokens.

    Parameters
    ----------
    brain_size : int
        The dimension of the input and output vectors
    vocab_size : int
        The size of the vocabulary
    num_layers : int
        The number of layers in the transformer
    num_heads : int
        The number of heads in the multi-head attention layer
    dim_feedforward : int
        The dimension of the feedforward layer in the transformer
    dropout : float
        The dropout probability
    """
    def __init__(self, brain_size: int, vocab_size: int, num_layers: int = 6,
                 num_heads: int = 8, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()

        # Transformer Decoder Layer
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=brain_size, nhead=num_heads,
                                                                    dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)

        # Linear layer to map to vocabulary
        self.fc = nn.Linear(brain_size, vocab_size)

    def forward(self, x):
        x = self.transformer_decoder(x, x)
        x = self.fc(x)
        return F.log_softmax(x, dim=-1)

    def to(self, device):
        self.transformer_decoder_layer = self.transformer_decoder_layer.to(device)
        self.transformer_decoder = self.transformer_decoder.to(device)
        self.fc = self.fc.to(device)
        return super().to(device)


class IntegratedMemoryNeuralNetwork(nn.Module):
    def __init__(self, vocab_size: int, brain_size: int = 512, embedding_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, pooling: str = 'max'):
        super().__init__()

        # Initializing the modules
        self.text_encoder = TextEncoderModule(embedding_dim, vocab_size, num_heads, 2,
                                              dim_feedforward, dropout, brain_size)
        self.in_piping = InPipingModule(brain_size, pooling)
        self.cognitive = CognitiveModule(brain_size, num_heads, dim_feedforward, num_layers, num_layers, dropout)
        self.out_piping = OutPipingModule(brain_size)
        self.text_decoder = TextDecoderModule(brain_size, vocab_size, 2, num_heads,
                                              dim_feedforward, dropout)

        # Initializing the brain state
        self.brain_state = torch.zeros(1, brain_size)

    def forward(self, x):
        if x.shape[0] != self.brain_state.shape[0]:
            self.brain_state = torch.zeros(x.shape[0], self.brain_state.shape[1], device=x.device)

        # Text encoding
        encoded_text = self.text_encoder(x)

        # In-piping: updating the brain state
        self.brain_state = self.in_piping(encoded_text, self.brain_state)

        # Cognitive processing
        cognitive_output = self.cognitive(self.brain_state)

        # Out-piping
        piped_output = self.out_piping(cognitive_output)

        # Text decoding
        decoded_text = self.text_decoder(piped_output)

        return decoded_text

    def to(self, device):
        self.text_encoder = self.text_encoder.to(device)
        self.in_piping = self.in_piping.to(device)
        self.cognitive = self.cognitive.to(device)
        self.out_piping = self.out_piping.to(device)
        self.text_decoder = self.text_decoder.to(device)
        self.brain_state = self.brain_state.to(device)
        return super().to(device)

    def reset_state(self):
        self.brain_state = torch.zeros(1, self.brain_state.shape[1], device=self.brain_state.device)


class PLIntegratedMemoryNeuralNetwork(pl.LightningModule):
    def __init__(self, vocab_size, brain_size, embedding_dim, num_heads, num_layers, dim_feedforward, dropout, learning_rate, context_length):
        super().__init__()
        self.model = IntegratedMemoryNeuralNetwork(vocab_size, brain_size, embedding_dim, num_heads, num_layers, dim_feedforward, dropout)
        self.learning_rate = learning_rate
        self.context_length = context_length
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        total_loss = 0
        num_steps = 0
        for i in range(1, len(batch['input_ids'])):
            start_idx = max(0, i - self.context_length)
            inputs = torch.vstack(batch['input_ids'][start_idx:i]).T
            targets = batch['input_ids'][i]
            outputs = self(inputs)
            loss = self.criterion(outputs, targets)
            total_loss += loss
            num_steps += 1
            self.log('step_loss', loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        avg_loss = total_loss / num_steps
        self.log('avg_train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.model.reset_state()
        return avg_loss

    def validation_step(self, batch, batch_idx):
        total_val_loss = 0
        num_steps = 0
        for i in range(1, len(batch['input_ids'])):
            start_idx = max(0, i - self.context_length)
            inputs = torch.vstack(batch['input_ids'][start_idx:i]).T
            targets = batch['input_ids'][i]
            outputs = self(inputs)
            val_loss = self.criterion(outputs, targets)
            if not torch.isnan(val_loss):
                total_val_loss += val_loss
                num_steps += 1
        avg_val_loss = total_val_loss / num_steps if num_steps > 0 else torch.tensor(0.0)
        self.log('avg_val_loss', avg_val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def on_train_epoch_end(self):
        self.model.reset_state()
