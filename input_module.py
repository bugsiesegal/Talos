import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod


class InputModule(nn.Module, ABC):
    def __init__(self, config):
        super(InputModule, self).__init__()

        self.config = config

        self.device = config.device

    @abstractmethod
    def forward(self, x):
        pass

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)


class TextInputModule(InputModule):
    def __init__(self, config):
        super().__init__(config)

        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.position_embedding = nn.Embedding(config.max_seq_len, config.embed_dim)
        self.linear = nn.Linear(config.embed_dim, config.embed_dim, bias=config.bias)

    def forward(self, x):
        device = x.device
        b, t = x.shape
        pos = torch.arange(0, t, dtype=torch.long, device=device)

        x = self.token_embedding(x)
        pos = self.position_embedding(pos)

        x = x + pos

        return self.linear(x)
