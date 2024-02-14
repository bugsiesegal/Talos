import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod


class OutputModule(nn.Module, ABC):
    def __init__(self, config):
        super(OutputModule, self).__init__()

        self.config = config

        self.device = config.device

    @abstractmethod
    def forward(self, x):
        pass

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)


class TextOutputModule(OutputModule):
    def __init__(self, config):
        super().__init__(config)

        self.linear = nn.Linear(config.embed_dim, config.vocab_size, bias=config.bias)
        self.pooling = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        x = self.pooling(x.permute(0, 2, 1)).squeeze(-1)
        return self.linear(x.reshape(x.shape[0], -1))
