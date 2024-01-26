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

        self.linear = nn.Linear(config.embed_dim * config.context_length, config.vocab_size, bias=config.bias)

    def forward(self, x):
        return self.linear(x.reshape(x.shape[0], -1))
