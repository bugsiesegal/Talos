import torch
import torch.nn as nn
import torch.nn.functional as F


class EmbeddingMerger(nn.Module):
    def __init__(self, config):
        super(EmbeddingMerger, self).__init__()
        self.config = config
        self.device = config.device
        self.linear = nn.Linear(config.embed_dim * 2, config.embed_dim, bias=config.bias)

    def forward(self, x, y) -> torch.Tensor:
        return self.linear(torch.cat([x, y], dim=-1))

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)


class EmbeddingSplitter(nn.Module):
    def __init__(self, config):
        super(EmbeddingSplitter, self).__init__()
        self.config = config
        self.device = config.device
        self.linear = nn.Linear(config.embed_dim, config.embed_dim * 2, bias=config.bias)

    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # Split the embedding into two parts of equal size
        return self.linear(x).chunk(2, dim=-1)

    def to(self, *args, **kwargs):
        self.device = args[0]
        return super().to(*args, **kwargs)
