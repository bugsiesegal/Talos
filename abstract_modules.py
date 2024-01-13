import torch
import torch.nn as nn
import torch.nn.functional as F

from abc import ABC, abstractmethod


class InputModule(nn.Module, ABC):
    """
    Abstract class for input module.

    Takes in a unspecific input and turns it into an embedding of shape (batch_size, embedding_size).

    Args:
        embedding_size (int): Size of the embedding.

    Attributes:
        embedding_size (int): Size of the embedding.
    """

    def __init__(self, embedding_size):
        super().__init__()

        self.embedding_size = embedding_size

    @abstractmethod
    def forward(self, x):
        pass


class OutputModule(nn.Module, ABC):
    """
    Abstract class for output module.

    Takes in an embedding and converts it into an unspecific output.

    Args:
        embedding_size (int): Size of the embedding.

    Attributes:
        embedding_size (int): Size of the embedding.
    """

    def __init__(self, embedding_size):
        super().__init__()

        self.embedding_size = embedding_size

    @abstractmethod
    def forward(self, x):
        pass
