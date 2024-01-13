from abstract_modules import OutputModule

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextualOutputModule(OutputModule):
    """
    Output module for textual output.

    Takes in an embedding and converts it into a sequence of words.

    Args:
        embedding_size (int): Size of the embedding.
        vocab_size (int): Size of the vocabulary.

    Attributes:
        embedding_size (int): Size of the embedding.
        sequential (nn.Sequential): Sequential layer.
    """

    def __init__(self, embedding_size, vocab_size):
        super().__init__(embedding_size)

        self.sequential = nn.Sequential(
            nn.Linear(embedding_size, vocab_size),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input of shape (batch_size, embedding_size).

        Returns:
            torch.Tensor: Embedding of shape (batch_size, seq_len).
        """
        x = self.sequential(x)
        return x
