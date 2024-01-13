from abstract_modules import InputModule

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextualInputModule(InputModule):
    """
    Input module for textual input.

    Takes in a sequence of words and turns it into an embedding of shape (batch_size, embedding_size).

    Args:
        embedding_size (int): Size of the embedding.
        vocab_size (int): Size of the vocabulary.
        embedding_dim (int): Dimension of the word embeddings.

    Attributes:
        embedding_size (int): Size of the embedding.
        embedding (nn.Embedding): Embedding layer.
        sequential (nn.Sequential): Sequential layer.
    """

    def __init__(self, embedding_size, vocab_size):
        super().__init__(embedding_size)

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.sequential = nn.Sequential(
            nn.Linear(embedding_size, embedding_size),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input of shape (batch_size, seq_len).

        Returns:
            torch.Tensor: Embedding of shape (batch_size, embedding_size).
        """
        x = self.embedding(x)
        x = self.sequential(x)
        return x
