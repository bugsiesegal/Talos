import torch
import torch.nn as nn
import torch.nn.functional as F


def select_output_token(logits, strategy="argmax", top_k=None, temperature=1.0):
    """
    Selects an output token based on the provided logits and selection strategy.

    :param logits: Tensor of logits from the model.
    :param strategy: Selection strategy ('argmax', 'sample', 'top_k', 'temperature').
    :param top_k: Parameter for 'top_k' strategy, selects from top k logits.
    :param temperature: Temperature parameter for 'temperature' strategy.
    :return: Selected token index.
    """

    if strategy not in ["argmax", "sample", "top_k", "temperature"]:
        raise ValueError(f"Unknown strategy: {strategy}")

    if strategy == "argmax":
        # Select the token with the highest probability
        return torch.argmax(logits, dim=-1).unsqueeze(-1)

    # Apply softmax to convert logits to probabilities
    probabilities = F.softmax(logits / temperature, dim=-1)

    if strategy == "sample" or strategy == "temperature":
        # Sample based on the probabilities
        return torch.multinomial(probabilities, 1)

    elif strategy == "top_k":
        # Ensure top_k is valid
        if top_k is None or not isinstance(top_k, int) or top_k <= 0:
            top_k = logits.size(-1)
        elif top_k > logits.size(-1):
            raise ValueError("top_k cannot be larger than the number of logits")

        top_k_logits, indices = torch.topk(logits, top_k)
        top_k_probabilities = F.softmax(top_k_logits / temperature, dim=-1)
        sample = torch.multinomial(top_k_probabilities, 1)
        return indices.gather(-1, sample)
