import torch
from sklearn.metrics import f1_score


def calculate_perplexity(loss):
    return torch.exp(loss)


def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == targets).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def calculate_f1(outputs, targets):
    _, predicted = torch.max(outputs, dim=1)
    # Convert tensors to numpy arrays for sklearn
    predicted = predicted.cpu().numpy()
    targets = targets.cpu().numpy()
    return f1_score(targets, predicted, average='weighted')
