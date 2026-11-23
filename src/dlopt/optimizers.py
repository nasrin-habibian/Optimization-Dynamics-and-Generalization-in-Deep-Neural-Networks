"""Optimizer factory."""
import torch

def get_optimizer(name, parameters, learning_rate, weight_decay=0.0):
    name = name.lower()
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "momentum":
        return torch.optim.SGD(parameters, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if name == "lbfgs":
        return torch.optim.LBFGS(parameters, lr=learning_rate, max_iter=10,
                                 history_size=10, line_search_fn="strong_wolfe")
    raise ValueError(f"Unknown optimizer: {name}")
