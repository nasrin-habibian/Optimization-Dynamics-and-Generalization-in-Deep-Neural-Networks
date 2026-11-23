"""Training loop and optimization diagnostics."""
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def _grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.detach().norm(2).item() ** 2
    return total ** 0.5

def _evaluate(model, x, y, criterion):
    model.eval()
    with torch.no_grad():
        out = model(x)
        return criterion(out, y).item(), (out.argmax(1) == y).float().mean().item()

def train_model(model, optimizer, x_train, y_train, x_test, y_test,
                epochs=80, batch_size=64, device="cpu"):
    model.to(device)
    xtr, ytr = x_train.to(device), y_train.to(device)
    xte, yte = x_test.to(device), y_test.to(device)
    criterion = nn.CrossEntropyLoss()
    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)
    loader = DataLoader(TensorDataset(xtr, ytr),
        batch_size=len(xtr) if is_lbfgs else batch_size,
        shuffle=not is_lbfgs,
        generator=torch.Generator().manual_seed(1234))
    history = []
    previous = [p.detach().clone() for p in model.parameters()]
    for epoch in range(epochs):
        start = time.perf_counter()
        model.train()
        grad_values = []
        for bx, by in loader:
            def closure():
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(bx), by)
                loss.backward()
                return loss
            if is_lbfgs:
                optimizer.step(closure)
                optimizer.zero_grad(set_to_none=True)
                criterion(model(bx), by).backward()
                grad_values.append(_grad_norm(model))
            else:
                loss = closure()
                grad_values.append(_grad_norm(model))
                optimizer.step()
        tr_loss, tr_acc = _evaluate(model, xtr, ytr, criterion)
        te_loss, te_acc = _evaluate(model, xte, yte, criterion)
        update_sq = 0.0
        with torch.no_grad():
            for old, p in zip(previous, model.parameters()):
                update_sq += (p - old).norm(2).item() ** 2
                old.copy_(p)
        history.append({
            "epoch": epoch + 1, "train_loss": tr_loss, "train_accuracy": tr_acc,
            "test_loss": te_loss, "test_accuracy": te_acc,
            "gradient_norm": sum(grad_values) / max(1, len(grad_values)),
            "update_norm": update_sq ** 0.5,
            "epoch_seconds": time.perf_counter() - start
        })
    return history
