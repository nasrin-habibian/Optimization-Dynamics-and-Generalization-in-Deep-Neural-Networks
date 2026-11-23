"""Leakage-safe two-moons data generation."""
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch

def load_dataset(n_samples=1200, noise=0.2, test_size=0.25, random_state=42):
    x, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    xtr, xte, ytr, yte = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scaler = StandardScaler()
    xtr = scaler.fit_transform(xtr)
    xte = scaler.transform(xte)
    return (torch.tensor(xtr, dtype=torch.float32),
            torch.tensor(xte, dtype=torch.float32),
            torch.tensor(ytr, dtype=torch.long),
            torch.tensor(yte, dtype=torch.long))
