import sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dlopt.data import load_dataset
from dlopt.models import MLP
from dlopt.optimizers import get_optimizer
from dlopt.training import train_model

def test_data_shapes_and_classes():
    xtr, xte, ytr, yte = load_dataset()
    assert xtr.shape[1] == xte.shape[1] == 2
    assert set(ytr.tolist()) == {0, 1}
    assert set(yte.tolist()) == {0, 1}

def test_optimizer_factory():
    for name in ("sgd", "momentum", "adamw", "lbfgs"):
        model = MLP()
        assert get_optimizer(name, model.parameters(), 0.01) is not None

def test_short_training():
    torch.manual_seed(3)
    xtr, xte, ytr, yte = load_dataset(n_samples=200)
    model = MLP(hidden_dim=8)
    opt = get_optimizer("adamw", model.parameters(), 0.003)
    hist = train_model(model, opt, xtr, ytr, xte, yte, epochs=2, batch_size=32)
    assert len(hist) == 2
    assert "gradient_norm" in hist[-1]
    assert 0 <= hist[-1]["test_accuracy"] <= 1
