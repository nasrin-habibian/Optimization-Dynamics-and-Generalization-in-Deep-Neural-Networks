"""Plot benchmark trajectories and generalization."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def create_plots(runs, histories, output_dir):
    out = Path(output_dir)
    plots = [
        ("train_loss", "loss_curves.png", "Training loss"),
        ("gradient_norm", "gradient_norm_curves.png", "Gradient L2 norm"),
        ("test_accuracy", "test_accuracy_curves.png", "Test accuracy")
    ]
    for metric, filename, ylabel in plots:
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, rows in histories.items():
            ax.plot([r[metric] for r in rows], alpha=0.45, label=label)
        ax.set(xlabel="Epoch", ylabel=ylabel, title=ylabel)
        ax.grid(alpha=0.25)
        if len(histories) <= 12:
            ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / filename, dpi=180)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 5))
    for noise, group in runs.groupby("noise"):
        means = group.groupby("optimizer")["final_test_accuracy"].mean()
        ax.plot(means.index, means.values, marker="o", label=f"noise={noise}")
    ax.set(xlabel="Optimizer", ylabel="Mean test accuracy", ylim=(0, 1),
           title="Generalization across noise levels")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "generalization_by_noise.png", dpi=180)
    plt.close(fig)
