from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def receptive_fields(net, path, n_hcs=4, n_mcs=16):
    """Plot input->hidden weights of the 'pixel on' units for a few minicolumns per hypercolumn."""
    cfg = net.cfg
    n_hcs, n_mcs = min(n_hcs, cfg.hidden_hcs), min(n_mcs, cfg.hidden_mcs)
    w = net.in_hid.weights()[0::2].T.reshape(cfg.hidden_hcs, cfg.hidden_mcs, 28, 28)
    fig, axes = plt.subplots(n_hcs, n_mcs, figsize=(n_mcs, n_hcs), squeeze=False)
    for h in range(n_hcs):
        # Scale to the positive weights; negatives are far larger and would wash them out.
        lim = w[h, :n_mcs].max().item()
        for m in range(n_mcs):
            axes[h, m].imshow(w[h, m], cmap="RdBu_r", vmin=-lim, vmax=lim)
            axes[h, m].axis("off")
    fig.suptitle("Input -> hidden weights (rows: hypercolumns, columns: minicolumns)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def confusion_matrix(net, x, y, path):
    n = net.cfg.n_classes
    pred = net.predict(x)
    cm = torch.zeros(n, n, dtype=torch.long)
    cm.index_put_((y, pred), torch.ones_like(y), accumulate=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(n):
        for j in range(n):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=7, color=color)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Test confusion matrix (accuracy {cm.diag().sum().item() / cm.sum().item():.2%})")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def save_all(net, x, y, prefix):
    FIG_DIR.mkdir(exist_ok=True)
    paths = [FIG_DIR / f"{prefix}_confusion.png"]
    confusion_matrix(net, x, y, paths[0])
    if net.in_hid is not None:
        paths.append(FIG_DIR / f"{prefix}_receptive_fields.png")
        receptive_fields(net, paths[1])
    return paths
