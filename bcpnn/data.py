from pathlib import Path

import torch
from torchvision.datasets import MNIST

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_mnist(subset=None, root=DATA_DIR):
    """Return (x_train, y_train, x_test, y_test) with pixels in [0, 1], flattened to 784."""
    out = []
    for train in (True, False):
        ds = MNIST(root=str(root), train=train, download=True)
        x = ds.data.reshape(len(ds), -1).float() / 255.0
        y = ds.targets.clone()
        if subset is not None and train:
            x, y = x[:subset], y[:subset]
        out += [x, y]
    return tuple(out)


def encode(x):
    """Encode each pixel as a 2-unit hypercolumn [x, 1 - x]; (B, 784) -> (B, 1568)."""
    return torch.stack((x, 1.0 - x), dim=-1).reshape(x.shape[0], -1)


def batches(n, batch_size, generator=None, shuffle=True):
    """Yield index tensors covering range(n)."""
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for start in range(0, n, batch_size):
        yield idx[start : start + batch_size]
