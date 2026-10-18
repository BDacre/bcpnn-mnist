import argparse
import time

import torch

from bcpnn.config import Config
from bcpnn.data import load_mnist
from bcpnn.network import RateBCPNN
from bcpnn.spiking import SpikingBCPNN


def parse_args():
    p = argparse.ArgumentParser(description="Feed-forward BCPNN on MNIST")
    p.add_argument("--model", choices=["rate", "baseline", "spiking"], default="rate")
    p.add_argument("--epochs", type=int, help="unsupervised epochs")
    p.add_argument("--subset", type=int, help="use only the first N training images")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden-hcs", type=int)
    p.add_argument("--hidden-mcs", type=int)
    p.add_argument("--patch-size", type=int, help="local receptive fields of this side length")
    p.add_argument("--mask-density", type=float, help="random receptive fields covering this fraction of pixels")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--plot", action="store_true", help="save figures to figures/")
    return p.parse_args()


def build_config(args):
    cfg = Config(seed=args.seed)
    if args.model == "baseline":
        cfg.hidden_hcs = 0
    for name in ("hidden_hcs", "hidden_mcs", "patch_size", "mask_density", "batch_size"):
        value = getattr(args, name)
        if value is not None:
            setattr(cfg, name, value)
    if args.epochs is not None:
        cfg.epochs_unsupervised = args.epochs
    return cfg


def main():
    args = parse_args()
    cfg = build_config(args)
    torch.manual_seed(cfg.seed)
    # Near one-hot softmax outputs produce denormals that make matmul very slow.
    torch.set_flush_denormal(True)
    x_train, y_train, x_test, y_test = load_mnist(args.subset)

    net = SpikingBCPNN(cfg) if args.model == "spiking" else RateBCPNN(cfg)

    start = time.perf_counter()
    net.train_unsupervised(x_train, log=print)
    net.train_supervised(x_train, y_train)
    elapsed = time.perf_counter() - start

    print(f"model {args.model}  seed {cfg.seed}  train time {elapsed:.1f}s")
    # Spiking inference simulates every time step, so score a slice of the training set.
    n_eval = 10000 if args.model == "spiking" else len(x_train)
    print(f"train accuracy {net.accuracy(x_train[:n_eval], y_train[:n_eval]):.4f}  (first {min(n_eval, len(x_train))} images)")
    print(f"test accuracy  {net.accuracy(x_test, y_test):.4f}")

    if args.plot:
        from bcpnn import viz

        for path in viz.save_all(net, x_test, y_test, prefix=args.model):
            print(f"saved {path}")


if __name__ == "__main__":
    main()
