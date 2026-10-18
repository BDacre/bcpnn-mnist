import torch

from bcpnn.data import batches, encode
from bcpnn.projection import Projection

N_PIXELS = 784


class RateBCPNN:
    """Rate-based feed-forward BCPNN: input -> hidden (unsupervised) -> output (supervised)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.gen = torch.Generator().manual_seed(cfg.seed)
        n_in = 2 * N_PIXELS

        self.in_hid = None
        n_readout, readout_hc_size = n_in, 2
        if cfg.hidden_hcs > 0:
            self.in_hid = Projection(
                n_in,
                cfg.hidden_hcs,
                cfg.hidden_mcs,
                pre_hc_size=2,
                alpha=cfg.alpha_hidden,
                gain=cfg.gain_hidden,
                bias_gain=cfg.bias_gain_hidden,
                eps=cfg.eps,
                mask=self._make_mask(),
            )
            n_readout, readout_hc_size = self.in_hid.n_post, cfg.hidden_mcs
        self.readout = Projection(
            n_readout,
            1,
            cfg.n_classes,
            pre_hc_size=readout_hc_size,
            alpha=cfg.alpha_output,
            gain=cfg.gain_output,
            bias_gain=cfg.bias_gain_output,
            eps=cfg.eps,
        )

    def _make_mask(self):
        """Fixed receptive field per hidden hypercolumn as a bool (n_in, n_hidden) mask, or None if dense."""
        cfg = self.cfg
        side = int(N_PIXELS**0.5)
        pix = torch.zeros(side, side, cfg.hidden_hcs, dtype=torch.bool)
        if cfg.patch_size > 0:
            size = min(cfg.patch_size, side)
            for h in range(cfg.hidden_hcs):
                r, c = torch.randint(0, side - size + 1, (2,), generator=self.gen).tolist()
                pix[r : r + size, c : c + size, h] = True
        elif cfg.mask_density < 1.0:
            k = max(1, round(cfg.mask_density * N_PIXELS))
            pix = pix.view(N_PIXELS, -1)
            for h in range(cfg.hidden_hcs):
                pix[torch.randperm(N_PIXELS, generator=self.gen)[:k], h] = True
        else:
            return None
        # Expand pixels -> 2 input units, hypercolumns -> minicolumns.
        pix = pix.reshape(N_PIXELS, cfg.hidden_hcs)
        return pix.repeat_interleave(2, dim=0).repeat_interleave(cfg.hidden_mcs, dim=1)

    @torch.no_grad()
    def hidden(self, x):
        """Hidden-layer activity for raw images x (B, 784)."""
        pre = encode(x)
        if self.in_hid is None:
            return pre
        return self.in_hid.forward(pre)

    @torch.no_grad()
    def train_unsupervised(self, x, epochs=None, log=None):
        if self.in_hid is None:
            return
        cfg = self.cfg
        epochs = cfg.epochs_unsupervised if epochs is None else epochs
        n = x.shape[0]
        if cfg.init_from_data > 0 and self.in_hid.n_updates == 0:
            sample = x[torch.randperm(n, generator=self.gen)[:10000]]
            self.in_hid.seed_from_data(encode(sample), cfg.init_from_data, self.gen)
        for epoch in range(epochs):
            for idx in batches(n, cfg.batch_size, self.gen):
                pre = encode(x[idx])
                self.in_hid.update(pre, self.in_hid.forward(pre))
            if log is not None:
                log(f"unsupervised epoch {epoch + 1}/{epochs}  usage entropy {self.usage_entropy(x[:5000]):.3f}")

    @torch.no_grad()
    def train_supervised(self, x, y, epochs=None):
        cfg = self.cfg
        epochs = cfg.epochs_supervised if epochs is None else epochs
        n = x.shape[0]
        for _ in range(epochs):
            for idx in batches(n, cfg.batch_size, self.gen):
                target = torch.nn.functional.one_hot(y[idx], cfg.n_classes).float()
                self.readout.update(self.hidden(x[idx]), target, running_average=True)

    @torch.no_grad()
    def predict(self, x, batch_size=1000):
        out = []
        for idx in batches(x.shape[0], batch_size, shuffle=False):
            out.append(self.readout.support(self.hidden(x[idx])).argmax(-1))
        return torch.cat(out)

    def accuracy(self, x, y):
        return (self.predict(x) == y).float().mean().item()

    @torch.no_grad()
    def usage_entropy(self, x, batch_size=1000):
        """Mean normalised entropy of minicolumn usage per hidden hypercolumn (1 = uniform)."""
        if self.in_hid is None:
            return float("nan")
        usage = torch.zeros(self.in_hid.n_post)
        for idx in batches(x.shape[0], batch_size, shuffle=False):
            usage += self.hidden(x[idx]).sum(0)
        p = usage.view(self.cfg.hidden_hcs, self.cfg.hidden_mcs)
        p = p / p.sum(-1, keepdim=True)
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
        return (ent / torch.log(torch.tensor(float(self.cfg.hidden_mcs)))).mean().item()
