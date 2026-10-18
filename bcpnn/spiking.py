import torch

from bcpnn.data import batches, encode
from bcpnn.network import RateBCPNN


class SpikingBCPNN(RateBCPNN):
    """Spiking variant of RateBCPNN with the same layers, projections and interface.

    Units emit stochastic spikes, spikes are low-pass filtered into Z traces, and the
    projections' P traces are learned from Z traces instead of from rates. Weights and
    biases are derived from the P traces exactly as in the rate model.
    """

    def __init__(self, cfg):
        if cfg.hidden_hcs <= 0:
            raise ValueError("SpikingBCPNN needs a hidden layer")
        super().__init__(cfg)

    def _spikes(self, prob):
        return torch.bernoulli(self.cfg.spk_pmax * prob, generator=self.gen)

    def _present(self, x, y=None, learn_hidden=False):
        """Present a batch of images for spk_steps steps and return output spike counts.

        learn_hidden: input->hidden learns and the output layer is not simulated.
        y given: the output is clamped to the label and the readout learns.
        Otherwise the output runs freely from the hidden spikes.
        """
        cfg = self.cfg
        rate = encode(x)
        b = rate.shape[0]
        k = 1.0 / cfg.spk_tau_z
        z_in = torch.zeros_like(rate)
        z_hid = torch.zeros(b, self.in_hid.n_post)
        z_out = torch.zeros(b, cfg.n_classes)
        counts = torch.zeros(b, cfg.n_classes)
        if y is not None:
            target = torch.nn.functional.one_hot(y, cfg.n_classes).float()
        n_learn = len(range(cfg.spk_settle + 1, cfg.spk_steps + 1, cfg.spk_learn_every))

        for t in range(1, cfg.spk_steps + 1):
            # Z traces start at zero each presentation; dividing by `norm` removes
            # the start-up bias so they estimate rates from the first step on.
            norm = cfg.spk_pmax * (1.0 - (1.0 - k) ** t)
            learn = t > cfg.spk_settle and (t - cfg.spk_settle - 1) % cfg.spk_learn_every == 0

            z_in.lerp_(self._spikes(rate), k)
            hid = self.in_hid.forward(z_in / norm)
            z_hid.lerp_(self._spikes(hid), k)

            if learn_hidden:
                if learn:
                    self.in_hid.update(z_in / norm, z_hid / norm, alpha=self.in_hid.alpha / n_learn)
            elif y is not None:
                z_out.lerp_(self._spikes(target), k)
                if learn:
                    self.readout.update(
                        z_hid / norm, z_out / norm, running_average=True, alpha=self.readout.alpha / n_learn
                    )
            else:
                out = self.readout.forward(z_hid / norm)
                # Tiny rate term only breaks ties between equal spike counts.
                counts += self._spikes(out) + 1e-4 * out
        return counts

    @torch.no_grad()
    def train_unsupervised(self, x, epochs=None, log=None):
        cfg = self.cfg
        epochs = cfg.epochs_unsupervised if epochs is None else epochs
        n = x.shape[0]
        if cfg.init_from_data > 0 and self.in_hid.n_updates == 0:
            sample = x[torch.randperm(n, generator=self.gen)[:10000]]
            self.in_hid.seed_from_data(encode(sample), cfg.init_from_data, self.gen)
        for epoch in range(epochs):
            for idx in batches(n, cfg.batch_size, self.gen):
                self._present(x[idx], learn_hidden=True)
            if log is not None:
                log(f"unsupervised epoch {epoch + 1}/{epochs}  usage entropy {self.usage_entropy(x[:5000]):.3f}")

    @torch.no_grad()
    def train_supervised(self, x, y, epochs=None):
        cfg = self.cfg
        epochs = cfg.epochs_supervised if epochs is None else epochs
        for _ in range(epochs):
            for idx in batches(x.shape[0], cfg.batch_size, self.gen):
                self._present(x[idx], y=y[idx])

    @torch.no_grad()
    def predict(self, x, batch_size=1000):
        out = []
        for idx in batches(x.shape[0], batch_size, shuffle=False):
            out.append(self._present(x[idx]).argmax(-1))
        return torch.cat(out)
