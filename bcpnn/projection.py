import torch


class Projection:
    """BCPNN projection from a pre-synaptic population onto post-synaptic hypercolumns.

    Holds the probability traces p_i (pre), p_j (post) and p_ij (joint) and derives
    weights w_ij = log(p_ij / (p_i p_j)) and biases b_j = log(p_j) from them.
    """

    def __init__(
        self,
        n_pre,
        post_hcs,
        post_mcs,
        *,
        pre_hc_size,
        alpha,
        gain=1.0,
        bias_gain=1.0,
        eps=1e-6,
        mask=None,
    ):
        self.n_pre = n_pre
        self.post_hcs = post_hcs
        self.post_mcs = post_mcs
        self.n_post = post_hcs * post_mcs
        self.alpha = alpha
        self.gain = gain
        self.bias_gain = bias_gain
        self.eps = eps
        # mask: bool (n_pre, n_post), True where a connection exists.
        self.mask = mask

        self.p_i = torch.full((n_pre,), 1.0 / pre_hc_size)
        self.p_j = torch.full((self.n_post,), 1.0 / post_mcs)
        self.p_ij = torch.outer(self.p_i, self.p_j)
        self.n_updates = 0
        self._w = None

    def seed_from_data(self, pre, strength, generator=None):
        """Start each post unit as a prototype of one randomly chosen pre-synaptic sample.

        strength in (0, 1] blends between the mean pre activity (0) and the sample (1).
        """
        self.p_i = pre.mean(0)
        idx = torch.randint(pre.shape[0], (self.n_post,), generator=generator)
        proto = torch.lerp(self.p_i.expand(self.n_post, -1), pre[idx], strength)
        self.p_ij = proto.T * self.p_j
        self._w = None

    def weights(self):
        if self._w is None:
            p_i = self.p_i.clamp_min(self.eps)
            p_j = self.p_j.clamp_min(self.eps)
            p_ij = self.p_ij.clamp_min(self.eps**2)
            w = torch.log(p_ij / torch.outer(p_i, p_j))
            if self.mask is not None:
                w = w * self.mask
            self._w = w
        return self._w

    def bias(self):
        return torch.log(self.p_j.clamp_min(self.eps))

    def support(self, pre):
        return self.bias_gain * self.bias() + pre @ self.weights()

    def activate(self, support):
        """Softmax within each hypercolumn."""
        s = support.view(-1, self.post_hcs, self.post_mcs)
        return torch.softmax(self.gain * s, dim=-1).view(-1, self.n_post)

    def forward(self, pre):
        return self.activate(self.support(pre))

    def update(self, pre, post, running_average=False, alpha=None):
        """Move the traces towards the batch statistics of (pre, post).

        With running_average=True the first 1/alpha updates use a plain running
        mean, so the traces are not biased towards their initial values.
        """
        self.n_updates += 1
        a = self.alpha if alpha is None else alpha
        if running_average:
            a = max(a, 1.0 / self.n_updates)
        b = pre.shape[0]
        self.p_i.lerp_(pre.mean(0), a)
        self.p_j.lerp_(post.mean(0), a)
        self.p_ij.lerp_(pre.T @ post / b, a)
        self._w = None
