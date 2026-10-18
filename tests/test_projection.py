import torch

from bcpnn.config import Config
from bcpnn.data import encode
from bcpnn.network import RateBCPNN
from bcpnn.projection import Projection
from bcpnn.spiking import SpikingBCPNN


def one_hot_hcs(n, hcs, mcs, gen):
    """Random activity with exactly one active minicolumn per hypercolumn."""
    idx = torch.randint(mcs, (n, hcs), generator=gen)
    return torch.nn.functional.one_hot(idx, mcs).float().view(n, hcs * mcs)


def test_encode_hypercolumns_sum_to_one():
    x = torch.rand(7, 784)
    enc = encode(x)
    assert enc.shape == (7, 1568)
    assert torch.allclose(enc.view(7, 784, 2).sum(-1), torch.ones(7, 784))
    assert torch.equal(enc[:, 0::2], x)


def test_independent_activity_gives_zero_weights():
    gen = torch.Generator().manual_seed(0)
    proj = Projection(8, 2, 4, pre_hc_size=4, alpha=0.01)
    for _ in range(200):
        pre = one_hot_hcs(4096, 2, 4, gen)
        post = one_hot_hcs(4096, 2, 4, gen)
        proj.update(pre, post, running_average=True)
    assert proj.weights().abs().max() < 0.05


def test_correlated_activity_gives_signed_weights():
    gen = torch.Generator().manual_seed(0)
    proj = Projection(4, 1, 4, pre_hc_size=4, alpha=0.01)
    for _ in range(200):
        pre = one_hot_hcs(256, 1, 4, gen)
        proj.update(pre, pre, running_average=True)
    w = proj.weights()
    eye = torch.eye(4, dtype=torch.bool)
    assert (w[eye] > 1.0).all()
    assert (w[~eye] < -1.0).all()
    # The learned mapping reproduces its input.
    assert torch.equal(proj.forward(pre).argmax(-1), pre.argmax(-1))


def test_forward_sums_to_one_per_hypercolumn():
    gen = torch.Generator().manual_seed(0)
    proj = Projection(8, 3, 5, pre_hc_size=4, alpha=0.01)
    proj.seed_from_data(one_hot_hcs(50, 2, 4, gen), 0.5, gen)
    out = proj.forward(one_hot_hcs(16, 2, 4, gen))
    assert torch.allclose(out.view(16, 3, 5).sum(-1), torch.ones(16, 3), atol=1e-5)


def test_traces_stay_valid_probabilities():
    gen = torch.Generator().manual_seed(0)
    proj = Projection(8, 2, 4, pre_hc_size=4, alpha=0.2)
    for _ in range(500):
        pre = one_hot_hcs(32, 2, 4, gen)
        proj.update(pre, proj.forward(pre))
    for p in (proj.p_i, proj.p_j, proj.p_ij):
        assert torch.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()
    assert torch.isfinite(proj.weights()).all()
    assert torch.allclose(proj.p_j.view(2, 4).sum(-1), torch.ones(2), atol=1e-4)


def test_masks_restrict_weights():
    for kw in ({"patch_size": 10}, {"mask_density": 0.25}):
        net = RateBCPNN(Config(hidden_hcs=3, hidden_mcs=4, **kw))
        mask = net.in_hid.mask
        assert mask.shape == (1568, 12)
        per_hc = mask[0::2, 0::4].sum(0)
        expected = 100 if "patch_size" in kw else 196
        assert (per_hc == expected).all()
        assert (net.in_hid.weights()[~mask] == 0).all()


def toy_classes(gen, n=2000):
    protos = (torch.rand(10, 784, generator=gen) > 0.5).float()
    y = torch.randint(10, (n,), generator=gen)
    flip = torch.rand(n, 784, generator=gen) < 0.05
    return torch.where(flip, 1 - protos[y], protos[y]), y


def test_network_learns_separable_toy_classes():
    x, y = toy_classes(torch.Generator().manual_seed(0))
    net = RateBCPNN(Config(hidden_hcs=2, hidden_mcs=20, epochs_unsupervised=2, batch_size=50))
    net.train_unsupervised(x)
    net.train_supervised(x, y)
    assert net.accuracy(x, y) > 0.95


def test_spiking_network_learns_separable_toy_classes():
    x, y = toy_classes(torch.Generator().manual_seed(0), n=500)
    net = SpikingBCPNN(Config(hidden_hcs=2, hidden_mcs=20, epochs_unsupervised=2, batch_size=50))
    net.train_unsupervised(x)
    net.train_supervised(x, y)
    assert net.accuracy(x, y) > 0.95
