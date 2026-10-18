from dataclasses import dataclass


@dataclass
class Config:
    # Hidden layer: hypercolumns x minicolumns. hidden_hcs=0 gives the
    # input->output baseline with no hidden layer.
    hidden_hcs: int = 30
    hidden_mcs: int = 100
    n_classes: int = 10

    # Receptive field of each hidden hypercolumn (fixed, not learned):
    #   patch_size > 0   a random patch_size x patch_size window of the image
    #   mask_density < 1 a random subset of that fraction of the pixels
    #   otherwise        every pixel (dense)
    patch_size: int = 0
    mask_density: float = 1.0

    # Trace learning rates (per mini-batch exponential moving average).
    alpha_hidden: float = 0.01
    alpha_output: float = 0.002

    # Softmax gain and bias gain per projection.
    gain_hidden: float = 1.0
    bias_gain_hidden: float = 1.0
    gain_output: float = 1.0
    bias_gain_output: float = 1.0

    # Symmetry breaking: each hidden minicolumn starts as a prototype blended
    # this far from the mean image towards one random training image. Starting
    # from uniform traces plus noise collapses onto one minicolumn per hypercolumn.
    init_from_data: float = 0.3

    # Floor applied to traces before taking logs.
    eps: float = 1e-6

    # Spiking model only. Each image is presented for spk_steps time steps; a
    # fully active unit spikes with probability spk_pmax per step; spikes are
    # low-pass filtered into Z traces with time constant spk_tau_z (in steps).
    # P traces are updated from the Z traces every spk_learn_every steps, once
    # spk_settle steps have passed since image onset.
    spk_steps: int = 50
    spk_pmax: float = 0.5
    spk_tau_z: float = 10.0
    spk_settle: int = 10
    spk_learn_every: int = 5

    epochs_unsupervised: int = 3
    epochs_supervised: int = 1
    batch_size: int = 128
    seed: int = 0
