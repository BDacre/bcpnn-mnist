# cortical-neurons-mnist

A small feed-forward BCPNN (Bayesian Confidence Propagation Neural Network) that classifies MNIST digits using only local Hebbian-Bayesian learning. There is no backpropagation and no gradient anywhere.

## Model

| Layer | Hypercolumns x minicolumns | Notes |
|---|---|---|
| Input | 784 x 2 | each pixel `x` becomes `[x, 1 - x]` |
| Hidden | 30 x 100 | softmax within each hypercolumn, learned without labels |
| Output | 1 x 10 | clamped to the label during training |

Each projection keeps probability traces `p_i`, `p_j`, `p_ij` (exponential moving averages of activity and co-activity) and derives

```
w_ij = log(p_ij / (p_i * p_j))        b_j = log(p_j)
```

Training has two phases: input -> hidden learns unsupervised, then it is frozen and hidden -> output learns with the output clamped to the label.

Three models share this code:

- `baseline` - input -> output only (a naive Bayes classifier).
- `rate` - units carry probabilities; traces are updated once per mini-batch.
- `spiking` - units emit stochastic spikes over 50 time steps per image; spikes are low-pass filtered into Z traces, and the P traces are learned from those.

## Running

```
uv sync
uv run pytest
uv run main.py --model baseline
uv run main.py --model rate --plot
uv run main.py --model rate --patch-size 10
uv run main.py --model spiking --subset 5000
```

`--plot` saves a confusion matrix and hidden receptive fields to `figures/`. Other options: `--seed`, `--epochs`, `--hidden-hcs`, `--hidden-mcs`, `--mask-density`, `--batch-size`. All hyperparameters live in `bcpnn/config.py`.

## Results

Test accuracy, 30 x 100 hidden layer, 3 unsupervised epochs, 12-core CPU.

| Model | Training images | Test accuracy | Train time |
|---|---|---|---|
| baseline | 60,000 | 84.5% | under 1 s |
| rate, dense | 60,000 | 92.6% (92.5 - 92.8 over 3 seeds) | about 45 s |
| rate, 10x10 patches | 60,000 | 96.2% (96.1 - 96.3 over 3 seeds) | about 65 s |
| spiking, dense | 60,000 | 92.6% (1 seed) | about 29 min |
| rate, dense | 5,000 | 91.0% | 5 s |
| spiking, dense | 5,000 | 90.9% | 145 s |

## Findings

- **Symmetry breaking needs data.** Starting from uniform traces plus noise, one minicolumn per hypercolumn wins every image and accuracy stays near 20%. Seeding each minicolumn from a random training image (`init_from_data`) fixes this.
- **Dense hypercolumns are redundant.** When every hypercolumn sees the whole image they all learn whole-digit prototypes, so adding hypercolumns adds little: tripling the hypercolumn count from 10 to 30 gained well under a point. Giving each hypercolumn a fixed local 10x10 patch (`--patch-size 10`) makes them encode different parts of the digit and lifts accuracy to about 96%. Random scattered pixel masks (`--mask-density`) help much less (about 93%).
- **The spiking model matches the rate model** on the same data and network size, (92.6% vs 92.6% on full MNIST, 90.9% vs 91.0% on 5,000 images), at roughly 30 to 40 times the compute.
