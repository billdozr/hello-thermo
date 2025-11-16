import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.nn import sigmoid

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule
from thrml.models.ising import (
    IsingEBM,
    IsingTrainingSpec,
    IsingSamplingProgram,  # not strictly needed yet, but handy
    estimate_kl_grad,
    hinton_init,
)
from thrml.pgm import SpinNode


def make_bars_stripes(side: int, include_all: bool = False) -> np.ndarray:
    """
    Return [n_patterns, side*side] boolean array for a bars & stripes dataset.
    True = 'on' pixel, False = 'off' pixel.
    """
    patterns = []
    masks = range(2**side) if include_all else range(1, 2**side - 1)

    # Horizontal stripes: choose a subset of rows to be ON
    for mask in masks:
        rows_on = [i for i in range(side) if mask & (1 << i)]
        img = np.zeros((side, side), dtype=bool)
        img[rows_on, :] = True
        patterns.append(img)

    # Vertical stripes: choose a subset of columns to be ON
    for mask in masks:
        cols_on = [i for i in range(side) if mask & (1 << i)]
        img = np.zeros((side, side), dtype=bool)
        img[:, cols_on] = True
        patterns.append(img)

    patterns = np.stack(patterns, axis=0)
    return patterns.reshape(len(patterns), side * side)


# Tiny dataset: 4x4 bars & stripes
side = 4
data_np = make_bars_stripes(side)
n_samples, n_visible = data_np.shape

print("Data shape:", data_np.shape)

# Visual sanity check
fig, axes = plt.subplots(2, 8, figsize=(10, 3))
for i, ax in enumerate(axes.flatten()):
    if i >= len(data_np):
        ax.axis("off")
        continue
    ax.imshow(data_np[i].reshape(side, side), cmap="gray_r", interpolation="nearest")
    ax.axis("off")
fig.suptitle("Bars & Stripes samples")
plt.tight_layout()

data = jnp.array(data_np, dtype=jnp.bool_)

key = jax.random.key(0)

n_hidden = 8  # small model for first pass

# Create spin nodes
visible_nodes = [SpinNode() for _ in range(n_visible)]
hidden_nodes = [SpinNode() for _ in range(n_hidden)]
nodes = visible_nodes + hidden_nodes

# Fully connected bipartite edges (visible_i, hidden_j)
edges = []
for v in visible_nodes:
    for h in hidden_nodes:
        edges.append((v, h))
n_edges = len(edges)

# Small random initialization for biases and weights
beta = jnp.array(
    1.0
)  # inverse temperature in Ising energy E(s) = -β(Σ b_i s_i + Σ J_ij s_i s_j)

key, subkey = jax.random.split(key)
biases = 0.01 * jax.random.normal(subkey, (len(nodes),))

key, subkey = jax.random.split(key)
weights = 0.01 * jax.random.normal(subkey, (n_edges,))

model = IsingEBM(nodes, edges, biases, weights, beta)
print("Nodes:", len(nodes), "Edges:", len(edges))

# Data block corresponds to visible units
data_block = Block(visible_nodes)

# Positive phase: data clamped, only hiddens are free
positive_blocks = [Block(hidden_nodes)]

# Negative phase: both visibles and hiddens are free (full Gibbs chain)
negative_blocks = [Block(visible_nodes), Block(hidden_nodes)]

# Simple sampling schedules for both phases
schedule_positive = SamplingSchedule(
    n_warmup=20,
    n_samples=20,
    steps_per_sample=2,
)

schedule_negative = SamplingSchedule(
    n_warmup=20,
    n_samples=20,
    steps_per_sample=2,
)

training_spec = IsingTrainingSpec(
    ebm=model,
    data_blocks=[data_block],
    conditioning_blocks=[],
    positive_sampling_blocks=positive_blocks,
    negative_sampling_blocks=negative_blocks,
    schedule_positive=schedule_positive,
    schedule_negative=schedule_negative,
)


def bools_to_spins(x_bool: jnp.ndarray) -> jnp.ndarray:
    """Map booleans to spins in {-1, +1}."""
    return jnp.where(x_bool, 1.0, -1.0)


def bernoulli_from_field(key, field, beta):
    """
    Given local fields h_i, sample s_i ~ Bernoulli(σ(2 β h_i)).
    Returns (new_key, samples_bool, probs).
    """
    logits = 2.0 * beta * field
    probs = sigmoid(logits)
    key, subkey = jax.random.split(key)
    samples = jax.random.bernoulli(subkey, probs)
    return key, samples.astype(jnp.bool_), probs


def sample_hidden_given_visible(
    key, v_bool, biases, weights, beta, n_visible, n_hidden
):
    """
    P(h | v) using Ising conditionals and the same (b, J) that THRML uses inside IsingEBM.
    """
    W = weights.reshape(n_visible, n_hidden)  # J_ij for vis i, hid j
    s_v = bools_to_spins(v_bool)  # shape [batch, n_visible]
    b_h = biases[n_visible:]  # last n_hidden entries

    field_h = b_h + s_v @ W  # [batch, n_hidden]
    key, h_samples, probs_h = bernoulli_from_field(key, field_h, beta)
    return key, h_samples, probs_h


def sample_visible_given_hidden(
    key, h_bool, biases, weights, beta, n_visible, n_hidden
):
    """
    P(v | h) using Ising conditionals.
    """
    W = weights.reshape(n_visible, n_hidden)
    s_h = bools_to_spins(h_bool)  # [batch, n_hidden]
    b_v = biases[:n_visible]

    field_v = b_v + s_h @ W.T  # [batch, n_visible]
    key, v_samples, probs_v = bernoulli_from_field(key, field_v, beta)
    return key, v_samples, probs_v


def reconstruction_error(key, data_bool, biases, weights, beta, n_visible, n_hidden):
    """
    One-step CD-like reconstruction:
      v0 -> h ~ p(h|v0) -> v1 ~ p(v|h)
    Returns (new_key, mse, bce, v_recon_samples).
    """
    x = data_bool.astype(jnp.float32)  # [N, D] in {0,1}

    key, h_samples, _ = sample_hidden_given_visible(
        key, data_bool, biases, weights, beta, n_visible, n_hidden
    )
    key, v_samples, v_probs = sample_visible_given_hidden(
        key, h_samples, biases, weights, beta, n_visible, n_hidden
    )

    mse = jnp.mean((x - v_probs) ** 2)
    bce = -jnp.mean(
        x * jnp.log(v_probs + 1e-6) + (1.0 - x) * jnp.log(1.0 - v_probs + 1e-6)
    )
    return key, mse, bce, v_samples


learning_rate = 0.05
n_epochs = 200

n_chains_pos = 1  # positive phase: 1 chain per mini-ensemble
n_chains_neg = n_samples  # negative phase: parallel chains

recon_mse_history = []
recon_bce_history = []
weight_norm_history = []

for epoch in range(n_epochs):
    # (Re)build IsingTrainingSpec with current model
    training_spec = IsingTrainingSpec(
        ebm=model,
        data_blocks=[data_block],
        conditioning_blocks=[],
        positive_sampling_blocks=positive_blocks,
        negative_sampling_blocks=negative_blocks,
        schedule_positive=schedule_positive,
        schedule_negative=schedule_negative,
    )

    # Init positive (clamped) chains: shape [n_chains_pos, batch, hidden_block_size]
    key, k_pos = jax.random.split(key)
    init_state_positive = hinton_init(
        k_pos,
        model,
        positive_blocks,
        batch_shape=(n_chains_pos, n_samples),
    )

    # Init negative (free) chains: one chain per sample, same batch shape used in docs
    key, k_neg = jax.random.split(key)
    init_state_negative = hinton_init(
        k_neg,
        model,
        negative_blocks,
        batch_shape=(n_chains_neg,),
    )

    # Estimate KL gradients from data vs model
    key, k_grad = jax.random.split(key)
    weight_grads, bias_grads, pos_moments, neg_moments = estimate_kl_grad(
        k_grad,
        training_spec,
        bias_nodes=nodes,
        weight_edges=edges,
        data=[data],  # shape [batch, n_visible]
        conditioning_values=[],  # no conditioning in this RBM
        init_state_positive=init_state_positive,
        init_state_negative=init_state_negative,
    )

    # Gradient descent step on KL(Q || P)
    biases = biases - learning_rate * bias_grads
    weights = weights - learning_rate * weight_grads

    # Update model object with new parameters
    model = IsingEBM(nodes, edges, biases, weights, beta)

    # Track simple reconstruction error (one-step CD) and weight norm
    key, mse, bce, v_recon_samples = reconstruction_error(
        key, data, biases, weights, beta, n_visible, n_hidden
    )
    recon_mse_history.append(float(mse))
    recon_bce_history.append(float(bce))
    weight_norm_history.append(float(jnp.linalg.norm(weights)))

    if epoch % 10 == 0 or epoch == n_epochs - 1:
        print(
            f"Epoch {epoch:3d} | recon MSE={mse:.4f}  BCE={bce:.4f}  ||W||={weight_norm_history[-1]:.3f}"
        )

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].plot(recon_mse_history)
ax[0].set_title("Reconstruction MSE")
ax[0].set_xlabel("Epoch")

ax[1].plot(recon_bce_history)
ax[1].set_title("Reconstruction BCE")
ax[1].set_xlabel("Epoch")

plt.tight_layout()


def plot_hidden_filters(weights, side, n_visible, n_hidden, n_cols=8):
    W = np.array(weights.reshape(n_visible, n_hidden))  # convert JAX -> NumPy

    n_rows = int(np.ceil(n_hidden / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows))
    axes = np.atleast_2d(axes)

    for j in range(n_hidden):
        r, c = divmod(j, n_cols)
        ax = axes[r, c]
        ax.imshow(W[:, j].reshape(side, side), cmap="bwr", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"h{j}", fontsize=8)

    # hide unused axes
    for j in range(n_hidden, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("Hidden-unit filters (visible → hidden couplings)")
    plt.tight_layout()


plot_hidden_filters(weights, side, n_visible, n_hidden)

# Use the last v_recon_samples from the training loop:
recon_np = np.array(v_recon_samples.astype(jnp.float32))

num_show = min(8, n_samples)
fig, axes = plt.subplots(2, num_show, figsize=(1.6 * num_show, 3))

for i in range(num_show):
    axes[0, i].imshow(
        data_np[i].reshape(side, side), cmap="gray_r", interpolation="nearest"
    )
    axes[0, i].axis("off")
    axes[0, i].set_title("data", fontsize=8)

    axes[1, i].imshow(
        recon_np[i].reshape(side, side), cmap="gray_r", interpolation="nearest"
    )
    axes[1, i].axis("off")
    axes[1, i].set_title("recon", fontsize=8)

plt.tight_layout()
