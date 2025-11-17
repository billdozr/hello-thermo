import time
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from jax.nn import sigmoid
from jax import lax

from thrml.block_management import Block
from thrml.block_sampling import SamplingSchedule, sample_states
from thrml.models.ising import (
    IsingEBM,
    IsingTrainingSpec,
    IsingSamplingProgram,
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


side = 8  # 8x8 bars & stripes

# All 8x8 bars & stripes patterns:
data_np_all = make_bars_stripes(side)

# Optional: subsample for speed (508 total BAS patterns for 8x8).
rng = np.random.RandomState(0)
n_train = 256  # or 512, or len(data_np_all)
idx = rng.choice(len(data_np_all), size=n_train, replace=False)
data_np = data_np_all[idx]

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
plt.savefig("01_dataset_samples.png", dpi=150, bbox_inches="tight")
print("Saved: 01_dataset_samples.png")
plt.close()

data = jnp.array(data_np, dtype=jnp.bool_)

key = jax.random.key(0)

n_hidden = 128

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
    n_warmup=10,   # Reduced for speed (was 20)
    n_samples=10,  # Reduced for speed (was 20)
    steps_per_sample=1,  # Reduced for speed (was 2)
)

schedule_negative = SamplingSchedule(
    n_warmup=20,   # Reduced for speed (was 50)
    n_samples=10,  # Reduced for speed (was 20)
    steps_per_sample=2,  # Reduced for speed (was 5)
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


def gibbs_python_baseline(
    biases,
    weights,
    beta,
    n_visible,
    n_hidden,
    warmup: int,
    n_samples: int,
    steps_per_sample: int,
    n_chains: int = 1,
    seed: int = 0,
):
    """
    Naive block-Gibbs sampler for the RBM using pure NumPy and Python loops.
    - biases: array of shape [n_visible + n_hidden]
    - weights: array of shape [n_edges], corresponding to a full bipartite W
    - beta: scalar inverse temperature
    - Schedule mirrors THRML: warmup, then n_samples with steps_per_sample between them.
    Returns: samples of shape [n_samples, n_chains, n_visible] as booleans.
    """
    rng = np.random.RandomState(seed)

    # Convert params to NumPy & unpack into visible/hidden parts
    b = np.asarray(biases, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64).reshape(n_visible, n_hidden)

    b_v = b[:n_visible]
    b_h = b[n_visible:]

    # Initial random spins
    v = rng.rand(n_chains, n_visible) < 0.5   # bool
    h = rng.rand(n_chains, n_hidden) < 0.5

    def bools_to_spins_np(x_bool):
        return np.where(x_bool, 1.0, -1.0)

    recorded = []
    total_steps = warmup + n_samples * steps_per_sample

    for t in range(total_steps):
        # ---- Update hidden given visible ----
        s_v = bools_to_spins_np(v)                         # [C, V]
        field_h = b_h + s_v @ W                            # [C, H]
        probs_h = 1.0 / (1.0 + np.exp(-2.0 * beta * field_h))
        h = rng.rand(*h.shape) < probs_h                   # bool

        # ---- Update visible given hidden ----
        s_h = bools_to_spins_np(h)                         # [C, H]
        field_v = b_v + s_h @ W.T                          # [C, V]
        probs_v = 1.0 / (1.0 + np.exp(-2.0 * beta * field_v))
        v = rng.rand(*v.shape) < probs_v                   # bool

        # Record samples in the same pattern as THRML:
        if t >= warmup and (t - warmup) % steps_per_sample == 0:
            recorded.append(v.copy())

    samples = np.stack(recorded, axis=0)  # [n_samples, n_chains, n_visible]
    return samples

learning_rate = 0.05
n_epochs = 50  # Reduced for faster training (was 200)

# Number of chains per phase (reduced for speed)
n_chains_pos = 4      # positive phase: 4 chains sharing the data (was 16)
n_chains_neg = 32     # negative phase: 32 parallel fantasy chains (was 256)


def single_epoch(carry, epoch_idx):
    """
    Single training epoch function for use with jax.lax.scan.
    Uses vmap-like vectorization through scan for maximum JIT efficiency.
    
    Args:
        carry: Dictionary containing:
            - biases: current bias parameters
            - weights: current weight parameters
            - key: JAX random key
        epoch_idx: Current epoch index (unused but required by scan)
    
    Returns:
        (updated_carry, metrics): Updated carry and metrics dictionary
    """
    biases = carry['biases']
    weights = carry['weights']
    key = carry['key']
    
    # Rebuild model from current parameters (avoids PyTree issues)
    model = IsingEBM(nodes, edges, biases, weights, beta)
    
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
    # Gradients of the Kullback-Leibler (KL) divergence
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

    # Track simple reconstruction error (one-step CD) and weight norm
    key, mse, bce, v_recon_samples = reconstruction_error(
        key, data, biases, weights, beta, n_visible, n_hidden
    )
    weight_norm = jnp.linalg.norm(weights)
    
    # Return updated carry and metrics
    updated_carry = {
        'biases': biases,
        'weights': weights,
        'key': key,
    }
    
    metrics = {
        'mse': mse,
        'bce': bce,
        'weight_norm': weight_norm,
        'v_recon_samples': v_recon_samples,
    }
    
    return updated_carry, metrics


# Initialize carry for scan (no model, rebuild from params each epoch)
initial_carry = {
    'biases': biases,
    'weights': weights,
    'key': key,
}

# Create epoch indices (just for iteration, values don't matter)
epoch_indices = jnp.arange(n_epochs)

# JIT compile the scan operation for maximum speed
print("Compiling training loop with scan...")
scan_fn = jax.jit(lambda carry, xs: lax.scan(single_epoch, carry, xs))
final_carry, all_metrics = scan_fn(initial_carry, epoch_indices)

# Extract final values
biases = final_carry['biases']
weights = final_carry['weights']
key = final_carry['key']
# Rebuild final model from final parameters
model = IsingEBM(nodes, edges, biases, weights, beta)
v_recon_samples = all_metrics['v_recon_samples'][-1]  # Last epoch's reconstruction

# Extract history for plotting
recon_mse_history = [float(x) for x in all_metrics['mse']]
recon_bce_history = [float(x) for x in all_metrics['bce']]
weight_norm_history = [float(x) for x in all_metrics['weight_norm']]

# Print periodic updates (every 10 epochs)
for epoch in range(0, n_epochs, 10):
    print(
        f"Epoch {epoch:3d} | recon MSE={recon_mse_history[epoch]:.4f}  "
        f"BCE={recon_bce_history[epoch]:.4f}  ||W||={weight_norm_history[epoch]:.3f}"
    )
if (n_epochs - 1) % 10 != 0:
    print(
        f"Epoch {n_epochs-1:3d} | recon MSE={recon_mse_history[-1]:.4f}  "
        f"BCE={recon_bce_history[-1]:.4f}  ||W||={weight_norm_history[-1]:.3f}"
    )

fig, ax = plt.subplots(1, 3, figsize=(14, 3))
ax[0].plot(recon_mse_history)
ax[0].set_title("Reconstruction MSE")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("MSE")
ax[0].grid(True, alpha=0.3)

ax[1].plot(recon_bce_history)
ax[1].set_title("Reconstruction BCE")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("BCE")
ax[1].grid(True, alpha=0.3)

ax[2].plot(weight_norm_history)
ax[2].set_title("Weight Norm ||W||")
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("L2 Norm")
ax[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("02_training_curves.png", dpi=150, bbox_inches="tight")
print("Saved: 02_training_curves.png")
plt.close()


def plot_hidden_filters(weights, side, n_visible, n_hidden, n_cols=8):
    W = np.array(weights.reshape(n_visible, n_hidden))  # convert JAX -> NumPy

    n_rows = int(np.ceil(n_hidden / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows))
    axes = np.atleast_2d(axes)

    # Find global min/max for consistent colormap scaling
    vmax = np.abs(W).max()
    vmin = -vmax

    for j in range(n_hidden):
        r, c = divmod(j, n_cols)
        ax = axes[r, c]
        ax.imshow(
            W[:, j].reshape(side, side),
            cmap="bwr",
            interpolation="nearest",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"h{j}", fontsize=8)

    # hide unused axes
    for j in range(n_hidden, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("Hidden-unit filters (visible → hidden couplings)")
    plt.tight_layout()
    return fig


fig_filters = plot_hidden_filters(weights, side, n_visible, n_hidden)
fig_filters.savefig("03_hidden_filters.png", dpi=150, bbox_inches="tight")
print("Saved: 03_hidden_filters.png")
plt.close()

# Use the last v_recon_samples from the training loop:
recon_np = np.array(v_recon_samples.astype(jnp.float32))

num_show = min(8, n_samples)

fig, axes = plt.subplots(
    2, num_show,
    figsize=(2.0 * num_show, 4.0),  # wider + taller than before
    constrained_layout=False,
)

# Make sure axes is 2D array even if num_show == 1
axes = np.atleast_2d(axes)

for i in range(num_show):
    # ---------- Original ----------
    axes[0, i].imshow(
        data_np[i].reshape(side, side),
        cmap="gray_r",
        interpolation="nearest",
    )
    axes[0, i].axis("off")
    axes[0, i].set_title(f"#{i}", fontsize=10)

    # ---------- Reconstruction ----------
    axes[1, i].imshow(
        recon_np[i].reshape(side, side),
        cmap="gray_r",
        interpolation="nearest",
    )
    axes[1, i].axis("off")

# Big y‑axis labels for the whole figure (not per‑subplot)
fig.text(
    0.01, 0.75, "Original",
    va="center", ha="left",
    rotation="vertical", fontsize=12,
)
fig.text(
    0.01, 0.25, "Reconstruction",
    va="center", ha="left",
    rotation="vertical", fontsize=12,
)

fig.suptitle("Data vs One-Step Reconstructions", fontsize=14)

# Adjust layout so labels aren't clipped
plt.subplots_adjust(
    left=0.10,       # leave space for the big y labels
    wspace=0.05,
    hspace=0.05,
)

plt.savefig("04_reconstructions.png", dpi=150, bbox_inches="tight")
print("Saved: 04_reconstructions.png")
plt.close()

# ------------------------------------------------------------
# 7. Free-running sampling with THRML (no data clamping)
# ------------------------------------------------------------

free_blocks = negative_blocks
free_program = IsingSamplingProgram(model, free_blocks, [])

n_model_samples = 16

# We’ll get 16 samples by asking the sampler to produce 16 steps,
# not by adding a batch dimension to the state.
free_schedule = SamplingSchedule(
    n_warmup=300,  # burn-in
    n_samples=n_model_samples,  # number of *recorded* samples
    steps_per_sample=20,
)

# IMPORTANT: scalar chain state => batch_shape=()
key, subkey = jax.random.split(key)
init_state_free = hinton_init(
    subkey,
    model,
    free_blocks,
    batch_shape=(),
)

# Run THRML Gibbs sampler, requesting only visible nodes
# --- Warm-up call: triggers JIT compile, not timed ---
key, subkey = jax.random.split(key)
_ = sample_states(
    subkey,
    free_program,
    free_schedule,
    init_state_free,
    state_clamp=[],
    nodes_to_sample=[Block(visible_nodes)],
)

# --- Timed call: reuses compiled kernel ---
key, subkey = jax.random.split(key)
t0_thrml = time.perf_counter()
free_samples_struct = sample_states(
    subkey,
    free_program,
    free_schedule,
    init_state_free,
    state_clamp=[],
    nodes_to_sample=[Block(visible_nodes)],
)
jax.block_until_ready(_[0])
thrml_elapsed = time.perf_counter() - t0_thrml
print(f"THRML free-running sampling (no compile) elapsed: {thrml_elapsed:.4f} s")

# sample_states returns the final state for each block.
# free_blocks = [Block(visible_nodes), Block(hidden_nodes)]
# So free_samples_struct[0] is the visible layer, [1] is the hidden layer
vis_chain = free_samples_struct[0]

# Convert to NumPy and collapse all non-feature axes
vis_chain_np = np.array(vis_chain)

# We expect the last axis to be n_visible; flatten the rest
if vis_chain_np.ndim >= 2:
    if vis_chain_np.shape[-1] != n_visible:
        # Best-effort reshape if THRML changed internal ordering
        vis_chain_np = vis_chain_np.reshape(-1, n_visible)
    else:
        vis_chain_np = vis_chain_np.reshape(-1, vis_chain_np.shape[-1])
else:
    raise RuntimeError(f"Unexpected THRML sample shape: {vis_chain_np.shape}")

# Take the first n_model_samples visible configurations
model_vis_samples = vis_chain_np[:n_model_samples]

# ------------------------------------------------------------
# Plot the free-running model samples (no conditioning)
# ------------------------------------------------------------
n_show = model_vis_samples.shape[0]
fig, axes = plt.subplots(1, n_show, figsize=(1.6 * n_show, 1.6))
if n_show == 1:
    axes = [axes]

for i in range(n_show):
    ax = axes[i]
    ax.imshow(
        model_vis_samples[i].reshape(side, side),
        cmap="gray_r",
        interpolation="nearest",
    )
    ax.axis("off")
    ax.set_title(f"{i}", fontsize=8)

fig.suptitle("Free-running THRML RBM samples", fontsize=12)
plt.tight_layout()
plt.savefig("05_free_running_samples.png", dpi=150, bbox_inches="tight")
print("Saved: 05_free_running_samples.png")
plt.close()

# ============================================================
# 8. Naive Python Gibbs baseline (no THRML)
# ============================================================

# Use the same schedule as THRML for fairness
warmup = free_schedule.n_warmup
steps_per_sample = free_schedule.steps_per_sample
n_samples_python = n_model_samples

biases_np = np.array(biases)
weights_np = np.array(weights)
beta_scalar = float(beta)

t0 = time.perf_counter()
python_samples = gibbs_python_baseline(
    biases_np,
    weights_np,
    beta_scalar,
    n_visible,
    n_hidden,
    warmup=warmup,
    n_samples=n_samples_python,
    steps_per_sample=steps_per_sample,
    n_chains=1,      # one chain, 16 samples over time (like THRML)
    seed=0,
)
python_elapsed = time.perf_counter() - t0
print(f"Naive Python Gibbs sampling elapsed: {python_elapsed:.4f} s")
if thrml_elapsed > 0:
    print(f"Speed ratio (Python / THRML): {python_elapsed / thrml_elapsed:.2f}x slower")

# python_samples: [n_samples, 1, n_visible] -> [n_samples, n_visible]
python_vis_samples = python_samples[:, 0, :]

# Plot them in the same 1×16 grid layout
n_show = python_vis_samples.shape[0]
fig, axes = plt.subplots(1, n_show, figsize=(1.6 * n_show, 1.6))
if n_show == 1:
    axes = [axes]

for i in range(n_show):
    ax = axes[i]
    ax.imshow(
        python_vis_samples[i].reshape(side, side),
        cmap="gray_r",
        interpolation="nearest",
    )
    ax.axis("off")
    ax.set_title(f"{i}", fontsize=8)

fig.suptitle("Free-running naive Python RBM samples", fontsize=12)
plt.tight_layout()
plt.savefig("06_free_running_python_samples.png", dpi=150, bbox_inches="tight")
print("Saved: 06_free_running_python_samples.png")
plt.close()

print("\n" + "=" * 60)
print("Training complete! Generated visualizations:")
print("  01_dataset_samples.png             - Original bars & stripes dataset")
print("  02_training_curves.png             - MSE, BCE, and weight norm over time")
print("  03_hidden_filters.png              - Learned hidden unit filters")
print("  04_reconstructions.png             - Original vs reconstructed samples")
print("  05_free_running_samples.png        - Free-running THRML RBM samples")
print("  06_free_running_python_samples.png - Free-running naive Python RBM samples")

print("=" * 60)
