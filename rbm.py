"""
Restricted Boltzmann Machine (RBM) implementation using THRML/JAX.

Demonstrates training an RBM on bars & stripes data using block Gibbs sampling
via the THRML library's Ising model interface.
"""

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


# ============================================================
# Data Generation
# ============================================================


def make_bars_stripes(side: int, include_all: bool = False) -> np.ndarray:
    """
    Generate bars & stripes dataset.

    Returns:
        Array of shape [n_patterns, side*side] with boolean values.
        True = 'on' pixel, False = 'off' pixel.
    """
    patterns = []
    masks = range(2**side) if include_all else range(1, 2**side - 1)

    for mask in masks:
        rows_on = [i for i in range(side) if mask & (1 << i)]
        img = np.zeros((side, side), dtype=bool)
        img[rows_on, :] = True
        patterns.append(img)

    for mask in masks:
        cols_on = [i for i in range(side) if mask & (1 << i)]
        img = np.zeros((side, side), dtype=bool)
        img[:, cols_on] = True
        patterns.append(img)

    patterns = np.stack(patterns, axis=0)
    return patterns.reshape(len(patterns), side * side)


# ============================================================
# Utility Functions
# ============================================================


def bools_to_spins(x_bool: jnp.ndarray) -> jnp.ndarray:
    """Map booleans to spins in {-1, +1}."""
    return jnp.where(x_bool, 1.0, -1.0)


def bernoulli_from_field(key, field, beta):
    """
    Sample from Bernoulli distribution given local fields.

    Given local fields h_i, sample s_i ~ Bernoulli(σ(2 β h_i)).

    Returns:
        Tuple of (new_key, samples_bool, probs).
    """
    logits = 2.0 * beta * field
    probs = sigmoid(logits)
    key, subkey = jax.random.split(key)
    samples = jax.random.bernoulli(subkey, probs)
    return key, samples.astype(jnp.bool_), probs


def sample_hidden_given_visible(
    key, v_bool, biases, weights, beta, n_visible, n_hidden
):
    """Compute P(h | v) using Ising conditionals."""
    W = weights.reshape(n_visible, n_hidden)
    s_v = bools_to_spins(v_bool)
    b_h = biases[n_visible:]

    field_h = b_h + s_v @ W
    key, h_samples, probs_h = bernoulli_from_field(key, field_h, beta)
    return key, h_samples, probs_h


def sample_visible_given_hidden(
    key, h_bool, biases, weights, beta, n_visible, n_hidden
):
    """Compute P(v | h) using Ising conditionals."""
    W = weights.reshape(n_visible, n_hidden)
    s_h = bools_to_spins(h_bool)
    b_v = biases[:n_visible]

    field_v = b_v + s_h @ W.T
    key, v_samples, probs_v = bernoulli_from_field(key, field_v, beta)
    return key, v_samples, probs_v


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
    Naive block-Gibbs sampler using pure NumPy.

    Returns:
        Samples of shape [n_samples, n_chains, n_visible] as booleans.
    """
    rng = np.random.RandomState(seed)

    b = np.asarray(biases, dtype=np.float64)
    W = np.asarray(weights, dtype=np.float64).reshape(n_visible, n_hidden)
    b_v = b[:n_visible]
    b_h = b[n_visible:]

    v = rng.rand(n_chains, n_visible) < 0.5
    h = rng.rand(n_chains, n_hidden) < 0.5

    def bools_to_spins_np(x_bool):
        return np.where(x_bool, 1.0, -1.0)

    recorded = []
    total_steps = warmup + n_samples * steps_per_sample

    for t in range(total_steps):
        s_v = bools_to_spins_np(v)
        field_h = b_h + s_v @ W
        probs_h = 1.0 / (1.0 + np.exp(-2.0 * beta * field_h))
        h = rng.rand(*h.shape) < probs_h

        s_h = bools_to_spins_np(h)
        field_v = b_v + s_h @ W.T
        probs_v = 1.0 / (1.0 + np.exp(-2.0 * beta * field_v))
        v = rng.rand(*v.shape) < probs_v

        if t >= warmup and (t - warmup) % steps_per_sample == 0:
            recorded.append(v.copy())

    samples = np.stack(recorded, axis=0)
    return samples


# ============================================================
# RBM Class, the heart of it all!
# ============================================================


class RBM:
    """Restricted Boltzmann Machine using THRML Ising model."""

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        learning_rate: float = 0.05,
        n_chains_pos: int = 4,
        n_chains_neg: int = 32,
    ):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.n_chains_pos = n_chains_pos
        self.n_chains_neg = n_chains_neg

        self.visible_nodes = [SpinNode() for _ in range(n_visible)]
        self.hidden_nodes = [SpinNode() for _ in range(n_hidden)]
        self.nodes = self.visible_nodes + self.hidden_nodes

        self.edges = []
        for v in self.visible_nodes:
            for h in self.hidden_nodes:
                self.edges.append((v, h))

        self.beta = jnp.array(1.0)
        self.biases = None
        self.weights = None
        self.model = None

        self.data_block = Block(self.visible_nodes)
        self.positive_blocks = [Block(self.hidden_nodes)]
        self.negative_blocks = [Block(self.visible_nodes), Block(self.hidden_nodes)]

        self.schedule_positive = SamplingSchedule(
            n_warmup=10,
            n_samples=10,
            steps_per_sample=1,
        )

        self.schedule_negative = SamplingSchedule(
            n_warmup=20,
            n_samples=10,
            steps_per_sample=2,
        )

    def initialize_parameters(self, key):
        """Initialize biases and weights with small random values."""
        key, subkey = jax.random.split(key)
        self.biases = 0.01 * jax.random.normal(subkey, (len(self.nodes),))

        key, subkey = jax.random.split(key)
        self.weights = 0.01 * jax.random.normal(subkey, (len(self.edges),))

        self.model = IsingEBM(
            self.nodes, self.edges, self.biases, self.weights, self.beta
        )
        return key

    def _reconstruction_error_with_params(self, key, data_bool, biases, weights):
        """Internal method to compute reconstruction error with given parameters."""
        x = data_bool.astype(jnp.float32)

        key, h_samples, _ = sample_hidden_given_visible(
            key,
            data_bool,
            biases,
            weights,
            self.beta,
            self.n_visible,
            self.n_hidden,
        )
        key, v_samples, v_probs = sample_visible_given_hidden(
            key,
            h_samples,
            biases,
            weights,
            self.beta,
            self.n_visible,
            self.n_hidden,
        )

        mse = jnp.mean((x - v_probs) ** 2)
        bce = -jnp.mean(
            x * jnp.log(v_probs + 1e-6) + (1.0 - x) * jnp.log(1.0 - v_probs + 1e-6)
        )
        return key, mse, bce, v_samples

    def reconstruction_error(self, key, data_bool):
        """
        Compute one-step reconstruction error: v0 -> h -> v1.

        Uses the model's current biases and weights.

        Returns:
            Tuple of (new_key, mse, bce, v_recon_samples).
        """
        return self._reconstruction_error_with_params(
            key, data_bool, self.biases, self.weights
        )

    def train_epoch(self, carry, epoch_idx, data, n_samples):
        """
        Single training epoch using THRML estimate_kl_grad.

        Args:
            carry: Dictionary with biases, weights, key
            epoch_idx: Current epoch index (unused but required by scan)
            data: Training data
            n_samples: Number of training samples

        Returns:
            Tuple of (updated_carry, metrics).
        """
        biases = carry["biases"]
        weights = carry["weights"]
        key = carry["key"]

        model = IsingEBM(self.nodes, self.edges, biases, weights, self.beta)

        training_spec = IsingTrainingSpec(
            ebm=model,
            data_blocks=[self.data_block],
            conditioning_blocks=[],
            positive_sampling_blocks=self.positive_blocks,
            negative_sampling_blocks=self.negative_blocks,
            schedule_positive=self.schedule_positive,
            schedule_negative=self.schedule_negative,
        )

        key, k_pos = jax.random.split(key)
        init_state_positive = hinton_init(
            k_pos,
            model,
            self.positive_blocks,
            batch_shape=(self.n_chains_pos, n_samples),
        )

        key, k_neg = jax.random.split(key)
        init_state_negative = hinton_init(
            k_neg,
            model,
            self.negative_blocks,
            batch_shape=(self.n_chains_neg,),
        )

        key, k_grad = jax.random.split(key)
        weight_grads, bias_grads, pos_moments, neg_moments = estimate_kl_grad(
            k_grad,
            training_spec,
            bias_nodes=self.nodes,
            weight_edges=self.edges,
            data=[data],
            conditioning_values=[],
            init_state_positive=init_state_positive,
            init_state_negative=init_state_negative,
        )

        biases = biases - self.learning_rate * bias_grads
        weights = weights - self.learning_rate * weight_grads

        key, mse, bce, v_recon_samples = self._reconstruction_error_with_params(
            key, data, biases, weights
        )
        weight_norm = jnp.linalg.norm(weights)

        updated_carry = {
            "biases": biases,
            "weights": weights,
            "key": key,
        }

        metrics = {
            "mse": mse,
            "bce": bce,
            "weight_norm": weight_norm,
            "v_recon_samples": v_recon_samples,
        }

        return updated_carry, metrics

    def train(self, key, data, n_epochs):
        """
        Train the RBM using jax.lax.scan for efficiency.

        Returns:
            Tuple of (final_key, metrics_history).
        """
        n_samples = data.shape[0]

        initial_carry = {
            "biases": self.biases,
            "weights": self.weights,
            "key": key,
        }

        epoch_indices = jnp.arange(n_epochs)

        print("Compiling training loop with scan...")
        scan_fn = jax.jit(
            lambda carry, xs: lax.scan(
                lambda c, idx: self.train_epoch(c, idx, data, n_samples), carry, xs
            )
        )
        final_carry, all_metrics = scan_fn(initial_carry, epoch_indices)

        self.biases = final_carry["biases"]
        self.weights = final_carry["weights"]
        self.model = IsingEBM(
            self.nodes, self.edges, self.biases, self.weights, self.beta
        )

        return final_carry["key"], all_metrics

    def sample_free_running(self, key, n_model_samples=16):
        """
        Generate samples from the model without data clamping.

        Returns:
            Tuple of (new_key, visible_samples, elapsed_time).
        """
        free_blocks = self.negative_blocks
        free_program = IsingSamplingProgram(self.model, free_blocks, [])

        free_schedule = SamplingSchedule(
            n_warmup=300,
            n_samples=n_model_samples,
            steps_per_sample=20,
        )

        key, subkey = jax.random.split(key)
        init_state_free = hinton_init(
            subkey,
            self.model,
            free_blocks,
            batch_shape=(),
        )

        key, subkey = jax.random.split(key)
        _ = sample_states(
            subkey,
            free_program,
            free_schedule,
            init_state_free,
            state_clamp=[],
            nodes_to_sample=[Block(self.visible_nodes)],
        )

        key, subkey = jax.random.split(key)
        t0 = time.perf_counter()
        free_samples_struct = sample_states(
            subkey,
            free_program,
            free_schedule,
            init_state_free,
            state_clamp=[],
            nodes_to_sample=[Block(self.visible_nodes)],
        )
        jax.block_until_ready(free_samples_struct[0])
        elapsed = time.perf_counter() - t0

        vis_chain = free_samples_struct[0]
        vis_chain_np = np.array(vis_chain)

        if vis_chain_np.ndim >= 2:
            if vis_chain_np.shape[-1] != self.n_visible:
                vis_chain_np = vis_chain_np.reshape(-1, self.n_visible)
            else:
                vis_chain_np = vis_chain_np.reshape(-1, vis_chain_np.shape[-1])
        else:
            raise RuntimeError(f"Unexpected sample shape: {vis_chain_np.shape}")

        model_vis_samples = vis_chain_np[:n_model_samples]

        return key, model_vis_samples, elapsed


# ============================================================
# Visualization Functions
# ============================================================


def plot_dataset_samples(data_np, side, filename="./images/01_dataset_samples.png"):
    """Plot sample images from the dataset."""
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    for i, ax in enumerate(axes.flatten()):
        if i >= len(data_np):
            ax.axis("off")
            continue
        ax.imshow(
            data_np[i].reshape(side, side), cmap="gray_r", interpolation="nearest"
        )
        ax.axis("off")
    fig.suptitle("Bars & Stripes samples")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def plot_training_curves(metrics, filename="./images/02_training_curves.png"):
    """Plot training metrics over epochs."""
    recon_mse_history = [float(x) for x in metrics["mse"]]
    recon_bce_history = [float(x) for x in metrics["bce"]]
    weight_norm_history = [float(x) for x in metrics["weight_norm"]]

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
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()

    return recon_mse_history, recon_bce_history, weight_norm_history


def plot_hidden_filters(
    weights,
    side,
    n_visible,
    n_hidden,
    filename="./images/03_hidden_filters.png",
    n_cols=8,
):
    """Visualize learned hidden unit filters."""
    W = np.array(weights.reshape(n_visible, n_hidden))

    n_rows = int(np.ceil(n_hidden / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.6 * n_rows))
    axes = np.atleast_2d(axes)

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

    for j in range(n_hidden, n_rows * n_cols):
        r, c = divmod(j, n_cols)
        axes[r, c].axis("off")

    fig.suptitle("Hidden-unit filters (visible → hidden couplings)")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def plot_reconstructions(
    data_np, recon_np, side, filename="./images/04_reconstructions.png"
):
    """Plot original data vs reconstructions."""
    num_show = min(8, len(data_np))

    fig, axes = plt.subplots(
        2,
        num_show,
        figsize=(2.0 * num_show, 4.0),
        constrained_layout=False,
    )
    axes = np.atleast_2d(axes)

    for i in range(num_show):
        axes[0, i].imshow(
            data_np[i].reshape(side, side),
            cmap="gray_r",
            interpolation="nearest",
        )
        axes[0, i].axis("off")
        axes[0, i].set_title(f"#{i}", fontsize=10)

        axes[1, i].imshow(
            recon_np[i].reshape(side, side),
            cmap="gray_r",
            interpolation="nearest",
        )
        axes[1, i].axis("off")

    fig.text(
        0.01,
        0.75,
        "Original",
        va="center",
        ha="left",
        rotation="vertical",
        fontsize=12,
    )
    fig.text(
        0.01,
        0.25,
        "Reconstruction",
        va="center",
        ha="left",
        rotation="vertical",
        fontsize=12,
    )

    fig.suptitle("Data vs One-Step Reconstructions", fontsize=14)

    plt.subplots_adjust(
        left=0.10,
        wspace=0.05,
        hspace=0.05,
    )

    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


def plot_free_running_samples(samples, side, filename, title):
    """Plot free-running samples from the model."""
    n_show = samples.shape[0]
    fig, axes = plt.subplots(1, n_show, figsize=(1.6 * n_show, 1.6))
    if n_show == 1:
        axes = [axes]

    for i in range(n_show):
        ax = axes[i]
        ax.imshow(
            samples[i].reshape(side, side),
            cmap="gray_r",
            interpolation="nearest",
        )
        ax.axis("off")
        ax.set_title(f"{i}", fontsize=8)

    fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved: {filename}")
    plt.close()


# ============================================================
# Main Entry Point
# ============================================================


def main():
    """Main training and evaluation pipeline."""

    # Data preparation
    side = 8
    data_np_all = make_bars_stripes(side)

    rng = np.random.RandomState(0)
    n_train = 256
    idx = rng.choice(len(data_np_all), size=n_train, replace=False)
    data_np = data_np_all[idx]

    n_samples, n_visible = data_np.shape
    print("Data shape:", data_np.shape)

    plot_dataset_samples(data_np, side)

    data = jnp.array(data_np, dtype=jnp.bool_)

    # Initialize RBM
    key = jax.random.key(0)
    n_hidden = 128

    rbm = RBM(
        n_visible=n_visible,
        n_hidden=n_hidden,
        learning_rate=0.05,
        n_chains_pos=4,
        n_chains_neg=32,
    )

    key = rbm.initialize_parameters(key)
    print(f"Nodes: {len(rbm.nodes)}, Edges: {len(rbm.edges)}")

    # Training
    n_epochs = 200
    key, metrics = rbm.train(key, data, n_epochs)

    v_recon_samples = metrics["v_recon_samples"][-1]

    mse_history, bce_history, weight_history = plot_training_curves(metrics)

    for epoch in range(0, n_epochs, 10):
        print(
            f"Epoch {epoch:3d} | recon MSE={mse_history[epoch]:.4f}  "
            f"BCE={bce_history[epoch]:.4f}  ||W||={weight_history[epoch]:.3f}"
        )
    if (n_epochs - 1) % 10 != 0:
        print(
            f"Epoch {n_epochs-1:3d} | recon MSE={mse_history[-1]:.4f}  "
            f"BCE={bce_history[-1]:.4f}  ||W||={weight_history[-1]:.3f}"
        )

    # Visualization
    plot_hidden_filters(rbm.weights, side, n_visible, n_hidden)

    recon_np = np.array(v_recon_samples.astype(jnp.float32))
    plot_reconstructions(data_np, recon_np, side)

    # Free-running sampling with THRML
    n_model_samples = 16
    key, thrml_samples, thrml_elapsed = rbm.sample_free_running(key, n_model_samples)
    print(f"THRML free-running sampling elapsed: {thrml_elapsed:.4f} s")

    plot_free_running_samples(
        thrml_samples,
        side,
        "./images/05_free_running_samples.png",
        "Free-running THRML RBM samples",
    )

    # Baseline Python sampling
    free_schedule = SamplingSchedule(
        n_warmup=300,
        n_samples=n_model_samples,
        steps_per_sample=20,
    )

    biases_np = np.array(rbm.biases)
    weights_np = np.array(rbm.weights)
    beta_scalar = float(rbm.beta)

    t0 = time.perf_counter()
    python_samples = gibbs_python_baseline(
        biases_np,
        weights_np,
        beta_scalar,
        n_visible,
        n_hidden,
        warmup=free_schedule.n_warmup,
        n_samples=n_model_samples,
        steps_per_sample=free_schedule.steps_per_sample,
        n_chains=1,
        seed=0,
    )
    python_elapsed = time.perf_counter() - t0
    print(f"Naive Python Gibbs sampling elapsed: {python_elapsed:.4f} s")
    if thrml_elapsed > 0:
        print(
            f"Speed ratio (Python / THRML): {python_elapsed / thrml_elapsed:.2f}x slower"
        )

    python_vis_samples = python_samples[:, 0, :]
    plot_free_running_samples(
        python_vis_samples,
        side,
        "./images/06_free_running_python_samples.png",
        "Free-running naive Python RBM samples",
    )

    print("\n" + "=" * 60)
    print("Training complete! Generated visualizations:")
    print("  01_dataset_samples.png             - Original bars & stripes dataset")
    print("  02_training_curves.png             - MSE, BCE, and weight norm over time")
    print("  03_hidden_filters.png              - Learned hidden unit filters")
    print("  04_reconstructions.png             - Original vs reconstructed samples")
    print("  05_free_running_samples.png        - Free-running THRML RBM samples")
    print(
        "  06_free_running_python_samples.png - Free-running naive Python RBM samples"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
