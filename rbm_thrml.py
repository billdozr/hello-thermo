"""
THRML-Powered Restricted Boltzmann Machine (RBM)
Implements an RBM using THRML's efficient Gibbs sampling for the bars-and-stripes dataset.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt

from thrml.pgm import SpinNode
from thrml.block_management import Block
from thrml.block_sampling import (
    BlockGibbsSpec,
    BlockSamplingProgram,
    SamplingSchedule,
    sample_states,
    sample_with_observation,
    SuperBlock,
)
from thrml.models.discrete_ebm import SpinEBMFactor, SpinGibbsConditional
from thrml.factor import FactorSamplingProgram
from thrml.observers import AbstractObserver


class FinalStateObserver(AbstractObserver):
    """Simple observer that just returns the final state of free blocks."""

    def init(self):
        return None

    def __call__(self, program, state_free, state_clamp, carry, step):
        # Just return the current state as-is
        return carry, state_free


def generate_bars_and_stripes(n_samples: int, image_size: int = 4, seed: int = 42) -> jnp.ndarray:
    """
    Generate the bars-and-stripes dataset.

    Args:
        n_samples: Number of samples to generate
        image_size: Size of the square images (default 4x4 = 16 pixels)
        seed: Random seed for reproducibility

    Returns:
        Binary images of shape (n_samples, image_size * image_size) with values in {-1, +1}
    """
    rng = np.random.RandomState(seed)
    data = []

    for _ in range(n_samples):
        if rng.rand() < 0.5:
            # Generate horizontal bars
            bar_idx = rng.randint(0, image_size)
            img = np.zeros((image_size, image_size))
            img[bar_idx, :] = 1
        else:
            # Generate vertical stripes
            stripe_idx = rng.randint(0, image_size)
            img = np.zeros((image_size, image_size))
            img[:, stripe_idx] = 1

        data.append(img.flatten())

    # Convert to JAX array with {-1, +1} values (THRML SpinNode convention)
    data = jnp.array(data)
    data = 2 * data - 1  # Convert {0, 1} to {-1, +1}

    return data


def visualize_bars_and_stripes(data: jnp.ndarray, n_examples: int = 10, image_size: int = 4):
    """Visualize some examples from the bars-and-stripes dataset."""
    fig, axes = plt.subplots(1, min(n_examples, len(data)), figsize=(12, 2))

    for i in range(min(n_examples, len(data))):
        img = data[i].reshape(image_size, image_size)
        # Convert back to {0, 1} for visualization
        img = (img + 1) / 2

        if n_examples == 1:
            ax = axes
        else:
            ax = axes[i]
        ax.imshow(img, cmap='gray', vmin=0, vmax=1)
        ax.axis('off')

    plt.suptitle('Bars and Stripes Dataset Examples')
    plt.tight_layout()
    plt.savefig('bars_and_stripes_examples.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to bars_and_stripes_examples.png")
    plt.close()


class RBM:
    """
    Restricted Boltzmann Machine using THRML's efficient Gibbs sampling.

    The RBM has an energy function:
    E(v, h) = -v^T W h - a^T v - b^T h

    where v are visible units, h are hidden units, W is the weight matrix,
    a are visible biases, and b are hidden biases.
    """

    def __init__(self, n_visible: int, n_hidden: int, seed: int = 42):
        """
        Initialize RBM architecture.

        Args:
            n_visible: Number of visible units
            n_hidden: Number of hidden units
            seed: Random seed for weight initialization
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # Create SpinNodes for visible and hidden units
        self.visible_nodes = [SpinNode() for _ in range(n_visible)]
        self.hidden_nodes = [SpinNode() for _ in range(n_hidden)]

        # Initialize weights and biases (small random values)
        key = jax.random.key(seed)
        key_w, key_a, key_b = jax.random.split(key, 3)

        # Weight matrix: shape (n_visible, n_hidden)
        # Small random initialization following typical RBM practice
        self.W = jax.random.normal(key_w, (n_visible, n_hidden)) * 0.01

        # Visible biases
        self.a = jax.random.normal(key_a, (n_visible,)) * 0.01

        # Hidden biases
        self.b = jax.random.normal(key_b, (n_hidden,)) * 0.01

        # Create blocks for block Gibbs sampling
        # Visible and hidden units form two separate blocks that we alternate between
        self.visible_block = Block(self.visible_nodes)
        self.hidden_block = Block(self.hidden_nodes)

        # Set up the block specification
        self.node_shape_dtypes = {SpinNode: jax.ShapeDtypeStruct((), jnp.bool_)}

    @property
    def factors(self):
        """Build the energy function factors for the RBM."""
        # For an RBM, we need:
        # 1. Visible bias terms: -a^T v
        # 2. Hidden bias terms: -b^T h
        # 3. Interaction terms: -v^T W h

        # Bias factors (linear terms)
        visible_bias_factor = SpinEBMFactor(
            node_groups=[self.visible_block],
            weights=self.a
        )

        hidden_bias_factor = SpinEBMFactor(
            node_groups=[self.hidden_block],
            weights=self.b
        )

        # Pairwise interaction factor: -v^T W h
        # We need to create edges connecting each visible node to each hidden node
        # Similar to Ising model, we need two blocks of equal size representing edges
        visible_edges = []
        hidden_edges = []
        edge_weights = []

        for i, v_node in enumerate(self.visible_nodes):
            for j, h_node in enumerate(self.hidden_nodes):
                visible_edges.append(v_node)
                hidden_edges.append(h_node)
                edge_weights.append(self.W[i, j])

        interaction_factor = SpinEBMFactor(
            node_groups=[Block(visible_edges), Block(hidden_edges)],
            weights=jnp.array(edge_weights)
        )

        return [visible_bias_factor, hidden_bias_factor, interaction_factor]

    def create_sampling_program(
        self,
        free_blocks: list[SuperBlock],
        clamped_blocks: list[Block]
    ) -> FactorSamplingProgram:
        """
        Create a FactorSamplingProgram for the RBM.

        Args:
            free_blocks: List of super blocks that are free to vary
            clamped_blocks: List of blocks that are held fixed

        Returns:
            A FactorSamplingProgram configured for this RBM
        """
        # Create the Gibbs conditional sampler for spin nodes
        sampler = SpinGibbsConditional()

        # Create the block specification
        spec = BlockGibbsSpec(free_blocks, clamped_blocks, self.node_shape_dtypes)

        # Create one sampler per free block
        samplers = [sampler for _ in spec.free_blocks]

        # Build the sampling program
        return FactorSamplingProgram(spec, samplers, self.factors, [])

    def sample_given_visible(
        self,
        key: jax.random.PRNGKey,
        visible_data: jnp.ndarray,
        n_steps: int = 1
    ) -> jnp.ndarray:
        """
        Sample hidden units given visible units (for positive phase).

        Args:
            key: JAX random key
            visible_data: Visible unit states, shape (batch_size, n_visible)
            n_steps: Number of Gibbs steps (usually 1 for positive phase)

        Returns:
            Hidden unit states, shape (batch_size, n_hidden)
        """
        # Create a sampling program where visible units are clamped
        # SuperBlock is a type alias for Block | tuple[Block, ...], so we pass Block directly
        program = self.create_sampling_program(
            free_blocks=[self.hidden_block],
            clamped_blocks=[self.visible_block]
        )

        # Initialize hidden states randomly
        batch_size = visible_data.shape[0]
        key_init, key_sample = jax.random.split(key)
        init_hidden = jax.random.bernoulli(
            key_init, 0.5, shape=(batch_size, self.n_hidden)
        ).astype(jnp.bool_)

        # Run sampling with custom observer
        schedule = SamplingSchedule(n_warmup=0, n_samples=1, steps_per_sample=n_steps)
        observer = FinalStateObserver()
        _, final_states = sample_with_observation(
            key_sample,
            program,
            schedule,
            init_chain_state=[init_hidden],
            state_clamp=[visible_data],
            observation_carry_init=observer.init(),
            f_observe=observer
        )

        # final_states is [state_free] from the last sample
        # state_free is a list with one element (the hidden block state)
        # Shape is (batch_size, n_hidden) but wrapped in sample dimension
        # Since we only have 1 sample (n_samples=1), we can just take the first one
        result = final_states[0]  # Extract the hidden block state
        # result has shape (1, batch_size, n_hidden), remove sample dim
        return result[0]  # Shape: (batch_size, n_hidden)

    def sample_given_hidden(
        self,
        key: jax.random.PRNGKey,
        hidden_data: jnp.ndarray,
        n_steps: int = 1
    ) -> jnp.ndarray:
        """
        Sample visible units given hidden units (for negative phase).

        Args:
            key: JAX random key
            hidden_data: Hidden unit states, shape (batch_size, n_hidden)
            n_steps: Number of Gibbs steps

        Returns:
            Visible unit states, shape (batch_size, n_visible)
        """
        # Create a sampling program where hidden units are clamped
        program = self.create_sampling_program(
            free_blocks=[self.visible_block],
            clamped_blocks=[self.hidden_block]
        )

        # Initialize visible states randomly
        batch_size = hidden_data.shape[0]
        key_init, key_sample = jax.random.split(key)
        init_visible = jax.random.bernoulli(
            key_init, 0.5, shape=(batch_size, self.n_visible)
        ).astype(jnp.bool_)

        # Run sampling with custom observer
        schedule = SamplingSchedule(n_warmup=0, n_samples=1, steps_per_sample=n_steps)
        observer = FinalStateObserver()
        _, final_states = sample_with_observation(
            key_sample,
            program,
            schedule,
            init_chain_state=[init_visible],
            state_clamp=[hidden_data],
            observation_carry_init=observer.init(),
            f_observe=observer
        )

        # final_states is [state_free] from the last sample
        # state_free is a list with one element (the visible block state)
        result = final_states[0]  # Extract the visible block state
        # result has shape (1, batch_size, n_visible), remove sample dim
        return result[0]  # Shape: (batch_size, n_visible)

    def cd_k_step(
        self,
        key: jax.random.PRNGKey,
        data: jnp.ndarray,
        k: int = 1,
        learning_rate: float = 0.01
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Perform one step of Contrastive Divergence-k (CD-k) training.

        Args:
            key: JAX random key
            data: Visible data, shape (batch_size, n_visible), boolean values
            k: Number of Gibbs sampling steps
            learning_rate: Learning rate for parameter updates

        Returns:
            Tuple of (new_W, new_a, new_b) - updated parameters
        """
        batch_size = data.shape[0]

        # Positive phase: clamp visible to data, sample hidden
        key_pos, key_neg = jax.random.split(key)
        h_pos = self.sample_given_visible(key_pos, data, n_steps=1)

        # Negative phase: start from h_pos, alternate k times
        h_neg = h_pos
        for i in range(k):
            key_neg, key_step = jax.random.split(key_neg)
            key_v, key_h = jax.random.split(key_step)
            v_neg = self.sample_given_hidden(key_v, h_neg, n_steps=1)
            h_neg = self.sample_given_visible(key_h, v_neg, n_steps=1)

        # Final negative sample
        key_neg, key_final = jax.random.split(key_neg)
        v_neg = self.sample_given_hidden(key_final, h_neg, n_steps=1)

        # Convert boolean to {-1, +1} for outer products
        # (THRML uses boolean, but for gradient calculation we need {-1, +1})
        data_spin = 2.0 * data.astype(jnp.float32) - 1.0
        h_pos_spin = 2.0 * h_pos.astype(jnp.float32) - 1.0
        v_neg_spin = 2.0 * v_neg.astype(jnp.float32) - 1.0
        h_neg_spin = 2.0 * h_neg.astype(jnp.float32) - 1.0

        # Compute positive and negative phase statistics
        # Positive phase: E_data[v_i h_j]
        pos_weights = jnp.einsum('bi,bj->ij', data_spin, h_pos_spin) / batch_size
        pos_visible_bias = jnp.mean(data_spin, axis=0)
        pos_hidden_bias = jnp.mean(h_pos_spin, axis=0)

        # Negative phase: E_model[v_i h_j]
        neg_weights = jnp.einsum('bi,bj->ij', v_neg_spin, h_neg_spin) / batch_size
        neg_visible_bias = jnp.mean(v_neg_spin, axis=0)
        neg_hidden_bias = jnp.mean(h_neg_spin, axis=0)

        # CD-k gradient: positive - negative
        grad_W = pos_weights - neg_weights
        grad_a = pos_visible_bias - neg_visible_bias
        grad_b = pos_hidden_bias - neg_hidden_bias

        # Update parameters (gradient ascent on log-likelihood, which is negative energy)
        # Note: Our energy has negative signs, so gradient descent = gradient ascent
        new_W = self.W + learning_rate * grad_W
        new_a = self.a + learning_rate * grad_a
        new_b = self.b + learning_rate * grad_b

        return new_W, new_a, new_b

    def train(
        self,
        key: jax.random.PRNGKey,
        data: jnp.ndarray,
        n_epochs: int = 10,
        batch_size: int = 10,
        k: int = 1,
        learning_rate: float = 0.01,
        verbose: bool = True
    ):
        """
        Train the RBM using Contrastive Divergence.

        Args:
            key: JAX random key
            data: Training data, shape (n_samples, n_visible), values in {-1, +1}
            n_epochs: Number of training epochs
            batch_size: Batch size for mini-batch training
            k: Number of Gibbs steps for CD-k
            learning_rate: Learning rate
            verbose: Whether to print progress
        """
        n_samples = data.shape[0]
        n_batches = n_samples // batch_size

        # Convert data from {-1, +1} to {0, 1} boolean
        data_bool = ((data + 1) / 2).astype(jnp.bool_)

        for epoch in range(n_epochs):
            # Shuffle data
            key, key_shuffle = jax.random.split(key)
            perm = jax.random.permutation(key_shuffle, n_samples)
            data_shuffled = data_bool[perm]

            for batch_idx in range(n_batches):
                # Get batch
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                batch_data = data_shuffled[start_idx:end_idx]

                # Perform CD-k update
                key, key_cd = jax.random.split(key)
                self.W, self.a, self.b = self.cd_k_step(
                    key_cd, batch_data, k=k, learning_rate=learning_rate
                )

            if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
                print(f"Epoch {epoch + 1}/{n_epochs} complete")

    def visualize_filters(self, image_size: int = 4, filename: str = 'rbm_filters.png'):
        """
        Visualize the learned weight filters.

        Args:
            image_size: Size of the square images (e.g., 4 for 4x4)
            filename: Output filename for the visualization
        """
        n_filters = min(self.n_hidden, 16)  # Show up to 16 filters
        n_cols = min(8, n_filters)
        n_rows = (n_filters + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.5, n_rows * 1.5))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i in range(n_filters):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]

            # Get the weight vector for this hidden unit
            weight_vec = self.W[:, i]
            # Reshape to image
            weight_img = weight_vec.reshape(image_size, image_size)

            # Plot
            im = ax.imshow(weight_img, cmap='RdBu', vmin=-jnp.max(jnp.abs(weight_img)),
                          vmax=jnp.max(jnp.abs(weight_img)))
            ax.set_title(f'Filter {i}', fontsize=8)
            ax.axis('off')

        # Hide any unused subplots
        for i in range(n_filters, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved filters to {filename}")
        plt.close()

    def reconstruct(
        self,
        key: jax.random.PRNGKey,
        data: jnp.ndarray,
        n_steps: int = 1
    ) -> jnp.ndarray:
        """
        Reconstruct visible data from input data.

        Args:
            key: JAX random key
            data: Input data, shape (batch_size, n_visible), values in {-1, +1}
            n_steps: Number of Gibbs steps for reconstruction

        Returns:
            Reconstructed data, shape (batch_size, n_visible), boolean values
        """
        # Convert to boolean
        data_bool = ((data + 1) / 2).astype(jnp.bool_)

        # Sample hidden given visible
        key_h, key_v = jax.random.split(key)
        hidden = self.sample_given_visible(key_h, data_bool, n_steps=1)

        # Reconstruct visible from hidden
        reconstructed = self.sample_given_hidden(key_v, hidden, n_steps=n_steps)

        return reconstructed

    def visualize_reconstructions(
        self,
        key: jax.random.PRNGKey,
        data: jnp.ndarray,
        n_examples: int = 10,
        image_size: int = 4,
        filename: str = 'rbm_reconstructions.png'
    ):
        """
        Visualize original data vs reconstructions.

        Args:
            key: JAX random key
            data: Original data, shape (n_samples, n_visible), values in {-1, +1}
            n_examples: Number of examples to show
            image_size: Size of square images
            filename: Output filename
        """
        n_examples = min(n_examples, data.shape[0])

        # Get reconstructions
        recon = self.reconstruct(key, data[:n_examples], n_steps=1)

        # Convert to {0, 1} for visualization
        data_viz = (data[:n_examples] + 1) / 2
        recon_viz = recon.astype(jnp.float32)

        fig, axes = plt.subplots(2, n_examples, figsize=(n_examples * 1.5, 3))

        for i in range(n_examples):
            # Original
            img_orig = data_viz[i].reshape(image_size, image_size)
            axes[0, i].imshow(img_orig, cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_title('Original', fontsize=10)

            # Reconstruction
            img_recon = recon_viz[i].reshape(image_size, image_size)
            axes[1, i].imshow(img_recon, cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Reconstructed', fontsize=10)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved reconstructions to {filename}")
        plt.close()

    def reconstruction_error(
        self,
        key: jax.random.PRNGKey,
        data: jnp.ndarray
    ) -> float:
        """
        Compute average reconstruction error.

        Args:
            key: JAX random key
            data: Test data, shape (n_samples, n_visible), values in {-1, +1}

        Returns:
            Average binary cross-entropy reconstruction error
        """
        recon = self.reconstruct(key, data, n_steps=1)

        # Convert to {0, 1}
        data_binary = (data + 1) / 2
        recon_float = recon.astype(jnp.float32)

        # Binary cross-entropy (treating reconstruction as probabilities)
        # For binary values, this is just proportion of mismatched bits
        error = jnp.mean(jnp.abs(data_binary - recon_float))

        return float(error)


if __name__ == "__main__":
    # Test the dataset generator
    print("Generating bars-and-stripes dataset...")
    data = generate_bars_and_stripes(n_samples=100, image_size=4)
    print(f"Generated dataset shape: {data.shape}")
    print(f"Data range: [{data.min()}, {data.max()}]")
    print(f"Unique values: {jnp.unique(data)}")

    # Visualize some examples
    visualize_bars_and_stripes(data, n_examples=10, image_size=4)
    print("\nDataset generation complete!")

    # Test RBM initialization
    print("\nInitializing RBM...")
    rbm = RBM(n_visible=16, n_hidden=8)  # 4x4 images = 16 pixels
    print(f"RBM created with {rbm.n_visible} visible and {rbm.n_hidden} hidden units")
    print(f"Weight matrix shape: {rbm.W.shape}")
    print(f"Visible bias shape: {rbm.a.shape}")
    print(f"Hidden bias shape: {rbm.b.shape}")

    # Test sampling methods
    print("\nTesting sampling methods...")
    key = jax.random.key(42)

    # Convert data from {-1, +1} to {0, 1} boolean
    data_bool = ((data[:5] + 1) / 2).astype(jnp.bool_)

    # Test sampling hidden given visible
    key_h, key_v = jax.random.split(key)
    print(f"Input visible data shape: {data_bool.shape}")
    hidden_samples = rbm.sample_given_visible(key_h, data_bool, n_steps=1)
    print(f"Sampled hidden states shape: {hidden_samples.shape}")

    # Test sampling visible given hidden
    visible_samples = rbm.sample_given_hidden(key_v, hidden_samples, n_steps=1)
    print(f"Reconstructed visible states shape: {visible_samples.shape}")

    print("\nSampling tests complete!")

    # Test training with better hyperparameters
    print("\nTraining RBM...")
    key_train = jax.random.key(123)

    # Train for more epochs with a larger dataset and better learning rate
    print("Generating larger training dataset...")
    train_data = generate_bars_and_stripes(n_samples=200, image_size=4, seed=100)

    rbm.train(
        key_train,
        train_data,
        n_epochs=200,
        batch_size=20,
        k=5,
        learning_rate=0.05,
        verbose=True
    )
    print("\nTraining complete!")

    # Evaluate reconstruction error
    key_eval = jax.random.key(456)
    error = rbm.reconstruction_error(key_eval, train_data[:50])
    print(f"\nReconstruction error: {error:.4f}")

    # Visualize learned filters
    print("\nVisualizing learned filters...")
    rbm.visualize_filters(image_size=4, filename='rbm_filters.png')

    # Visualize reconstructions
    print("Visualizing reconstructions...")
    key_recon = jax.random.key(789)
    rbm.visualize_reconstructions(key_recon, train_data[:10], n_examples=10, image_size=4,
                                   filename='rbm_reconstructions.png')

    print("\n" + "="*60)
    print("THRML-Powered RBM Demo Complete!")
    print("="*60)
    print(f"✓ Trained RBM with {rbm.n_visible} visible and {rbm.n_hidden} hidden units")
    print(f"✓ Using THRML's hardware-efficient Gibbs sampling")
    print(f"✓ Final reconstruction error: {error:.4f}")
    print(f"✓ Generated visualizations:")
    print(f"  - bars_and_stripes_examples.png (dataset)")
    print(f"  - rbm_filters.png (learned filters)")
    print(f"  - rbm_reconstructions.png (reconstructions)")
    print("="*60)
