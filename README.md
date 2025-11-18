# THRML-Powered Restricted Boltzmann Machine

A Restricted Boltzmann Machine (RBM) implementation using the THRML library's hardware-efficient Gibbs sampling for the bars-and-stripes dataset.

## Overview

This project demonstrates training an RBM on the bars-and-stripes pattern recognition task using THRML's Ising model framework for efficient Gibbs sampling. The implementation showcases both THRML's accelerated sampling and includes a naive Python baseline for performance comparison.

## What is an RBM?

A Restricted Boltzmann Machine is an energy-based probabilistic graphical model with:
- **Visible layer**: Input data (64 pixels for 8×8 images)
- **Hidden layer**: Latent features (128 units)
- **Bipartite structure**: Connections only between layers, not within layers

The model learns to represent data patterns through an energy function formulated as an Ising model:
![Energy function](./images/energy_function.png)

## Running the Code

```bash
# Install dependencies
uv sync

# Run the RBM training
uv run python rbm.py
```

This will:
1. Generate the bars-and-stripes dataset
2. Train the RBM for 200 epochs (~1-2 minutes depending on hardware)
3. Generate all visualization plots in `./images/`
4. Print training progress and performance metrics

## Results

## Model Configuration

- **Visible units**: 64 (8×8 pixel grid)
- **Hidden units**: 128
- **Training data**: 256 bars-and-stripes patterns (subsampled from 508 total)
- **Inverse temperature (β)**: 1.0
- **Learning rate**: 0.05
- **Training epochs**: 200

## Key Components

### RBM Class

The main `RBM` class encapsulates the entire model:

**Initialization**
- Creates visible and hidden `SpinNode` objects for THRML's Ising model
- Defines fully-connected bipartite edges between layers
- Sets up sampling schedules for positive and negative phases
- Configurable learning rate and chain counts

**Training Methods**
- `initialize_parameters()` - Small random initialization for biases and weights
- `train()` - Main training loop using `jax.lax.scan` for efficiency
- `train_epoch()` - Single epoch with gradient estimation via THRML's `estimate_kl_grad()`
- `reconstruction_error()` - Compute MSE and BCE for monitoring training

**Sampling Methods**
- `sample_free_running()` - Generate unconditioned samples from the trained model

### Data Generation

`make_bars_stripes()` generates binary patterns with horizontal/vertical bars:
- Each pattern is a subset of rows (horizontal) or columns (vertical) turned ON
- Excludes all-black and all-white patterns
- Returns 8×8 flattened boolean arrays

### Conditional Sampling Functions

Implements exact Ising conditionals for block Gibbs sampling:
- `sample_hidden_given_visible()` - P(h|v) using local fields
- `sample_visible_given_hidden()` - P(v|h) using local fields
- `bernoulli_from_field()` - Bernoulli sampling with probability σ(2βh)

### Baseline Sampler

`gibbs_python_baseline()` provides a naive NumPy implementation:
- Manual alternating updates of hidden then visible layers
- Same sampling schedule as THRML for fair comparison
- Pure Python loops without JIT compilation
- Used for performance benchmarking

### Visualization Functions

Five plotting functions generate training insights:
- `plot_dataset_samples()` - Sample training data
- `plot_training_curves()` - MSE, BCE, and weight norm over epochs
- `plot_hidden_filters()` - Learned feature detectors (weights as 8×8 images)
- `plot_reconstructions()` - Original vs reconstructed samples
- `plot_free_running_samples()` - Model-generated patterns

## Training Strategy

**Positive Phase** (data-driven):
- Visible units clamped to training data
- 4 parallel chains sampling hidden units
- Schedule: 10 warmup steps, 10 samples, 1 step between samples

**Negative Phase** (model-driven):
- Both visible and hidden units free
- 32 parallel fantasy chains
- Schedule: 20 warmup steps, 10 samples, 2 steps between samples

**Gradient Estimation**:
Uses Contrastive Divergence via THRML's `estimate_kl_grad()` to compute KL divergence gradients with respect to biases and weights.

## Output Visualizations

The script generates six visualizations in `./images/`:

### Training Data

Sample bars-and-stripes patterns used for training:

![Bars & Stripes Dataset](./images/01_dataset_samples.png)

### Training Progress

MSE, BCE, and weight norm over 200 epochs showing convergence:

![Training Curves](./images/02_training_curves.png)

### Learned Hidden Filters

The 128 hidden units learn feature detectors for vertical bars, horizontal bars, and composite patterns:

![Hidden Filters](./images/03_hidden_filters.png)

### Reconstruction Quality

Original data (top row) vs one-step reconstructions (bottom row):

![Reconstructions](./images/04_reconstructions.png)

### All Outputs

1. `01_dataset_samples.png` - Sample bars-and-stripes training data
2. `02_training_curves.png` - MSE, BCE, and weight norm over 200 epochs
3. `03_hidden_filters.png` - 128 learned feature detectors (weights reshaped to 8×8)
4. `04_reconstructions.png` - Original vs one-step reconstructed images
5. `05_free_running_samples.png` - 16 patterns generated by THRML sampler
6. `06_free_running_python_samples.png` - 16 patterns from Python baseline

## Performance Comparison

The implementation includes timing benchmarks comparing THRML's hardware-optimized Gibbs sampling against a naive Python implementation. THRML leverages JAX's JIT compilation and XLA optimization for significant speedup.

Expected output:
```
THRML free-running sampling elapsed: X.XXXX s
Naive Python Gibbs sampling elapsed: Y.YYYY s
Speed ratio (Python / THRML): ZZ.ZZx slower
```

## How It Works

### Energy-Based Learning

The RBM learns by adjusting its energy landscape to:
- **Lower energy** for observed data patterns (positive phase)
- **Raise energy** for model-generated patterns (negative phase)

This is achieved through gradient descent on the KL divergence between data and model distributions.

### Ising Model Formulation

THRML represents the RBM as an Ising spin glass:
- Boolean pixels {0,1} map to spins {-1,+1}
- Biases become local fields
- Weights become coupling constants
- Exact conditional distributions enable efficient block Gibbs sampling

### Block Gibbs Sampling

Since layers are conditionally independent given the other layer:
- All hidden units can be sampled in parallel given visible states
- All visible units can be sampled in parallel given hidden states
- This bipartite structure enables efficient hardware parallelization

### JAX Optimization

The training loop uses `jax.lax.scan` for efficiency:
- Parameters are passed through a carry dictionary during training
- Instance variables (`self.biases`, `self.weights`) are updated only after training completes
- This functional approach enables JAX's JIT compilation and automatic differentiation

## Requirements

- Python ≥3.11
- JAX
- NumPy
- Matplotlib
- THRML library
